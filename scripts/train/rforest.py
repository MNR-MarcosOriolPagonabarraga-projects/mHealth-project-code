import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def get_dataset_metadata(npz_path: Path):
    """Safely opens the npz using a memory map to read array shapes without loading data."""
    with np.load(npz_path, mmap_mode='r') as data:
        # Number of windows is the first outer dimension of the label tensor
        n_samples = data['labels'].shape[0]
        _, channels, bins = data['psd'].shape
        n_features = channels * bins
    return n_samples, n_features

def batch_generator(npz_path: Path, batch_size: int = 4096):
    """Streaming generator that yields small, memory-safe mini-batches directly from the disk."""
    data = np.load(npz_path, mmap_mode='r')
    psd_mmap = data['psd']
    labels_mmap = data['labels']
    n_samples = labels_mmap.shape[0]

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        X_batch = psd_mmap[i:end_idx]
        y_batch = labels_mmap[i:end_idx]  # Shape: (batch_size, 3000)
        
        # Flatten feature maps on the fly: (batch_size, 2, 159) -> (batch_size, 318)
        X_batch_flat = X_batch.reshape(X_batch.shape[0], -1)
        
        # --- CRITICAL FIX: Down-convert dense timeline labels to 1 target per window ---
        # Calculate what percentage of each 30-second window is occupied by an arousal event
        # If more than 10% of the 3000 samples are 1, class is 1. Else 0.
        arousal_ratios = np.sum(y_batch == 1, axis=1) / y_batch.shape[1]
        y_batch_windowed = (arousal_ratios > 0.1).astype(np.int64)
        
        yield X_batch_flat, y_batch_windowed

def main():
    print("[*] Initializing Incremental Random Forest Pipeline via Warm-Starting")
    
    train_path = Path("data/processed/train.npz")
    val_path = Path("data/processed/test.npz")
    
    n_train_samples, n_features = get_dataset_metadata(train_path)
    n_val_samples, _ = get_dataset_metadata(val_path)
    print(f"[*] Train set size: {n_train_samples:,} windows | Features per window: {n_features}")
    print(f"[*] Val set size:   {n_val_samples:,} windows")

    # 1. Compute scaling factors incrementally
    print("\n[*] Step 1: Fitting scaling parameters step-by-step...")
    scaler = StandardScaler()
    for X_batch, _ in batch_generator(train_path, batch_size=8192):
        scaler.partial_fit(X_batch)

    # 2. Define explicit class weights derived from global window structures
    # Total window layout ratio is roughly 12.5x imbalanced at sample level,
    # but when collapsed to windows, it aligns closer to an 8:1 ratio.
    # We pass 'balanced' string to Random Forest which handles it beautifully per batch natively.
    
    # 3. Initialize Random Forest with Warm-Starting
    rf = RandomForestClassifier(
        n_estimators=5,             # Start with 5 trees, we will append 5 more per batch
        max_depth=12,               # Capped depth prevents trees from expanding infinitely in RAM
        class_weight="balanced",    # Perfectly safe now that dimensions match 1:1 row-wise
        random_state=42,
        warm_start=True,
        n_jobs=-1
    )

    print("\n[*] Step 2: Training Random Forest incrementally over memory-mapped batches...")
    
    batch_count = 0
    for X_batch, y_batch in batch_generator(train_path, batch_size=4096):
        X_batch_scaled = scaler.transform(X_batch)
        
        # Fit the current batch
        rf.fit(X_batch_scaled, y_batch)
        
        # Increment tree allocation for the NEXT batch so the forest grows over time
        rf.n_estimators += 5
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"    -> Processed {batch_count} batches. Current Forest Size: {rf.n_estimators} trees.")

    # 4. Out-of-core evaluation on validation set
    print("\n[*] Step 3: Running inference on validation set...")
    all_preds = []
    all_targets = []
    
    for X_batch, y_batch in batch_generator(val_path, batch_size=8192):
        X_batch_scaled = scaler.transform(X_batch)
        preds = rf.predict(X_batch_scaled)
        
        all_preds.append(preds)
        all_targets.append(y_batch)
        
    y_val_true = np.concatenate(all_targets)
    y_val_pred = np.concatenate(all_preds)

    # 5. Output Final Results
    val_acc = accuracy_score(y_val_true, y_val_pred)
    val_recall = recall_score(y_val_true, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)

    print("\n" + "="*40)
    print("      INCREMENTAL WARM-START RANDOM FOREST METRICS")
    print("="*40)
    print(f"Accuracy : {val_acc*100:.1f}%")
    print(f"Recall   : {val_recall*100:.1f}%")
    print(f"F1 Score : {val_f1:.3f}")
    print("-" * 40)
    print("Detailed Classification Report:")
    print(classification_report(y_val_true, y_val_pred, target_names=["Non-Arousal (0)", "Arousal (1)"]))

if __name__ == "__main__":
    main()