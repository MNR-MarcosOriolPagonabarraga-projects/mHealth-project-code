import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def load_psd_data(npz_path: Path, undersample: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Loads npz data and flattens PSD arrays for Scikit-Learn."""
    data = np.load(npz_path)
    
    psd = data['psd']      # Expected shape: (N, channels, bins)
    labels = data['labels'] # Expected shape: (N,)
    
    # Flatten the channels and bins into a single 1D feature vector per window
    # Shape transforms from (N, 2, 159) -> (N, 318)
    X = psd.reshape(psd.shape[0], -1)
    y = labels

    if undersample:
        # Simple undersampling of the majority class (non-arousals)
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        n_pos = len(pos_indices)
        
        # Randomly select an equal number of negative samples
        np.random.seed(42) # For reproducibility
        selected_neg_indices = np.random.choice(neg_indices, size=n_pos, replace=False)
        
        # Combine and shuffle the selected indices
        combined_indices = np.concatenate((pos_indices, selected_neg_indices))
        np.random.shuffle(combined_indices)
        
        X = X[combined_indices]
        y = y[combined_indices]
    
    return X, y

def main():
    print("[*] Initializing Random Forest Pipeline")
    
    # 1. Load Data
    train_path = Path("data/processed/train.npz")
    val_path = Path("data/processed/test.npz")
    
    print(f"[*] Loading training data from {train_path}...")
    X_train, y_train = load_psd_data(train_path, undersample=True)
    
    print(f"[*] Loading validation data from {val_path}...")
    X_val, y_val = load_psd_data(val_path, undersample=True)
    
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    print(f"[*] Training Dataset -> Neg: {n_neg} | Pos: {n_pos}")
    print("[*] Flattened PSD Feature shape:", X_train.shape)

    # 2. Initialize Model
    # class_weight="balanced" automatically adjusts weights inversely proportional to class frequencies
    # n_jobs=-1 uses all available CPU cores for faster training
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=1 # Provides a built-in progress bar during training
    )

    # 3. Train Model
    print("\n[*] Training Random Forest... (This may take a moment)")
    rf.fit(X_train, y_train)

    # 4. Evaluate Model
    print("\n[*] Running inference on validation set...")
    y_pred = rf.predict(X_val)
    
    val_acc = accuracy_score(y_val, y_pred)
    val_recall = recall_score(y_val, y_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_pred, zero_division=0)

    # 5. Output Results
    print("\n" + "="*40)
    print("      RANDOM FOREST VALIDATION METRICS")
    print("="*40)
    print(f"Accuracy : {val_acc*100:.1f}%")
    print(f"Recall   : {val_recall*100:.1f}%")
    print(f"F1 Score : {val_f1:.3f}")
    print("-" * 40)
    print("Detailed Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Non-Arousal (0)", "Arousal (1)"]))

if __name__ == "__main__":
    main()