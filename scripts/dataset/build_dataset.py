import random
from pathlib import Path
import numpy as np
from src.process import PatientProcessor
from src.config import PreprocessConfig

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def process_and_aggregate(patient_dirs, processor, split_name):
    """Processes a list of patients, balances them, and saves a single .npz file."""
    all_contexts = []
    all_signals = []
    all_labels = []
    
    # We only need to save these once
    fs = None
    ch_names = None
    
    print(f"\n--- Processing {split_name.upper()} Set ({len(patient_dirs)} patients) ---")
    
    for p_dir in patient_dirs:
        print(f"  -> {p_dir.name}...", end=" ")

        # Extract features
        raw_features = processor(p_dir)
        
        # Store constants on the first successful patient
        if fs is None:
            fs = raw_features["fs"]
            ch_names = raw_features["ch_names"]

            
        # Store in RAM
        all_contexts.append(raw_features["context_windows"])
        all_signals.append(raw_features["eeg_windows"])
        all_labels.append(raw_features["labels"])
        
        print(f"Kept {len(raw_features['labels'])} windows.")

    # 4. Concatenate all patients into master arrays
    print(f"\nAggregating {split_name} data...")
    final_contexts = np.concatenate(all_contexts, axis=0)
    final_signals = np.concatenate(all_signals, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Total {split_name} shape: Signals: {final_contexts.shape}, Labels: {final_labels.shape}")
    
    # 5. Save to disk
    out_path = OUT_DIR / f"{split_name}.npz"
    np.savez_compressed(
        out_path, 
        eeg_windows=final_signals,
        context_windows=final_contexts,
        labels=final_labels, 
        fs=fs, 
        ch_names=ch_names
    )
    print(f"[+] Saved {split_name} to {out_path.name}")

def main():
    cfg = PreprocessConfig()
    processor = PatientProcessor(cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all patients
    patient_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    
    # Shuffle patients for random train/test split
    random.shuffle(patient_dirs)
    
    # 80/20 Split
    split_idx = int(len(patient_dirs) * 0.8)
    train_patients = patient_dirs[:split_idx]
    test_patients = patient_dirs[split_idx:]
    
    # Process and build files
    process_and_aggregate(train_patients, processor, "train")
    process_and_aggregate(test_patients, processor, "test")

if __name__ == "__main__":
    main()