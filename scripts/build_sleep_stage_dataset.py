import random
from pathlib import Path
import numpy as np

from src.utils import load_signals_and_arousals
from src.config import PreprocessConfig


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SLEEP_STAGES = {
    0 : "wakefulness", 
    1 : "s1", 
    2 : "s2", 
    3 : "s3", 
    4 : "REM", 
    5 : "undefined"
}

def get_stage_signal(signals, sleep_stages, stage):
    stage_mask = np.where(sleep_stages == stage)
    return signals[stage_mask]

def extract_windows(signals, sleep_stages):
    for stage in SLEEP_STAGES:
        pass



def process_and_aggregate(patient_dirs, processor, split_name):
    """Processes a list of patients, balances them, and saves a single .npz file."""
    all_contexts = []
    all_sleep_stages = []
    
    # We only need to save these once
    fs = None
    ch_names = None
    
    print(f"\n--- Processing {split_name.upper()} Set ({len(patient_dirs)} patients) ---")
    
    for p_dir in patient_dirs:
        print(f"  -> {p_dir.name}...", end=" ")
        eeg_signals, _, sleep_stages = load_signals_and_arousals(p_dir.name, include_sleep_stages=True)



    
    # 5. Save to disk
    out_path = OUT_DIR / f"{split_name}.npz"
    np.savez_compressed(
        out_path, 
        context_windows=final_contexts,
        sleep_stages = final_sleep_stages, 
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