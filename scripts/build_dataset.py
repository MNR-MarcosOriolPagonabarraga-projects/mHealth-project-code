import argparse
from pathlib import Path
import numpy as np
from src.process import PatientProcessor
from src.config import PreprocessConfig

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

def main():
    cfg = PreprocessConfig()
    processor = PatientProcessor(cfg)

    # Find all patient directories (e.g., 0005, 0029)
    patient_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    
    print(f"Found {len(patient_dirs)} patients. Starting processing...")

    for p_dir in patient_dirs:
        print(f"Processing {p_dir.name}...")
        
        # Keep the try/except ONLY here at the script level, so one bad patient 
        # doesn't crash the whole multi-hour batch job.
        try:
            features = processor(p_dir)
            
            out_file = OUT_DIR / f"tr03-{p_dir.name}_preprocessed.npz"
            np.savez_compressed(out_file, **features)
            
            print(f"  -> Saved {len(features['labels'])} windows to {out_file.name}")
            
        except Exception as e:
            print(f"  [!] Failed to process {p_dir.name}: {e}")

if __name__ == "__main__":
    main()