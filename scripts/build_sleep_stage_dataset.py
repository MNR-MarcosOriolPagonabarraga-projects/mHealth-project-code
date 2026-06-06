import random
from pathlib import Path
import numpy as np

from src.dsp import compute_full_recording_bandpower, apply_continuous_filters
from src.utils import load_signals_and_arousals, find_stable_blocks, downsample_signal
from src.config import PreprocessConfig

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

def extract_windows(full_spectral_timeline: np.ndarray, sleep_stages: dict, fs: int, hop_length: int, win_sec: int = 30):
    """
    Extracts 30-second spectral windows for valid sleep stages from a dictionary of masks.
    """
    win_samp = win_sec * fs
    step_samp = win_samp # 30s step for no overlap
    
    all_features = []
    all_labels = []
    
    # Map your integer labels to the exact keys in your loaded dictionary
    STAGE_KEY_MAP = {
        0: 'wake',
        1: 'nonrem1',
        2: 'nonrem2',
        3: 'nonrem3',
        4: 'rem'
    }
    
    for stage_int, dict_key in STAGE_KEY_MAP.items():
        # Ensure the key exists in the patient's data
        if dict_key not in sleep_stages:
            continue
            
        # Extract the boolean mask array for this specific stage
        mask = sleep_stages[dict_key]
        
        # 1. Get contiguous blocks for this stage
        blocks = find_stable_blocks(mask, win_samp)
        
        # 2. Extract consecutive 30-second windows
        for on, off in blocks:
            for start_idx in range(on, off - win_samp + 1, step_samp):
                end_idx = start_idx + win_samp
                
                # Convert raw sample indices to STFT step indices
                start_step_idx = start_idx // hop_length
                end_step_idx = end_idx // hop_length
                
                # Slice STFT Features
                spec_win = full_spectral_timeline[start_step_idx:end_step_idx, :]
                
                all_features.append(spec_win)
                all_labels.append(stage_int)
                
    return all_features, all_labels

def process_and_aggregate(patient_dirs, cfg, split_name):
    """Processes a list of patients, balances them, and saves a single .npz file."""
    all_spectral_band_windows = []
    all_sleep_stages = []
    
    # We only need to save these once
    fs = None
    ch_names = None
    channels = list(cfg.eeg_indices.values())
    
    print(f"\n--- Processing {split_name.upper()} Set ({len(patient_dirs)} patients) ---")
    
    for p_dir in patient_dirs:
        print(f"  -> {p_dir.name}...", end=" ")
        
        # Load Data
        eeg_signals, _, sleep_stages = load_signals_and_arousals(p_dir, channels, include_sleep_stages=True)
        eeg_signals = downsample_signal(eeg_signals, downsample_factor=2)
        eeg_signals = np.mean(eeg_signals, axis=0)
        eeg_signals = apply_continuous_filters(eeg_signals, cfg)
        sleep_stages = {key : values[::2] for key, values in sleep_stages.items()}
        
        # Pull metadata from your processor/config
        # (Adjust these attributes based on how your PreprocessConfig is actually structured)
        current_fs = 100
        hop_length = 100
        
        if fs is None:
            fs = current_fs
            ch_names = cfg.ch_names if hasattr(cfg, 'ch_names') else ["Ch1", "Ch2"]

        full_spectral_timeline = compute_full_recording_bandpower(eeg_signals, fs=fs, hop_length=hop_length)
        
        # Extract Windows & Labels
        patient_features, patient_labels = extract_windows(
            full_spectral_timeline=full_spectral_timeline, 
            sleep_stages=sleep_stages, 
            fs=fs, 
            hop_length=hop_length, 
            win_sec=30
        )
        
        if len(patient_features) > 0:
            all_spectral_band_windows.append(np.array(patient_features))
            all_sleep_stages.append(np.array(patient_labels))
            print(f"Extracted {len(patient_features)} windows.")
        else:
            print("No valid 30s windows found.")

    # Aggregate all patients into massive master arrays
    print(f"\nAggregating {split_name} data...")
    if len(all_spectral_band_windows) > 0:
        final_spectral_band_windows = np.concatenate(all_spectral_band_windows, axis=0)
        labels_int = np.concatenate(all_sleep_stages, axis=0)
        
        # Apply One-Hot Encoding to the final aggregated labels (excluding 'undefined')
        num_classes = 5 
        final_sleep_stages = np.eye(num_classes)[labels_int]
    else:
        print(f"WARNING: No data extracted for {split_name} set!")
        final_spectral_band_windows = np.array([])
        final_sleep_stages = np.array([])
    
    # Save to disk
    out_path = OUT_DIR / f"{split_name}.npz"
    np.savez_compressed(
        out_path, 
        spectral_band_windows=final_spectral_band_windows,
        sleep_stages=final_sleep_stages, 
        fs=fs, 
        ch_names=ch_names
    )
    print(f"[+] Saved {split_name} to {out_path.name} with shape {final_spectral_band_windows.shape}")

def main():
    cfg = PreprocessConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all patients
    patient_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    
    if not patient_dirs:
        print("No patient directories found in data/raw. Exiting.")
        return
        
    # Shuffle patients for random train/test split
    random.seed(42) # Set a seed for reproducible splits
    random.shuffle(patient_dirs)
    
    # 80/20 Split
    split_idx = int(len(patient_dirs) * 0.8)
    train_patients = patient_dirs[:split_idx]
    test_patients = patient_dirs[split_idx:]
    
    # Process and build files
    process_and_aggregate(train_patients, cfg, "sleep_stage_train")
    process_and_aggregate(test_patients, cfg, "sleep_stage_test")

if __name__ == "__main__":
    main()