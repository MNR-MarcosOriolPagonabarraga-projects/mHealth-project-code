import random
from pathlib import Path
import numpy as np

from offline.dsp import compute_full_recording_bandpower, apply_continuous_filters
from src.utils import load_signals_and_arousals, find_stable_blocks, downsample_signal
from offline.config import PreprocessConfig

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

def extract_windows(full_spectral_timeline: np.ndarray, sleep_stages: dict, fs: int, hop_length: int, win_sec: int = 30):
    """
    Extracts 30-second spectral windows for valid sleep stages.
    Merges N1 and N2 into a single 'Light Sleep' class (Label 1).
    """
    win_samp = win_sec * fs
    step_samp = win_samp # 30s step for no overlap
    
    all_features = []
    all_labels = []
    
    # Updated Map: Arrays of keys allow us to merge multiple stages into one integer label
    STAGE_KEY_MAP = {
        0: ['wake'],
        1: ['nonrem1', 'nonrem2'], # Merged N1 and N2 into "Light Sleep"
        2: ['nonrem3'],            # Deep Sleep
        3: ['rem']
    }
    
    for stage_int, dict_keys in STAGE_KEY_MAP.items():
        
        # Combine the boolean masks for all keys in this group
        mask = None
        for key in dict_keys:
            if key in sleep_stages:
                if mask is None:
                    mask = sleep_stages[key].copy()
                else:
                    # Bitwise OR merges the 1s from both masks
                    mask = mask | sleep_stages[key] 
                    
        # If none of the keys existed for this patient, skip
        if mask is None:
            continue
            
        # 1. Get contiguous blocks for this combined mask
        blocks = find_stable_blocks(mask, win_samp)
        
        # 2. Extract consecutive 30-second windows
        for on, off in blocks:
            for start_idx in range(on + win_samp, off - (2 * win_samp) + 1, step_samp):
                
                curr_on, curr_off = start_idx, start_idx + win_samp
                past_on, past_off = start_idx - win_samp, start_idx
                fut_on, fut_off = start_idx + win_samp, start_idx + (2 * win_samp)
                
                c_start, c_end = curr_on // hop_length, curr_off // hop_length
                p_start, p_end = past_on // hop_length, past_off // hop_length
                f_start, f_end = fut_on // hop_length, fut_off // hop_length
                
                win_curr = full_spectral_timeline[c_start:c_end, :] 
                win_past = full_spectral_timeline[p_start:p_end, :] 
                win_fut  = full_spectral_timeline[f_start:f_end, :]  
                
                contextual_window = np.concatenate([win_past, win_curr, win_fut], axis=1)
                
                all_features.append(contextual_window)
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
        eeg_signals = apply_continuous_filters(eeg_signals, cfg)
        sleep_stages = {key : values[::2] for key, values in sleep_stages.items()}
        
        # Pull metadata from your processor/config
        # (Adjust these attributes based on how your PreprocessConfig is actually structured)
        current_fs = 100
        hop_length = 25
        n_fft = 200
        
        if fs is None:
            fs = current_fs
            ch_names = cfg.ch_names if hasattr(cfg, 'ch_names') else ["Ch1", "Ch2"]

        full_spectral_timeline = compute_full_recording_bandpower(eeg_signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
        full_spectral_timeline = zscore_norm(full_spectral_timeline)
        
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

    print(f"\nAggregating {split_name} data...")
    if len(all_spectral_band_windows) > 0:
        final_spectral_band_windows = np.concatenate(all_spectral_band_windows, axis=0)
        labels_int = np.concatenate(all_sleep_stages, axis=0)
        
        # Apply One-Hot Encoding to the final aggregated labels
        num_classes = 4
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

def zscore_norm(signals):
    means = np.mean(signals, axis=0)
    stds = np.std(signals, axis=0) + 1e-8
    normalized_signals = (signals - means) / stds

    return normalized_signals

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