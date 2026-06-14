import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.offline.config import PreprocessConfig
from src.offline.dsp import apply_causal_filters, compute_full_recording_bandpower, clip_outliers, downsample_all
from src.offline.windowing import extract_arousal_windows, extract_sleep_stage_windows
from src.offline.data_io import load_signals_and_annotations

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def process_patient(patient_dir: Path, cfg: PreprocessConfig) -> dict:
    """End-to-end extraction pipeline for a single patient."""
    # 1. Load Data
    signals, arousals, sleep_stages = load_signals_and_annotations(
        patient_dir, 
        channels=list(cfg.channels.values())
    )

    # 2. Continuous DSP
    signals = clip_outliers(signals, clip_threshold=cfg.clip_threshold)
    signals = apply_causal_filters(signals, cfg)
    
    signals, arousals, sleep_stages = downsample_all(
        cfg.downsample_factor, 
        signals, arousals, sleep_stages
    )
    
    # 3. Spectral Feature Extraction
    spectral_timeline = compute_full_recording_bandpower(
        signals, 
        fs=cfg.fs, 
        n_fft=cfg.n_fft, 
        hop_length=cfg.hop_length
    )
    
    # 4. Window Extraction
    arousal_data = extract_arousal_windows(
        signals, arousals, spectral_timeline, 
        fs=cfg.fs, 
        pre_sec=cfg.arousal_pre_sec, post_sec=cfg.arousal_post_sec, 
        ctx_sec=cfg.arousal_ctx_sec, hop_len=cfg.hop_length, n_fft=cfg.n_fft,
        neg_ratio=cfg.windows_neg_ratio
    )
    
    stage_data = extract_sleep_stage_windows(
        spectral_timeline, sleep_stages, 
        fs=cfg.fs, hop_length=cfg.hop_length, win_sec=cfg.sleep_win_sec
    )
    
    return {
        "arousal": arousal_data,
        "stages": stage_data,
        "fs": cfg.fs,
        "ch_names": list(cfg.channels.keys())
    }

def process_and_aggregate_split(patient_dirs: list[Path], cfg: PreprocessConfig, split_name: str):
    """Processes a list of patients and saves massive aggregated .npz files."""    
    # Arousal Accumulators
    arousal_eeg = []
    arousal_ctx = []
    arousal_lbl = []
    
    # Sleep Stage Accumulators
    stage_feats = []
    stage_lbl = []
    
    fs = None
    ch_names = None
    
    for p_dir in tqdm(patient_dirs):
        features = process_patient(p_dir, cfg)
        
        # Capture metadata from the first successful patient
        if fs is None:
            fs = features["fs"]
            ch_names = features["ch_names"]
            
        # Accumulate Arousals
        if len(features["arousal"]["labels"]) > 0:
            arousal_eeg.append(np.stack(features["arousal"]["eeg_windows"]))
            arousal_ctx.append(np.stack(features["arousal"]["context_windows"]))
            arousal_lbl.append(np.array(features["arousal"]["labels"], dtype=np.int8))
            
        # Accumulate Sleep Stages
        if len(features["stages"]["labels"]) > 0:
            stage_feats.append(np.stack(features["stages"]["sleep_features"]))
            stage_lbl.append(np.array(features["stages"]["labels"], dtype=np.int8))
            
    # Concatenate and Save Arousals
    arousal_dir = PROCESSED_DIR / "arousals"
    arousal_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAggregating {split_name} arousals...")
    if arousal_lbl:
        # np.concatenate joins the individual patient batches into one massive array
        final_a_eeg = np.concatenate(arousal_eeg, axis=0).astype(np.float32)
        final_a_ctx = np.concatenate(arousal_ctx, axis=0).astype(np.float32)
        final_a_lbl = np.concatenate(arousal_lbl, axis=0)
        
        out_path = arousal_dir / f"arousals_{split_name}.npz"
        np.savez_compressed(
            out_path,
            eeg_windows=final_a_eeg,
            context_windows=final_a_ctx,
            labels=final_a_lbl,
            fs=fs,
            ch_names=ch_names
        )
        print(f"[+] Saved {out_path.name} (Shape: {final_a_eeg.shape})")
    
    # Concatenate and Save Sleep Stages
    stage_dir = PROCESSED_DIR / "sleep_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Aggregating {split_name} sleep stages...")
    if stage_lbl:
        final_s_feats = np.concatenate(stage_feats, axis=0).astype(np.float32)
        final_s_lbl = np.concatenate(stage_lbl, axis=0)
        
        out_path = stage_dir / f"sleep_stages_{split_name}.npz"
        np.savez_compressed(
            out_path,
            sleep_features=final_s_feats,
            labels=final_s_lbl,
            fs=fs
        )
        print(f"[+] Saved {out_path.name} (Shape: {final_s_feats.shape})")

def main():
    cfg = PreprocessConfig()
    patient_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    
    if not patient_dirs:
        print("No patient directories found in data/raw. Exiting.")
        return
        
    random.seed(42)
    random.shuffle(patient_dirs)
    
    split_idx = int(len(patient_dirs) * 0.8)
    train_patients = patient_dirs[:split_idx]
    test_patients = patient_dirs[split_idx:]
    
    process_and_aggregate_split(train_patients, cfg, "train")
    process_and_aggregate_split(test_patients, cfg, "test")

if __name__ == "__main__":
    main()