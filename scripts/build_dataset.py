import random
from pathlib import Path
import numpy as np

from src.offline.config import PreprocessConfig
from src.offline.data_io import load_signals_and_arousals, save_npz
from src.offline.dsp import apply_causal_filters, compute_spectral_timeline
from src.offline.windowing import extract_arousal_windows, extract_sleep_stage_windows

def process_patient(patient_dir: Path, cfg: PreprocessConfig) -> dict:
    # Load
    signals, arousals, sleep_stages = load_signals_and_arousals(patient_dir)
    
    # Process
    filtered_signals = apply_causal_filters(signals, cfg)
    spectral_timeline = compute_spectral_timeline(filtered_signals, cfg)
    
    # Batch
    arousal_data = extract_arousal_windows(filtered_signals, arousals, cfg)
    stage_data = extract_sleep_stage_windows(spectral_timeline, sleep_stages, cfg)
    
    return {
        "arousal": arousal_data,
        "stages": stage_data,
        "fs": cfg.fs
    }

def main():
    cfg = PreprocessConfig()
    raw_dirs = list(Path("data/raw").iterdir())
    
    for patient_dir in raw_dirs:
        features = process_patient(patient_dir, cfg)
        