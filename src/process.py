import numpy as np
from pathlib import Path

from src.utils import load_signals_and_arousals, extract_classification_windows, downsample_signal
from src.dsp import build_filters, filter_channel
from src.config import PreprocessConfig

class PatientProcessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg

    def __call__(self, raw_patient_dir: Path) -> dict:
        """
        Core pipeline to transform a raw PhysioNet patient directory into windowed features.
        Assumes a 'happy path' for readability.
        """
        # 1. Load Raw Data
        signals_raw, arousals, sleep_stages = self._load_raw_data(raw_patient_dir)
        signals_raw = self._clip_outliers(signals_raw, self.cfg.clip_threshold)
        
        # 2. Continuous Filtering
        filtered_signals = self._apply_continuous_filters(signals_raw)

        # 3. Downsample
        if self.cfg.downsample_factor > 1:
            filtered_signals = downsample_signal(filtered_signals, self.cfg.downsample_factor)
            arousals = arousals[::self.cfg.downsample_factor]

        # 4. Normalize Signals
        filtered_signals = self._normalize_signals(filtered_signals)

        # 5. Extract windows of interest
        signal_windows, context_windows, out_labels = extract_classification_windows(
            filtered_signals, 
            arousals, 
            fs=self.cfg.fs,
            win_sec=self.cfg.win_sec,
            neg_ratio=self.cfg.windows_neg_ratio
        )

        return {
            "patient": raw_patient_dir.name,
            "eeg_windows": signal_windows,
            "context_windows": context_windows,
            "sleep_stages": sleep_stages,
            "labels": out_labels,
            "fs": self.cfg.fs,
            "ch_names": list(self.cfg.eeg_indices.keys())
        }

    def _load_raw_data(self, raw_patient_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        return load_signals_and_arousals(
            raw_patient_dir, 
            channels=list(self.cfg.eeg_indices.values()), 
            include_sleep_stages=True)

    def _apply_continuous_filters(self, signals_raw: np.ndarray) -> np.ndarray:
        sos_bp, b_notch, a_notch = build_filters(self.cfg)
        filtered_signals = np.stack([
            filter_channel(chann, sos_bp, b_notch, a_notch) 
            for chann in signals_raw
        ])
        return filtered_signals
    
    def _normalize_signals(self, signals: np.ndarray) -> np.ndarray:
        # Normalize each channel to zero mean and unit variance
        return (signals - np.mean(signals, axis=1, keepdims=True)) / np.std(signals, axis=1, keepdims=True)

    def _clip_outliers(self, signals: np.ndarray, clip_threshold: float = 200.0) -> np.ndarray:
        # Clip outliers based on a multiple of the standard deviation
        mean = np.mean(signals, axis=1, keepdims=True)
        return np.clip(signals, mean - clip_threshold, mean + clip_threshold)
