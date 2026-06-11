import numpy as np
from pathlib import Path

from src.utils import load_signals_and_arousals, extract_classification_windows, downsample_signal
from offline.dsp import build_filters, filter_channel
from offline.config import PreprocessConfig

class PatientProcessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg

    def __call__(self, raw_patient_dir: Path) -> dict:
        """
        Core pipeline to transform a raw PhysioNet patient directory into windowed features.
        Assumes a 'happy path' for readability.
        """
        # 1. Load Raw Data
        signals_raw, arousals = self._load_raw_data(raw_patient_dir, sleep_stages=False)
        signals_raw = self._clip_outliers(signals_raw, self.cfg.clip_threshold)
        
        # 2. Continuous Filtering
        filtered_signals = self._apply_continuous_filters(signals_raw)

        # 3. Downsample
        if self.cfg.downsample_factor > 1:
            filtered_signals = downsample_signal(filtered_signals, self.cfg.downsample_factor)
            arousals = arousals[::self.cfg.downsample_factor]

        # 5. Extract windows of interest
        signal_windows, context_windows, out_labels = extract_classification_windows(
            filtered_signals, 
            arousals, 
            fs=self.cfg.fs,
            win_sec=self.cfg.win_sec,
            neg_ratio=self.cfg.windows_neg_ratio
        )

        signal_windows = self._normalize_signals(signal_windows, axis=2)
        context_windows = self._normalize_signals(context_windows, axis=1)
        if len(out_labels) == 0:
            clean_signals = np.empty((0, 2, 1500), dtype=np.float32)
            clean_contexts = np.empty((0, 149, 2 * 5), dtype=np.float32)
            clean_labels = np.empty((0,), dtype=np.int8) 
        else:
            clean_signals = np.stack(signal_windows).astype(np.float32)
            clean_contexts = np.stack(context_windows).astype(np.float32)
            clean_labels = np.array(out_labels, dtype=np.int8)

        return {
            "patient": raw_patient_dir.name,
            "eeg_windows": clean_signals,
            "context_windows": clean_contexts,
            "labels": clean_labels,
            "fs": self.cfg.fs,
            "ch_names": list(self.cfg.eeg_indices.keys())
        }

    def _load_raw_data(self, raw_patient_dir: Path, sleep_stages: bool = False) -> tuple[np.ndarray, np.ndarray]:
        return load_signals_and_arousals(
            raw_patient_dir, 
            channels=list(self.cfg.eeg_indices.values()), 
            include_sleep_stages=sleep_stages)

    def _apply_continuous_filters(self, signals_raw: np.ndarray) -> np.ndarray:
        sos_bp, b_notch, a_notch = build_filters(self.cfg)
        filtered_signals = np.stack([
            filter_channel(chann, sos_bp, b_notch, a_notch) 
            for chann in signals_raw
        ])
        return filtered_signals
    
    def _normalize_signals(self, signal_windows: np.ndarray, axis: int) -> np.ndarray:
        if len(signal_windows) > 0:
            means = np.mean(signal_windows, axis=axis, keepdims=True)
            stds = np.std(signal_windows, axis=axis, keepdims=True) + 1e-8
            signal_windows = (signal_windows - means) / stds
            return signal_windows

    def _clip_outliers(self, signals: np.ndarray, clip_threshold: float = 200.0) -> np.ndarray:
        # Clip outliers based on a multiple of the standard deviation
        mean = np.mean(signals, axis=1, keepdims=True)
        return np.clip(signals, mean - clip_threshold, mean + clip_threshold)
