import numpy as np
from pathlib import Path

from src.utils import load_signals_and_arousals, label_window
from src.dsp import build_filters, filter_channel, compute_psd
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
        signals_raw, arousals = self._load_raw_data(raw_patient_dir)
        n_samples = signals_raw.shape[1]
        
        # 2. Continuous Filtering
        filtered_signals = self._apply_continuous_filters(signals_raw)

        # 3. Windowing & Feature Extraction
        out_signals, out_psd, out_labels = self._extract_window_features(n_samples, arousals, filtered_signals)

        return {
            "patient": raw_patient_dir.name,
            "signals": np.stack(out_signals),
            "psd": np.stack(out_psd),
            "labels": np.array(out_labels),
            "fs": self.cfg.fs,
            "ch_names": list(self.cfg.eeg_indices.keys())
        }
    
    def _load_raw_data(self, raw_patient_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        return load_signals_and_arousals(raw_patient_dir)

    def _apply_continuous_filters(self, signals_raw: np.ndarray) -> np.ndarray:
        sos_bp, b_notch, a_notch = build_filters(self.cfg)
        filtered_signals = np.stack([
            filter_channel(signals_raw[idx], sos_bp, b_notch, a_notch) 
            for idx in self.cfg.eeg_indices.values()
        ])
        return filtered_signals

    def _extract_window_features(self, n_samples: int, arousals: np.ndarray, filtered_signals: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
        ws = self.cfg.win_samples
        n_windows = n_samples // ws
        
        out_signals, out_psd, out_labels = [], [], []
        
        for w in range(n_windows):
            i0, i1 = w * ws, (w + 1) * ws
            
            # Determine arousal label for the window
            lw = label_window(arousals[i0:i1])
            if lw == -1:
                continue  # Skip unscored windows
                
            win_sig = filtered_signals[:, i0:i1]
            
            # Compute PSD for both channels
            win_psd = np.stack([
                compute_psd(win_sig[c], self.cfg)[1] for c in range(filtered_signals.shape[0])
            ])
            
            out_signals.append(win_sig)
            out_psd.append(win_psd)
            out_labels.append(lw)
        
        return out_signals, out_psd, out_labels
