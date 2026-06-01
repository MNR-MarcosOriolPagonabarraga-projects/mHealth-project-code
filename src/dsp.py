import scipy.signal
import numpy as np
from src.config import PreprocessConfig



def build_filters(cfg: PreprocessConfig):
    sos_bp = scipy.signal.butter(
        4,
        [cfg.bp_low_hz, cfg.bp_high_hz],
        btype="bandpass",
        fs=cfg.fs,
        output="sos",
    )
    b_notch, a_notch = scipy.signal.iirnotch(cfg.notch_freq_hz, cfg.notch_q, fs=cfg.fs)
    return sos_bp, b_notch, a_notch


def filter_channel(sig: np.ndarray, sos_bp, b_notch, a_notch) -> np.ndarray:
    sig = scipy.signal.sosfiltfilt(sos_bp, sig)
    sig = scipy.signal.filtfilt(b_notch, a_notch, sig)
    return sig.astype(np.float32)


def compute_psd(sig_win: np.ndarray, cfg: PreprocessConfig):
    # Extract the bounds directly from the config object inside the function
    low, high = cfg.bp_low_hz, cfg.bp_high_hz
    
    freqs, psd = scipy.signal.welch(
        sig_win,
        fs=cfg.fs,
        nperseg=cfg.welch_nperseg,
        noverlap=cfg.welch_noverlap,
        window="hann",
        scaling="density",
    )
    mask = (freqs >= low) & (freqs <= high)
    return freqs[mask], psd[mask]
