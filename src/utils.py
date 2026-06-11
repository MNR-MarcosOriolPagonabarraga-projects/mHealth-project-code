import numpy as np
from scipy.signal import decimate

from offline.config import FS
from offline.dsp import compute_full_recording_bandpower

def downsample_signal(signal: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Safe Downsample using smooth decimation to avoid aliasing."""
    if downsample_factor <= 1:
        return signal
    
    return decimate(signal, downsample_factor, ftype='iir', zero_phase=True)

