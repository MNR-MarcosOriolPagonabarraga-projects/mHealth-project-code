import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, lfilter, sosfilt_zi, lfilter_zi

class CausalStatefulFilter:
    def __init__(self, fs: int, low_hz: float, high_hz: float, notch_hz: float, notch_q: float):
        self.sos_bp = butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
        self.b_notch, self.a_notch = iirnotch(notch_hz, notch_q, fs=fs)
        
        # Track states (zi) exactly like Zig's x1, x2, y1, y2 buffers
        self.zi_sos = sosfilt_zi(self.sos_bp)
        self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)

    def process(self, x: float) -> float:
        x_arr = np.array([x])
        x_notch, self.zi_notch = lfilter(self.b_notch, self.a_notch, x_arr, zi=self.zi_notch)
        x_bp, self.zi_sos = sosfilt(self.sos_bp, x_notch, zi=self.zi_sos)
        return float(x_bp[0])

class Downsampler:
    def __init__(self, factor: int):
        self.factor = factor
        self.counter = 0

    def push(self, sample: float) -> float | None:
        self.counter += 1
        if self.counter == self.factor:
            self.counter = 0
            return sample
        return None

def compute_bandpowers(win_samples: np.ndarray, fs: int, n_fft: int) -> np.ndarray:
    num_channels, win_length = win_samples.shape
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win_length) / (win_length - 1)))
    windowed = win_samples * hann

    padded = np.zeros((num_channels, n_fft), dtype=np.float32)
    padded[:, :win_length] = windowed
    
    power_spec = np.abs(np.fft.fft(padded, axis=1)) ** 2

    bin_res = fs / n_fft
    bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0)]
    out_bands = np.zeros((num_channels, len(bands)), dtype=np.float32)

    for b_idx, (low, high) in enumerate(bands):
        low_idx = max(int(low / bin_res), 0)
        high_idx = min(int(high / bin_res) + 1, n_fft // 2)
        out_bands[:, b_idx] = np.log1p(np.sum(power_spec[:, low_idx:high_idx], axis=1))

    return out_bands.flatten()

def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, ddof=0, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (arr - mean) / std