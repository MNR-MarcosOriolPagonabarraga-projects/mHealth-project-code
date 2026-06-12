import torch
import numpy as np
from scipy.signal import iirnotch, butter, sosfilt, lfilter, welch

from src.offline.config import PreprocessConfig


def clip_outliers(signals: np.ndarray, clip_threshold: float = 200.0) -> np.ndarray:
    # Clip outliers based on a multiple of the standard deviation
    mean = np.mean(signals, axis=1, keepdims=True)
    return np.clip(signals, mean - clip_threshold, mean + clip_threshold)


def downsample_all(factor: int, *arrays: np.ndarray) -> tuple:
    """
    Downsamples any number of arrays by the given factor.
    Always returns a tuple of the same length as the input.
    """
    if factor <= 1:
        return arrays
    
    return tuple(arr[..., ::factor] for arr in arrays)


def build_filters(cfg: PreprocessConfig):
    sos_bp = butter(
        4,
        [cfg.bp_low_hz, cfg.bp_high_hz],
        btype="bandpass",
        fs=cfg.original_fs,
        output="sos",
    )
    b_notch, a_notch = iirnotch(cfg.notch_freq_hz, cfg.notch_q, fs=cfg.original_fs)
    return sos_bp, b_notch, a_notch


def filter_channel_causal(sig: np.ndarray, sos_bp, b_notch, a_notch) -> np.ndarray:
    sig = sosfilt(sos_bp, sig)
    sig = lfilter(b_notch, a_notch, sig)
    return sig.astype(np.float32)


def apply_causal_filters(signals_raw: np.ndarray, cfg) -> np.ndarray:
        sos_bp, b_notch, a_notch = build_filters(cfg)
        if signals_raw.ndim > 1:
            filtered_signals = np.stack([
                filter_channel_causal(chann, sos_bp, b_notch, a_notch) 
                for chann in signals_raw
            ])
        else:
            filtered_signals = filter_channel_causal(signals_raw, sos_bp, b_notch, a_notch)
        return filtered_signals


def compute_psd(sig_win: np.ndarray, cfg: PreprocessConfig):
    # Extract the bounds directly from the config object inside the function
    low, high = cfg.bp_low_hz, cfg.bp_high_hz
    
    freqs, psd = welch(
        sig_win,
        fs=cfg.original_fs,
        nperseg=cfg.welch_nperseg,
        noverlap=cfg.welch_noverlap,
        window="hann",
        scaling="density",
    )
    mask = (freqs >= low) & (freqs <= high)
    return freqs[mask], psd[mask]


def extract_stft_features(batch_signals: torch.Tensor, fs: int = 100, device: str = 'cuda') -> torch.Tensor:
    """
    Args:
        batch_signals: torch.Tensor of shape (B, C, T) -> e.g., (528, 2, 1500)
    """
    # Flatten the Batch and Channel dimensions together into a 2D matrix
    # Shape changes from (B, C, T) -> (B * C, T) -> e.g., (1056, 1500)
    B, C, T = batch_signals.shape
    x_flattened = batch_signals.reshape(B * C, T)
    
    # STFT hyper-parameters
    n_fft = int(2.0 * fs)
    win_length = int(2.0 * fs)
    hop_length = int(0.25 * fs)
    
    window = torch.hann_window(win_length, device=device)
    
    # Compute STFT on the 2D tensor
    # Output shape: (B * C, Freq_Bins, Time_Bins)
    stft_out = torch.stft(
        x_flattened,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        normalized=False,
        return_complex=True
    )
    
    psd = torch.abs(stft_out) ** 2
    eps = 1e-10
    log_psd = 10 * torch.log10(psd + eps)
    
    # Bin index 1 to 81 covers 0.5Hz to 40.0Hz
    log_psd_cropped = log_psd[:, 1:81, :]
    _, Freq_Bins, Time_Bins = log_psd_cropped.shape
    spectrogram_4d = log_psd_cropped.view(B, C, Freq_Bins, Time_Bins)
    
    return spectrogram_4d


def batch_extract_stft(signals_np: np.ndarray, fs: int = 100, batch_size: int = 2000) -> np.ndarray:
    """
    Safely routes massive datasets through the GPU STFT function in chunks to prevent VRAM overflow.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    n_total = signals_np.shape[0]
    if n_total == 0:
        return np.empty((0, signals_np.shape[1], 80, 53), dtype=np.float32)
    
    processed_batches = []
    
    # Slide through the massive array in safe VRAM chunks
    for i in range(0, n_total, batch_size):
        batch_signals = signals_np[i : i + batch_size]
        batch_signals = torch.from_numpy(batch_signals).to(dtype=torch.float32, device=device)
        
        # Compute on GPU
        batch_stft = extract_stft_features(batch_signals, fs=fs, device=device)
        
        # Append the processed CPU numpy array to our list
        processed_batches.append(batch_stft.cpu().numpy())
        
    # Stack all chunks back into one massive processed dataset
    return np.concatenate(processed_batches, axis=0)


def compute_full_recording_bandpower(signals: np.ndarray, fs: int, n_fft: int = 512, hop_length: int = 200) -> np.ndarray:
    """
    Vectorized PyTorch computation of rolling bandpowers for the entire recording.
    Signals shape: (n_channels, total_samples)
    Returns: (total_stft_time_steps, n_channels * 5)
    """
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    if not signals.flags['C_CONTIGUOUS']:
        signals = np.ascontiguousarray(signals)
        
    # Move to available device for speed, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Convert numpy array to PyTorch tensor
    x = torch.from_numpy(signals).float().to(device) # Shape: (n_channels, total_samples)
    
    # Compute STFT for all channels simultaneously
    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        x, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        window=window, 
        return_complex=True
    ) # Shape: (n_channels, freq_bins, total_time_steps)
    
    # Square magnitude to get the Spectrogram (saves 50% RAM vs raw complex STFT)
    spec = torch.abs(stft_out) ** 2 
    
    # Define clinical EEG bands mapped to FFT bin indices
    bin_res = fs / n_fft  # 100 / 256 = 0.3906 Hz
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'sigma': (12.0, 16.0),
        'beta':  (16.0, 30.0)
    }
    
    band_powers = []
    for name, (low_hz, high_hz) in bands.items():
        # The math automatically handles the new 256 grid perfectly
        low_idx = max(int(low_hz / bin_res), 0)
        high_idx = min(int(high_hz / bin_res) + 1, spec.shape[1])
        
        power = torch.sum(spec[:, low_idx:high_idx, :], dim=1)
        power = torch.log1p(power)
        band_powers.append(power)
        
    # Stack bands: (5, n_channels, total_time_steps)
    stacked = torch.stack(band_powers, dim=0)
    
    # Permute to easily pull time steps: (total_time_steps, n_channels, 5)
    stacked = stacked.permute(2, 0, 1) 
    
    # Flatten channels and bands: (total_time_steps, n_channels * 5)
    # e.g., if 2 channels, features at step t will be [ch0_delta..ch0_beta, ch1_delta..ch1_beta]
    total_time_steps = stacked.shape[0]
    final_features = stacked.reshape(total_time_steps, -1)
    
    return final_features.cpu().numpy()