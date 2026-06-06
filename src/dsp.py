import torch
import numpy as np
from scipy.signal import iirnotch, butter, sosfiltfilt, filtfilt, welch, spectrogram

from src.config import PreprocessConfig



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


def filter_channel(sig: np.ndarray, sos_bp, b_notch, a_notch) -> np.ndarray:
    sig = sosfiltfilt(sos_bp, sig)
    sig = filtfilt(b_notch, a_notch, sig)
    return sig.astype(np.float32)

def apply_continuous_filters(signals_raw: np.ndarray, cfg) -> np.ndarray:
        sos_bp, b_notch, a_notch = build_filters(cfg)
        if signals_raw.ndim > 1:
            filtered_signals = np.stack([
                filter_channel(chann, sos_bp, b_notch, a_notch) 
                for chann in signals_raw
            ])
        else:
            filtered_signals = filter_channel(signals_raw, sos_bp, b_notch, a_notch)
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
    n_fft = int(2.0 * fs)                 # 200 samples
    win_length = int(2.0 * fs)            # 200 samples
    hop_length = int(0.25 * fs)           # 25 samples
    
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
    
    # Unflatten the dimensions back into a 4D batch structure
    # Shape changes from (B * C, Freqs, Time) -> (B, C, Freqs, Time)
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
    spec = torch.log1p(spec)
    
    # Define clinical EEG bands mapped to FFT bin indices
    bin_res = fs / n_fft
    bands = {
        'delta': (int(0.5 / bin_res), int(4.0 / bin_res)),
        'theta': (int(4.0 / bin_res), int(8.0 / bin_res)),
        'alpha': (int(8.0 / bin_res), int(12.0 / bin_res)),
        'sigma': (int(12.0 / bin_res), int(16.0 / bin_res)),
        'beta':  (int(16.0 / bin_res), int(30.0 / bin_res))
    }
    
    band_powers = []
    for name, (low_idx, high_idx) in bands.items():
        # Sum energy across the frequency bins (dim=1) for all channels and time steps
        power = torch.sum(spec[:, low_idx:high_idx, :], dim=1) # Shape: (n_channels, total_time_steps)
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