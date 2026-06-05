import torch
import numpy as np
from scipy.signal import iirnotch, butter, sosfiltfilt, filtfilt, welch, stft

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