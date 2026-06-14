import numpy as np
import torch
from torch.utils.data import Dataset

class ArousalsDataset(Dataset):
    def __init__(self, npz_path, is_train: bool = False):
        """
        Loads the pre-aggregated train or test .npz file entirely into RAM.
        """
        print(f"Loading {npz_path} into RAM...")
        data = np.load(npz_path)
        
        # Convert to PyTorch tensors
        self.signals = torch.tensor(data['eeg_windows'], dtype=torch.float32)
        self.context = torch.tensor(data['context_windows'], dtype=torch.float32)        
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        
        # Flag to toggle data augmentation
        self.is_train = is_train
        
    def __len__(self):
        return len(self.labels)

    def _apply_eeg_noise(self, signal: torch.Tensor, snr_db: float = 30.0) -> torch.Tensor:
        """Adds subtle Gaussian White Noise proportional to the signal power."""
        # Calculate signal power
        signal_power = torch.mean(signal ** 2)
        if signal_power == 0:
            return signal
            
        # Calculate required noise variance based on desired Signal-to-Noise Ratio (SNR)
        # 30dB is standard for clean, subtle augmentation
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        std_dev = torch.sqrt(noise_power)
        
        # Add random zero-mean Gaussian noise
        noise = torch.randn_like(signal) * std_dev
        return signal + noise

    def _apply_context_noise(self, context: torch.Tensor, scale_range: float = 0.05) -> torch.Tensor:
        """Slightly shifts the baseline bandpowers to prevent over-memorizing the exact macro-state."""
        # Generates a random scale value between e.g., 0.95 and 1.05
        scale_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * scale_range
        return context * scale_factor

    def __getitem__(self, idx):
        # Always clone to avoid modifying the core dataset stored in RAM
        x_signal = self.signals[idx].clone()
        x_context = self.context[idx].clone()
        y_label = self.labels[idx]
        
        # Only inject noise if we are actively training
        if self.is_train:
            x_signal = self._apply_eeg_noise(x_signal, snr_db=32.0)  # Subtle 32dB SNR noise
            x_context = self._apply_context_noise(x_context, scale_range=0.03)  # Max ±3% scaling shift
            
        return x_signal, x_context.permute(1, 0), y_label


class SleepStageDataset(Dataset):
    def __init__(self, npz_path, is_train=False, max_mask_pct=0.15):
        print(f"[*] Loading {npz_path} into RAM...")
        data = np.load(npz_path)
        
        # Load features and labels matching your dataset builder
        self.features = torch.tensor(data['sleep_features'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        
        self.is_train = is_train
        self.max_mask_pct = max_mask_pct

    def _tiny_mask_drop(self, x):
        if self.is_train and self.max_mask_pct > 0:
            time_steps = x.shape[0]  # Assumes shape is (Time, Features)
            mask_t = int(time_steps * self.max_mask_pct)
            
            if mask_t > 0:
                # Pick a random starting point for the temporal mask
                t0 = torch.randint(0, time_steps - mask_t, (1,)).item()
                x[t0:t0+mask_t, :] = 0.0
            
            return x
        return x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Clone to prevent modifying the original dataset in RAM
        x = self.features[idx].clone()
        y = self.labels[idx]
        x = self._tiny_mask_drop(x)
            
        return x.permute(1,0), y


