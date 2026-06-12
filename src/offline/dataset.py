import numpy as np
import torch
from torch.utils.data import Dataset

class ArousalsDataset(Dataset):
    def __init__(self, npz_path):
        """
        Loads the pre-aggregated train or test .npz file entirely into RAM.
        Works flawlessly for small subsets, but will require the caching 
        technique we discussed earlier if scaled to the full 135 GB dataset.
        """
        print(f"Loading {npz_path} into RAM...")
        data = np.load(npz_path)
        
        # Convert to PyTorch tensors
        self.signals = torch.tensor(data['eeg_windows'], dtype=torch.float32)
        self.context = torch.tensor(data['context_windows'])        
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.context[idx].permute(1, 0), self.labels[idx]


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


