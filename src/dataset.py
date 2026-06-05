import numpy as np
import torch
from torch.utils.data import Dataset

class PhysNetDualDataset(Dataset):
    def __init__(self, npz_path):
        """
        Loads the pre-aggregated train or test .npz file entirely into RAM.
        Works flawlessly for small subsets, but will require the caching 
        technique we discussed earlier if scaled to the full 135 GB dataset.
        """
        print(f"Loading {npz_path} into RAM...")
        data = np.load(npz_path)
        
        # Convert to PyTorch tensors
        self.signals = torch.tensor(data['signals'], dtype=torch.float32)
        self.psd = torch.tensor(data['psd'], dtype=torch.float32)
        
        # BCEWithLogitsLoss expects float32 targets, not long integers
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.psd[idx], self.labels[idx]


class PhysNetSignalDataset(Dataset):
    def __init__(self, npz_path):
        """
        Loads the pre-aggregated train or test .npz file entirely into RAM.
        Works flawlessly for small subsets, but will require the caching 
        technique we discussed earlier if scaled to the full 135 GB dataset.
        """
        print(f"Loading {npz_path} into RAM...")
        data = np.load(npz_path)
        
        # Convert to PyTorch tensors
        self.signals = torch.tensor(data['signals'], dtype=torch.float32)        
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class PhysNetStftDataset(Dataset):
    def __init__(self, npz_path):
        """
        Loads the pre-aggregated train or test .npz file entirely into RAM.
        Works flawlessly for small subsets, but will require the caching 
        technique we discussed earlier if scaled to the full 135 GB dataset.
        """
        print(f"Loading {npz_path} into RAM...")
        data = np.load(npz_path)
        
        # Convert to PyTorch tensors
        self.signals = torch.tensor(data['stft_windows'], dtype=torch.float32)        
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

