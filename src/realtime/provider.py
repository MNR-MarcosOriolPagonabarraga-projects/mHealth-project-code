import numpy as np
from pathlib import Path

class SampleProvider:
    def __init__(self, bin_path: Path, num_channels: int):
        self.bin_path = bin_path
        self.num_channels = num_channels
        flat_data = np.fromfile(self.bin_path, dtype=np.float32)
        self.raw_signals = flat_data.reshape(-1, self.num_channels)

    def __len__(self) -> int:
        """
        Returns the total number of samples. 
        This is what allows tqdm to calculate the % completion!
        """
        return self.raw_signals.shape[0]

    def __iter__(self):
        """Yields a single multi-channel sample array at a time."""
        # For a massive file, we could chunk this. For typical emulation, 
        # reading into RAM and yielding sequentially perfectly emulates the data stream.
        for sample in self.raw_signals:
            yield sample