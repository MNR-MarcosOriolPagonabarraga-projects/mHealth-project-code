import numpy as np

class CircularBuffer:
    def __init__(self, shape: tuple):
        """Shape should be (features/channels, time_steps)."""
        self.buffer = np.zeros(shape, dtype=np.float32)
        self.capacity = shape[-1]
        self.idx = 0

    def push(self, data: np.ndarray) -> None:
        self.buffer[..., self.idx] = data
        self.idx = (self.idx + 1) % self.capacity

    def get_ordered(self) -> np.ndarray:
        return np.roll(self.buffer, -self.idx, axis=-1)