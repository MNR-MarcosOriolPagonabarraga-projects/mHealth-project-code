import numpy as np
from src.config import PreprocessConfig
from .buffers import CircularBuffer
from .dsp import compute_bandpowers

class StreamProcessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.num_channels = len(cfg.channels)
        self.features_per_step = self.num_channels * 5 

        self.win_length = int(2.0 * cfg.fs)
        self.raw_history = CircularBuffer((self.num_channels, self.win_length))
        
        self.arousal_temporal = CircularBuffer((self.num_channels, cfg.win_samples))
        self.arousal_context = CircularBuffer((self.features_per_step, cfg.arousal_tensor_shape[1][2]))
        
        sleep_context_steps = int((cfg.sleep_win_sec * 2) * cfg.fs / cfg.hop_length)
        self.sleep_context = CircularBuffer((self.features_per_step, sleep_context_steps))

        self.sample_counter = 0

    def push(self, sample: np.ndarray) -> None:
        self.raw_history.push(sample)
        self.arousal_temporal.push(sample)
        self.sample_counter += 1

        if self.sample_counter % self.cfg.hop_length == 0:
            self._compute_stft_step()

    def _compute_stft_step(self) -> None:
        history = self.raw_history.get_ordered()
        features = compute_bandpowers(history, self.cfg.fs, self.cfg.n_fft)
        
        self.arousal_context.push(features)
        self.sleep_context.push(features)

    def extract_sleep_tensor(self) -> np.ndarray:
        """Returns the Doublet context (Past + Current) aligned for the Sleep Stage Net."""
        history = self.sleep_context.get_ordered()
        half_idx = history.shape[1] // 2
        
        past = history[:, :half_idx]
        current = history[:, half_idx:]
        
        return np.concatenate([past, current], axis=0)