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
        
        # Managing the identical 120-step doublet structure split across a 2D array matrix
        self.sleep_context = CircularBuffer((self.features_per_step * 2, 60))
        self.sleep_packet_counter = 0
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
        
        # Replicate the Zig layout routing math
        t = self.sleep_packet_counter % 60
        offset = (self.sleep_packet_counter // 60) * 10
        
        self.sleep_context.buffer[offset:offset+10, t] = features
        self.sleep_packet_counter = (self.sleep_packet_counter + 1) % 120

    def extract_sleep_tensor(self) -> np.ndarray:
        """Unrolls the doublet context cleanly into Past (0:10) and Current (10:20)."""
        unrolled = np.zeros((20, 60), dtype=np.float32)
        
        for i in range(60):
            # Resolve the circular indices for the Oldest step (Past) and Newest step (Current)
            past_idx = (self.sleep_packet_counter + i) % 120
            curr_idx = (self.sleep_packet_counter + 60 + i) % 120
            
            # Map physical locations in the buffer
            past_row_base = (past_idx // 60) * 10
            past_col = past_idx % 60
            
            curr_row_base = (curr_idx // 60) * 10
            curr_col = curr_idx % 60
            
            # Pack cleanly into the top and bottom halves
            unrolled[0:10, i] = self.sleep_context.buffer[past_row_base : past_row_base + 10, past_col]
            unrolled[10:20, i] = self.sleep_context.buffer[curr_row_base : curr_row_base + 10, curr_col]
            
        return unrolled