from dataclasses import dataclass, field

@dataclass
class PreprocessConfig:
    original_fs: int = 200
    epoch_pre_sec: int = 10
    epoch_post_sec: int = 5
    windows_neg_ratio: int = 1
    bp_low_hz: float = 0.5
    bp_high_hz: float = 40.0
    notch_freq_hz: float = 60.0
    notch_q: float = 30.0
    downsample_factor: int = 2
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None
    eeg_indices: dict = field(
        default_factory=lambda: {"C3-M2": 2, "C4-M1": 3}
    )
    clip_threshold: float = 200.0
    all_channels = [
        "F3-M2",
        "F4-M1",
        "C3-M2",
        "C4-M1",
        "O1-M2",
        "O2-M1",
        "E1-M2",
        "Chin1-Chin2",
        "ABD",
        "CHEST",
        "AIRFLOW",
        "SaO2",
        "ECG",
    ]
    
    def __post_init__(self) -> None:
        if self.welch_nperseg is None:
            object.__setattr__(self, "welch_nperseg", self.fs * 4)
        if self.welch_noverlap is None:
            object.__setattr__(self, "welch_noverlap", self.fs * 2)

    @property
    def win_sec(self) -> int:
        return self.epoch_pre_sec + self.epoch_post_sec

    @property
    def win_samples(self) -> int:
        return self.fs * self.win_sec
    
    @property
    def fs(self) -> int:
        return self.original_fs // self.downsample_factor