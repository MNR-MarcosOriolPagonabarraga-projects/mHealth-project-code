from dataclasses import dataclass, field

FS = 200  # Hz

ALL_CHANNELS = [
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

EEG_IDX = [0, 1, 2, 3, 4, 5]
FOCUS_IDX = [2, 3]
EEG_CHANNELS = {"C3-M2": 2, "C4-M1": 3}

VIZ_START_MIN = 60
VIZ_END_MIN = 70

# Exploration plot colors
_EXP_COLORS = {
    "eeg_focus": "#2563eb",
    "eeg_other": "#94a3b8",
    "arousal": "#ef4444",
    "non_arousal": "#3b82f6",
    "no_scored": "#d1d5db",
}

@dataclass
class PreprocessConfig:
    original_fs: int = FS
    epoch_pre_sec: int = 20
    epoch_post_sec: int = 15
    windows_neg_ratio: int = 2
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
        return FS // self.downsample_factor