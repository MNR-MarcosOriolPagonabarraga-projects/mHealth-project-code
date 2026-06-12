from dataclasses import dataclass, field

@dataclass
class PreprocessConfig:
    original_fs: int = 200
    arousal_pre_sec: int = 10
    arousal_post_sec: int = 5
    arousal_ctx_sec: int = 300
    windows_neg_ratio: int = 1
    bp_low_hz: float = 0.5
    bp_high_hz: float = 40.0
    notch_freq_hz: float = 60.0
    notch_q: float = 30.0
    hop_length = 50
    n_fft = 256
    downsample_factor: int = 2
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None
    channels: dict = field(
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
        return self.arousal_pre_sec + self.arousal_post_sec

    @property
    def win_samples(self) -> int:
        return self.fs * self.win_sec
    
    @property
    def fs(self) -> int:
        return self.original_fs // self.downsample_factor


@dataclass
class TrainArousalsConfig:
    epochs: int = 50
    train_path: str = "data/processed/arousals/arousals_train.npz"
    test_path: str = "data/processed/arousals/arousals_test.npz"
    out_path: str = "models/arousals"
    class_names = ["No Arousal", "Arousal"]
    pred_threshold: float = 0.4
    batch_size: int = 256
    num_workers: int = 4
    train_balance: float = 1
    lr: float = 8e-4
    weight_decay: float = 1e-3
    eta_min: float = 1e-6

@dataclass
class TrainSleepStagesConfig:
    epochs: int = 40
    train_path: str = "data/processed/sleep_stage/sleep_stages_train.npz"
    test_path: str = "data/processed/sleep_stage/sleep_stages_test.npz"
    out_path: str = "models/sleep_stage"
    class_names = ['Wake', 'Light Sleep', 'Deep Sleep', 'REM']
    batch_size: int = 256
    num_workers: int = 4
    train_balance: list = field(
        default_factory=lambda: [1.0, 0.6, 1.4, 1.8]
    )
    lr: float = 5e-5
    weight_decay: float = 1e-2
    eta_min: float = 1e-6
