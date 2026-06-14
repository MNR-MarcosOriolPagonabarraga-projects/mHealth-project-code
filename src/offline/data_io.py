import h5py
import numpy as np
from pathlib import Path
from scipy.io import loadmat


def load_signals_and_annotations(
    patient_path: str | Path,
    channels: list
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load ``val`` from ``.mat`` and ``data/arousals`` from HDF5 ``-arousal.mat``."""
    patient_name = patient_path.name
    mat_path = patient_path / f"{patient_name}.mat"
    arousal_path = patient_path / f"{patient_name}-arousal.mat"

    mat = loadmat(mat_path)

    signals = mat["val"][channels, :].astype(np.float32)

    sleep_stages: dict[str, np.ndarray] = {}
    with h5py.File(arousal_path, "r") as f:
        arousals = f["data/arousals"][()].flatten()

        sleep_stages = np.full_like(arousals, fill_value=-1, dtype=np.int8)
        stage_map = {
            'wake': 0,
            'nonrem1': 1, 
            'nonrem2': 1,  # Merged N1/N2 into Light Sleep (1)
            'nonrem3': 2,  # Deep Sleep (2)
            'rem': 3       # REM (3)
        }
        
        grp = f["data/sleep_stages"]
        for k, label_int in stage_map.items():
            if k in grp:
                mask = grp[k][()].flatten()
                sleep_stages[mask == 1] = label_int

    return signals, arousals, sleep_stages


def save_npz(out_path, keys, values):
    data_dict = dict(zip(keys, values))
    np.savez_compressed(
        out_path,
        **data_dict
    )
