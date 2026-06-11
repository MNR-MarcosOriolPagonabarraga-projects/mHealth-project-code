import numpy as np
from h5py import h5py
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

        grp = f["data/sleep_stages"]
        for k in ("wake", "nonrem1", "nonrem2", "nonrem3", "rem", "undefined"):
            sleep_stages[k] = grp[k][()].flatten()

    return signals, arousals, sleep_stages


def save_npz(out_path, keys, values):
    data_dict = dict(zip(keys, values))
    np.savez_compressed(
        out_path,
        **data_dict
    )
