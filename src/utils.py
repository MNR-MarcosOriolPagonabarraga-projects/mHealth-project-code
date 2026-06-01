import h5py
from pathlib import Path
import numpy as np
import scipy.io

from src.config import FS


def _base_without_extension(base_path: str | Path) -> Path:
    p = Path(base_path)
    for suffix in (".mat", ".hea"):
        if p.suffix.lower() == suffix:
            return p.with_suffix("")
    return p


def load_signals_and_arousals(
    base_path: str | Path,
    *,
    include_sleep_stages: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load ``val`` from ``.mat`` and ``data/arousals`` from HDF5 ``-arousal.mat``."""
    base = _base_without_extension(base_path)
    mat_path = base / f"tr03-{base.name}.mat"
    arousal_path = base / f"tr03-{base.name}-arousal.mat"
    print(mat_path, arousal_path)

    mat = scipy.io.loadmat(mat_path)
    signals = mat["val"].astype(np.float32)

    sleep_stages: dict[str, np.ndarray] = {}
    with h5py.File(arousal_path, "r") as f:
        arousals = f["data/arousals"][()].flatten()
        if include_sleep_stages:
            grp = f["data/sleep_stages"]
            for k in ("wake", "nonrem1", "nonrem2", "nonrem3", "rem", "undefined"):
                sleep_stages[k] = grp[k][()].flatten()

    if include_sleep_stages:
        return signals, arousals, sleep_stages
    return signals, arousals


def describe_recording(arousals: np.ndarray, fs: int = FS) -> None:
    n_samples = arousals.shape[0]
    print(f"\n{'=' * 55}")
    print(f"  Muestras: {n_samples:,}")
    duration_min = n_samples / fs / 60
    print(f"  Duracion: {duration_min:.1f} min ({duration_min / 60:.2f} h)")
    print("\n  Distribucion arousals:")
    for v, label in [(-1, "No scored"), (0, "Non-arousal"), (1, "Arousal (+1)")]:
        n = int(np.sum(arousals == v))
        pct = 100 * n / n_samples if n_samples else 0.0
        print(f"    {label:15s}: {n:>8,} muestras  ({pct:.1f}%)  ~ {n / fs / 60:.1f} min")
    print(f"{'=' * 55}\n")


def print_batch_summary(summary: list[dict]) -> None:
    print(f"{'Paciente':<14} {'Ventanas':>9} {'Arousal':>9} {'Non-ar':>9} {'Ratio%':>8} {'MB':>6}")
    print("-" * 60)
    total_win = total_ar = total_non = 0
    for s in summary:
        print(
            f"  {s['patient']:<12} {s['n_windows']:>9} {s['n_arousal']:>9} "
            f"{s['n_nonarousal']:>9} {s['ratio_pct']:>7.1f}% {s['size_mb']:>5.1f}"
        )
        total_win += s["n_windows"]
        total_ar += s["n_arousal"]
        total_non += s["n_nonarousal"]

    print("-" * 60)
    ratio_total = total_ar / (total_ar + total_non) * 100 if (total_ar + total_non) else 0
    print(f"  {'TOTAL':<12} {total_win:>9} {total_ar:>9} {total_non:>9} {ratio_total:>7.1f}%")
    print()
    if total_ar > 0:
        print("Pesos sugeridos para weighted CrossEntropyLoss:")
        print(f"  weight = torch.tensor([1.0, {total_non/total_ar:.2f}])  # [non-arousal, arousal]")
    print()

def label_window(arousal_win: np.ndarray) -> int:
    n = len(arousal_win)
    if n == 0:
        return -1
    if np.sum(arousal_win == 1) / n > 0.5:
        return 1
    if np.sum(arousal_win == 0) / n > 0.5:
        return 0
    return -1