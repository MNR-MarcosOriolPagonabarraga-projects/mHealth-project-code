"""
PhysioNet Challenge 2018: load recordings, preprocessing to .npz windows, exploration plots.

CLI (from repo root, with ``src`` on ``PYTHONPATH``, or after ``pip install -e .``):

  python process.py --data-dir Data --out-dir Processed
  python process.py explore Data/0005/tr03-0005
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal

# ---------------------------------------------------------------------------
# Constants (recording layout)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


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
    mat_path = base.with_suffix(".mat")
    arousal_path = Path(str(base) + "-arousal.mat")

    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing signal file: {mat_path}")
    if not arousal_path.is_file():
        raise FileNotFoundError(f"Missing arousal file: {arousal_path}")

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


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    fs: int = FS
    win_sec: int = 30
    bp_low_hz: float = 0.5
    bp_high_hz: float = 40.0
    notch_freq_hz: float = 50.0
    notch_q: float = 30.0
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None

    def __post_init__(self) -> None:
        if self.welch_nperseg is None:
            object.__setattr__(self, "welch_nperseg", self.fs * 4)
        if self.welch_noverlap is None:
            object.__setattr__(self, "welch_noverlap", self.fs * 2)

    @property
    def win_samples(self) -> int:
        return self.fs * self.win_sec


def build_filters(cfg: PreprocessConfig):
    sos_bp = scipy.signal.butter(
        4,
        [cfg.bp_low_hz, cfg.bp_high_hz],
        btype="bandpass",
        fs=cfg.fs,
        output="sos",
    )
    b_notch, a_notch = scipy.signal.iirnotch(cfg.notch_freq_hz, cfg.notch_q, fs=cfg.fs)
    return sos_bp, b_notch, a_notch


def filter_channel(sig: np.ndarray, sos_bp, b_notch, a_notch) -> np.ndarray:
    sig = scipy.signal.sosfiltfilt(sos_bp, sig)
    sig = scipy.signal.filtfilt(b_notch, a_notch, sig)
    return sig.astype(np.float32)


def label_window(arousal_win: np.ndarray) -> int:
    n = len(arousal_win)
    if n == 0:
        return -1
    if np.sum(arousal_win == 1) / n > 0.5:
        return 1
    if np.sum(arousal_win == 0) / n > 0.5:
        return 0
    return -1


def compute_psd(sig_win: np.ndarray, cfg: PreprocessConfig, sos_bp_bounds: tuple[float, float]):
    low, high = sos_bp_bounds
    freqs, psd = scipy.signal.welch(
        sig_win,
        fs=cfg.fs,
        nperseg=cfg.welch_nperseg,
        noverlap=cfg.welch_noverlap,
        window="hann",
        scaling="density",
    )
    mask = (freqs >= low) & (freqs <= high)
    return freqs[mask], psd[mask]


def process_patient(
    base_path: str | Path,
    sos_bp,
    b_notch,
    a_notch,
    cfg: PreprocessConfig,
    *,
    eeg_indices: dict[str, int] | None = None,
    verbose: bool = True,
) -> dict:
    eeg_indices = eeg_indices or EEG_CHANNELS
    pid = Path(base_path).name
    if verbose:
        print(f"  [{pid}] Cargando...", end=" ", flush=True)

    signals_raw, arousals = load_signals_and_arousals(base_path)
    bp_bounds = (cfg.bp_low_hz, cfg.bp_high_hz)

    n_samples = signals_raw.shape[1]
    ws = cfg.win_samples
    n_windows = n_samples // ws
    if verbose:
        print(f"{n_samples:,} muestras -> {n_windows} ventanas", end=" ", flush=True)

    filtered = np.stack(
        [filter_channel(signals_raw[idx], sos_bp, b_notch, a_notch) for idx in eeg_indices.values()]
    )

    freqs, _ = compute_psd(filtered[0, :ws], cfg, bp_bounds)
    out_signals, out_psd, out_labels = [], [], []

    for w in range(n_windows):
        i0, i1 = w * ws, (w + 1) * ws
        lw = label_window(arousals[i0:i1])
        if lw == -1:
            continue
        win_sig = filtered[:, i0:i1]
        win_psd = np.stack([compute_psd(win_sig[c], cfg, bp_bounds)[1] for c in range(2)])
        out_signals.append(win_sig)
        out_psd.append(win_psd)
        out_labels.append(lw)

    if not out_labels:
        raise ValueError(f"No valid windows for {pid}")

    signals_arr = np.stack(out_signals)
    psd_arr = np.stack(out_psd)
    labels_arr = np.array(out_labels)

    n_ar = int((labels_arr == 1).sum())
    n_non = int((labels_arr == 0).sum())
    if verbose:
        ratio = n_ar / (n_ar + n_non) * 100 if (n_ar + n_non) else 0
        print(
            f"-> {len(labels_arr)} validas  (arousal={n_ar}, non-arousal={n_non}, ratio={ratio:.1f}%)"
        )

    return {
        "signals": signals_arr,
        "psd": psd_arr,
        "freqs": freqs,
        "labels": labels_arr,
        "patient": np.str_(pid),
        "ch_names": np.array(list(eeg_indices.keys())),
        "fs": np.int64(cfg.fs),
        "win_sec": np.int64(cfg.win_sec),
        "n_arousal": np.int64(n_ar),
        "n_nonarousal": np.int64(n_non),
    }


def discover_patient_roots(data_dir: str | Path, pattern: str = "tr03-*.mat") -> list[Path]:
    root = Path(data_dir)
    mats = sorted(
        p
        for p in glob.glob(os.path.join(str(root), "**", pattern), recursive=True)
        if not p.endswith("-arousal.mat")
    )
    return [Path(p).with_suffix("") for p in mats]


def run_batch(
    data_dir: Path,
    out_dir: Path | None,
    *,
    cfg: PreprocessConfig | None = None,
    verbose: bool = True,
) -> list[dict]:
    cfg = cfg or PreprocessConfig()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    patient_bases = discover_patient_roots(data_dir)
    if verbose:
        print(f"Pacientes encontrados: {len(patient_bases)}")
        for b in patient_bases:
            print(f"  {b}")

    sos_bp, b_notch, a_notch = build_filters(cfg)
    summary: list[dict] = []
    if verbose:
        print(f"\nProcesando {len(patient_bases)} paciente(s)...\n")

    for base in patient_bases:
        pid = base.name
        try:
            result = process_patient(base, sos_bp, b_notch, a_notch, cfg, verbose=verbose)
        except Exception as e:
            print(f"  ERROR en {pid}: {e}")
            continue

        stem = str(result["patient"]) + "_preprocessed.npz"
        out_path = out_dir / stem if out_dir else base.parent / stem
        np.savez_compressed(out_path, **result)
        size_mb = out_path.stat().st_size / 1e6
        summary.append(
            {
                "patient": str(result["patient"]),
                "n_windows": len(result["labels"]),
                "n_arousal": int(result["n_arousal"]),
                "n_nonarousal": int(result["n_nonarousal"]),
                "ratio_pct": float((result["labels"] == 1).mean() * 100),
                "size_mb": size_mb,
                "path": str(out_path),
            }
        )
        if verbose:
            print(f"  -> Guardado: {out_path}  ({size_mb:.1f} MB)\n")

    if verbose:
        print("=" * 60)
        print(f"COMPLETADO: {len(summary)}/{len(patient_bases)} paciente(s) OK")

    return summary


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


# ---------------------------------------------------------------------------
# Exploration plots
# ---------------------------------------------------------------------------


def _shade_labels(ax, arousal_seg: np.ndarray, t_axis: np.ndarray) -> None:
    for val, color, alpha in [
        (1, _EXP_COLORS["arousal"], 0.25),
        (-1, _EXP_COLORS["no_scored"], 0.3),
    ]:
        mask = arousal_seg == val
        changes = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(
                t_axis[s],
                t_axis[min(e, len(t_axis) - 1)],
                color=color,
                alpha=alpha,
                linewidth=0,
            )


def _plot_eeg_panel(ax, sig: np.ndarray, t_axis: np.ndarray, ch_name: str, is_focus: bool) -> None:
    sig_z = (sig - sig.mean()) / (sig.std() + 1e-9)
    color = _EXP_COLORS["eeg_focus"] if is_focus else _EXP_COLORS["eeg_other"]
    lw = 0.5 if is_focus else 0.3
    ax.plot(t_axis, sig_z, lw=lw, color=color)
    ax.set_ylabel(ch_name, fontsize=8, rotation=0, labelpad=60, va="center")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)


def print_channel_stats(signals: np.ndarray) -> None:
    print(f"  {'Canal':15s}  {'min':>8}  {'max':>8}  {'mean':>8}  {'std':>8}")
    print(f"  {'-' * 55}")
    for i, ch in enumerate(ALL_CHANNELS):
        if i >= signals.shape[0]:
            break
        srow = signals[i]
        marker = "  <- FOCO" if i in FOCUS_IDX else ("  (EEG)" if i in EEG_IDX else "")
        print(f"  {ch:15s}  {srow.min():8.1f}  {srow.max():8.1f}  {srow.mean():8.2f}  {srow.std():8.2f}{marker}")
    print()


def plot_overview(
    signals: np.ndarray,
    arousals: np.ndarray,
    sleep_stages: dict[str, np.ndarray],
    base_name: str,
    *,
    viz_start_min: float = VIZ_START_MIN,
    viz_end_min: float = VIZ_END_MIN,
    dpi: int = 130,
) -> Path:
    t0 = int(viz_start_min * 60 * FS)
    t1 = int(viz_end_min * 60 * FS)
    t1 = min(t1, signals.shape[1])
    t_axis = np.arange(t1 - t0) / FS / 60

    arousal_seg = arousals[t0:t1]

    n_rows = len(EEG_IDX) + 2
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.08, height_ratios=[3] * len(EEG_IDX) + [2, 2])
    fig.suptitle(
        f"{base_name} — Senales EEG crudas (min {viz_start_min}–{viz_end_min})\n"
        "Foco: C3-M2 y C4-M1 | Rojo = arousal | Gris = no scored",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )

    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    for row, idx in enumerate(EEG_IDX):
        ax = axes[row]
        _shade_labels(ax, arousal_seg, t_axis)
        ch = ALL_CHANNELS[idx]
        _plot_eeg_panel(ax, signals[idx, t0:t1], t_axis, ch, is_focus=(idx in FOCUS_IDX))
        if row < len(EEG_IDX) - 1:
            ax.tick_params(labelbottom=False)

    ax_lab = axes[-2]
    for val, color, label in [
        (1, _EXP_COLORS["arousal"], "+1 Arousal"),
        (0, _EXP_COLORS["non_arousal"], "0 Non-arousal"),
        (-1, _EXP_COLORS["no_scored"], "-1 No scored"),
    ]:
        mask = arousal_seg == val
        ax_lab.fill_between(t_axis, mask.astype(float) * val, step="mid", color=color, alpha=0.7, label=label)
    ax_lab.set_ylabel("Arousal", fontsize=8, rotation=0, labelpad=60, va="center")
    ax_lab.set_yticks([-1, 0, 1])
    ax_lab.set_yticklabels(["-1", "0", "+1"], fontsize=7)
    ax_lab.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.8)
    ax_lab.spines[["top", "right"]].set_visible(False)
    ax_lab.tick_params(labelbottom=False)

    ax_ss = axes[-1]
    stage_order = ["wake", "nonrem1", "nonrem2", "nonrem3", "rem", "undefined"]
    stage_val = dict(zip(stage_order, [0, 1, 2, 3, 4, 5]))
    y = np.zeros(t1 - t0)
    for stage in stage_order:
        mask = sleep_stages[stage][t0:t1].astype(bool)
        y[mask] = stage_val[stage]
    ax_ss.step(t_axis, y, where="mid", color="#374151", lw=0.8)
    ax_ss.set_yticks(range(6))
    ax_ss.set_yticklabels(["Wake", "N1", "N2", "N3", "REM", "Undef"], fontsize=7)
    ax_ss.set_ylabel("Sleep\nstage", fontsize=8, rotation=0, labelpad=60, va="center")
    ax_ss.set_xlabel("Tiempo (min desde inicio del segmento)", fontsize=9)
    ax_ss.spines[["top", "right"]].set_visible(False)

    out = Path(f"{base_name}_overview.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Guardado: {out}")
    return out


def plot_arousal_zoom(
    signals: np.ndarray,
    arousals: np.ndarray,
    base_name: str,
    *,
    zoom_sec: int = 30,
    dpi: int = 130,
) -> Path | None:
    changes = np.diff((arousals == 1).astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    target = None
    for s, e in zip(starts, ends):
        if (e - s) > 2 * FS:
            target = (s, e)
            break

    if target is None:
        print("  WARN: No se encontro un arousal largo (>2 s) para el zoom.")
        return None

    center = (target[0] + target[1]) // 2
    t0 = max(0, center - zoom_sec * FS)
    t1 = min(signals.shape[1], center + zoom_sec * FS)
    t_axis = (np.arange(t1 - t0) - (center - t0)) / FS

    arousal_seg = arousals[t0:t1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{base_name} — Zoom en arousal (±{zoom_sec}s)\nC3-M2 y C4-M1", fontsize=11, fontweight="bold")

    titles = ["C3-M2", "C4-M1", "Etiqueta"]
    for row, idx in enumerate([2, 3]):
        ax = axes[row]
        _shade_labels(ax, arousal_seg, t_axis)
        sig = signals[idx, t0:t1]
        sig_z = (sig - sig.mean()) / (sig.std() + 1e-9)
        ax.plot(t_axis, sig_z, lw=0.6, color=_EXP_COLORS["eeg_focus"])
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel(titles[row], fontsize=9)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    ax = axes[2]
    ax.fill_between(t_axis, arousal_seg == 1, step="mid", color=_EXP_COLORS["arousal"], alpha=0.8, label="Arousal")
    ax.fill_between(
        t_axis,
        -(arousal_seg == -1).astype(float),
        step="mid",
        color=_EXP_COLORS["no_scored"],
        alpha=0.6,
        label="No scored",
    )
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Etiqueta", fontsize=9)
    ax.set_xlabel("Tiempo (s)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path(f"{base_name}_zoom_arousal.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Guardado: {out}")
    return out


def run_exploration_notebook_style(base_path: str | Path, *, patient_label: str | None = None) -> None:
    base = Path(base_path)
    base_display = patient_label or base.name

    signals, arousals, sleep_stages = load_signals_and_arousals(base, include_sleep_stages=True)

    print(f"\n{'=' * 55}")
    print(f"  Paciente: {base_display}")
    print(f"  Canales:  {signals.shape[0]}")
    describe_recording(arousals)

    print("  Estadisticas por canal:")
    print_channel_stats(signals)

    print("  Generando figura 1: vision general...")
    plot_overview(signals, arousals, sleep_stages, base.name)

    print("  Generando figura 2: zoom en arousal...")
    plot_arousal_zoom(signals, arousals, base.name)

    print("\nListo.")


def _cli_preprocess() -> None:
    p = argparse.ArgumentParser(description="Preprocess PhysioNet Challenge 2018 recordings to .npz windows.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    summary = run_batch(args.data_dir, args.out_dir)
    print_batch_summary(summary)


def _cli_explore() -> None:
    p = argparse.ArgumentParser(description="Plot overview + arousal-zoom figures.")
    p.add_argument("base_path", type=Path)
    args = p.parse_args()
    run_exploration_notebook_style(args.base_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "explore":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        _cli_explore()
    else:
        _cli_preprocess()
