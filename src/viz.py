import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from pathlib import Path

from src.utils import load_signals_and_arousals, describe_recording
from src.config import FS, ALL_CHANNELS, EEG_IDX, FOCUS_IDX, VIZ_START_MIN, VIZ_END_MIN, _EXP_COLORS


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


def run_exploration(base_path: str | Path, *, patient_label: str | None = None) -> None:
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


def plot_history(history: dict, save_path: Path):
    """
    Universally plots training history metrics.
    Automatically detects loss keys and any other evaluation metrics present.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Identify what metrics are present in the dictionary dynamically
    loss_keys = [k for k in history.keys() if 'loss' in k]
    metric_keys = [k for k in history.keys() if 'loss' not in k]
    
    # Setup a clean 1-row, 2-column plot grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    # 2. Universal Loss Subplot (Left)
    for key in loss_keys:
        # Standardize labels for the legend (e.g., 'train_loss' -> 'Train Loss')
        clean_label = key.replace('_', ' ').title()
        ax1.plot(epochs, history[key], marker='o', label=clean_label, linewidth=1.5)
        
    ax1.set_title("Model Loss Progression", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Epochs", fontsize=10)
    ax1.set_ylabel("Loss Value", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9)
    
    # 3. Universal Metrics Subplot (Right)
    if metric_keys:
        for key in metric_keys:
            clean_label = key.replace('_', ' ').title().replace('Val ', 'Validation ')
            
            # Check if metrics are raw fractions (0.0 to 1.0) and convert to percentages if true
            values = history[key]
            if all(0.0 <= v <= 1.01 for v in values if v is not None):
                values = [v * 100 for v in values]
                y_label = "Score (%)"
            else:
                y_label = "Value"
                
            ax2.plot(epochs, values, marker='s', label=clean_label, linewidth=1.5)
            
        ax2.set_title("Validation Performance Metrics", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Epochs", fontsize=10)
        ax2.set_ylabel(y_label, fontsize=10)
        
        # If it's a percentage axis, bound it nicely between 0 and 105%
        if y_label == "Score (%)":
            ax2.set_ylim(-5, 105)
            
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=9)
    else:
        # Clean fallback if no validation metrics were gathered yet
        ax2.text(0.5, 0.5, "No Validation Metrics Tracked", ha='center', va='center', fontsize=12)
        ax2.axis('off')
        
    plt.suptitle("Model Training History Summary", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save chart to the designated run folder
    save_file = save_path / "training_history_curves.png"
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved training evolution curves to: {save_file.name}")


def plot_epoch_confusion_matrix(tp, fp, fn, tn, save_path):
    """
    Generates and saves a clean Seaborn heatmap of the confusion matrix for a specific epoch.
    """
    # Construct the standard 2x2 confusion matrix array
    # Format: [[TN, FP], [FN, TP]]
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Generate labels containing both raw count and relative percentage
    total_samples = np.sum(cm) + 1e-8
    labels = [
        [f"{tn}\n({tn/total_samples*100:.1f}%)", f"{fp}\n({fp/total_samples*100:.1f}%)"],
        [f"{fn}\n({fn/total_samples*100:.1f}%)", f"{tp}\n({tp/total_samples*100:.1f}%)"]
    ]
    labels = np.array(labels)
    
    # Plot using a clean, readable color palette (Blues highlight true predictions nicely)
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        cbar=False, 
        ax=ax,
        xticklabels=["No Arousal", "Arousal"],
        yticklabels=["No Arousal", "Arousal"],
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    ax.set_title(f"Confusion Matrix", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Ground-Truth Label", fontsize=11, labelpad=10)
    
    plt.tight_layout()
    # Save with a sequential naming format inside your timestamped RUN_DIR
    plt.savefig(save_path / f"confusion_matrix.png", dpi=150)
    plt.close(fig)