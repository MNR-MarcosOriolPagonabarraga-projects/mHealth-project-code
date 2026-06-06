import h5py
from pathlib import Path
import numpy as np
import scipy.io
from scipy.signal import decimate

from src.config import FS
from src.dsp import compute_full_recording_bandpower


def _base_without_extension(base_path: str | Path) -> Path:
    p = Path(base_path)
    for suffix in (".mat", ".hea"):
        if p.suffix.lower() == suffix:
            return p.with_suffix("")
    return p


def load_signals_and_arousals(
    base_path: str | Path,
    channels: list,
    include_sleep_stages: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load ``val`` from ``.mat`` and ``data/arousals`` from HDF5 ``-arousal.mat``."""
    base = _base_without_extension(base_path)
    mat_path = base / f"{base.name}.mat"
    arousal_path = base / f"{base.name}-arousal.mat"
    print(mat_path, arousal_path)

    mat = scipy.io.loadmat(mat_path)

    signals = mat["val"][channels, :].astype(np.float32)

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


def downsample_signal(signal: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Safe Downsample using smooth decimation to avoid aliasing."""
    if downsample_factor <= 1:
        return signal
    
    return decimate(signal, downsample_factor, ftype='iir', zero_phase=True)


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

    if np.sum(arousal_win == 1) / n > 0.1:  # e.g., >3 seconds in a 30s window
        return 1
    if np.sum(arousal_win == 0) / n > 0.5:
        return 0
    return -1

def extract_event_locked_windows(
    signals: np.ndarray, 
    arousals: np.ndarray, 
    fs: int, 
    pre_sec: int = 20, 
    post_sec: int = 15,
    neg_ratio: int = 2
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    
    pre_samp = pre_sec * fs
    post_samp = post_sec * fs
    win_len = pre_samp + post_samp
    n_channels = signals.shape[0]
    
    out_signals = []
    out_labels = []
    
    # Find all Arousal Onsets (where label goes from 0 to 1)
    onsets = np.where(np.diff(arousals) == 1)[0] + 1 
    
    valid_pos_count = 0
    
    # Extract Positive Windows
    for onset in onsets:
        i0 = onset - pre_samp
        i1 = onset + post_samp
        
        # Check bounds and artifact (-1) presence
        if i0 >= 0 and i1 <= len(arousals):
            if not np.any(arousals[i0:i1] == -1):
                out_signals.append(signals[:, i0:i1])
                out_labels.append(arousals[i0:i1])
                valid_pos_count += 1
                
    # Extract Negative Windows (1:1 or 1:neg_ratio balance)
    target_negatives = valid_pos_count * neg_ratio
    extracted_negatives = 0
    max_attempts = target_negatives * 10 # Prevent infinite loops
    attempts = 0
    
    while extracted_negatives < target_negatives and attempts < max_attempts:
        attempts += 1
        # Pick a random starting point
        i0 = np.random.randint(0, len(arousals) - win_len)
        i1 = i0 + win_len
        
        # Ensure the entire window is purely 0 (no arousals, no artifacts)
        if np.all(arousals[i0:i1] == 0):
            out_signals.append(signals[:, i0:i1])
            out_labels.append(arousals[i0:i1])
            extracted_negatives += 1

    if len(out_labels) == 0:
        clean_signals = np.empty((0, n_channels, win_len), dtype=np.float32)
        clean_labels = np.empty((0, win_len), dtype=np.int8)
    else:
        clean_signals = np.stack(out_signals).astype(np.float32)
        clean_labels = np.stack(out_labels).astype(np.int8)

    return clean_signals, clean_labels


def extract_classification_windows(
    signals: np.ndarray, 
    arousals: np.ndarray, 
    fs: int, 
    win_sec: int = 15,
    ctx_sec: int = 300, # 5 minutes of context
    neg_ratio: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    win_samp = win_sec * fs
    ctx_samp = ctx_sec * fs
    n_channels = signals.shape[0]
    
    # Parameters for the rolling STFT window logic
    n_fft = 512
    hop_length = 200 # 2-second step size
    
    # Pre-compute the entire recording's timeline of spectral features via PyTorch
    full_spectral_timeline = compute_full_recording_bandpower(signals, fs, n_fft, hop_length)
    
    out_signals = []
    out_contexts = []
    out_labels = []
    
    # Find all Arousal Onsets (where label goes from 0 to 1)
    onsets = np.where(np.diff(arousals) == 1)[0] + 1 
    valid_pos_count = 0
    offset_sec = 10
    
    # Calculate how many historical steps we need to grab for a 5-minute block
    # (Context Duration - Window Size) / Step Size + 1
    required_ctx_steps = int((ctx_sec - (n_fft / fs)) / (hop_length / fs)) + 1 # Matches your 148 steps
    
    for onset in onsets:
        pre_samp = offset_sec * fs
        i0 = onset - pre_samp
        i1 = i0 + win_samp
        
        # Bound check: Ensure there is enough room behind i0 for a 5-minute context
        if (i0 - ctx_samp) >= 0 and i1 <= len(arousals):
            if not np.any(arousals[i0:i1] == -1):
                out_signals.append(signals[:, i0:i1])
                
                # Convert the raw audio sample index (i0) back to the matching STFT time-step index
                # Mapping formula: step_index = sample_index // hop_length
                end_step_idx = i0 // hop_length
                start_step_idx = end_step_idx - required_ctx_steps
                
                # Slice the pre-computed feature matrix to get the exact (148, n_channels * 5) block
                ctx_feats = full_spectral_timeline[start_step_idx:end_step_idx, :]
                out_contexts.append(ctx_feats)
                
                out_labels.append(1)
                valid_pos_count += 1
            
    # Extract Negative Windows (Pure Sleep)
    target_negatives = valid_pos_count * neg_ratio
    extracted_negatives = 0
    max_attempts = target_negatives * 10 
    attempts = 0
    
    while extracted_negatives < target_negatives and attempts < max_attempts:
        attempts += 1
        # Start sampling indexes strictly after the 5-minute context requirements
        i0 = np.random.randint(ctx_samp, len(arousals) - win_samp)
        i1 = i0 + win_samp
        
        # Ensure the entire 15s window is purely 0
        if np.all(arousals[i0:i1] == 0):
            out_signals.append(signals[:, i0:i1])
            
            # Pull matching context slice from the pre-computed spectral matrix
            end_step_idx = i0 // hop_length
            start_step_idx = end_step_idx - required_ctx_steps
            ctx_feats = full_spectral_timeline[start_step_idx:end_step_idx, :]
            out_contexts.append(ctx_feats)
            
            out_labels.append(0)
            extracted_negatives += 1

    if len(out_labels) == 0:
        clean_signals = np.empty((0, n_channels, win_samp), dtype=np.float32)
        clean_contexts = np.empty((0, required_ctx_steps, n_channels * 5), dtype=np.float32)
        clean_labels = np.empty((0,), dtype=np.int8) 
    else:
        clean_signals = np.stack(out_signals).astype(np.float32)
        clean_contexts = np.stack(out_contexts).astype(np.float32)
        clean_labels = np.array(out_labels, dtype=np.int8)

    return clean_signals, clean_contexts, clean_labels

def find_stable_blocks(stage_mask: np.ndarray, min_len_samp: int) -> list:
    """
    Finds contiguous blocks of 1s in a boolean array that are >= min_len_samp.
    Returns a list of (onset_index, offset_index) tuples.
    """
    edges = np.diff(np.concatenate(([0], stage_mask, [0])))
    onsets = np.where(edges == 1)[0]
    offsets = np.where(edges == -1)[0]
    
    valid_blocks = []
    for on, off in zip(onsets, offsets):
        if (off - on) >= min_len_samp:
            valid_blocks.append((on, off))
            
    return valid_blocks
