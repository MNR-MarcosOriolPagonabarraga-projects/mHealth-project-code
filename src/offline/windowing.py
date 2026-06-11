import numpy as np

# ==========================================
# PRIVATE HELPERS (Keeps main functions clean)
# ==========================================

def _get_historical_context(timeline: np.ndarray, target_samp: int, hop_length: int, required_steps: int) -> np.ndarray:
    """Slices a continuous historical block (e.g., 5 minutes prior) from the STFT timeline."""
    end_step = target_samp // hop_length
    start_step = end_step - required_steps
    return timeline[start_step:end_step, :]

def _get_doublet_context(timeline: np.ndarray, start_samp: int, win_samp: int, hop_length: int) -> np.ndarray:
    """Slices and concatenates [Past, Current] blocks for real-time sleep stage context."""
    c_start = start_samp // hop_length
    c_end = (start_samp + win_samp) // hop_length
    
    p_start = (start_samp - win_samp) // hop_length
    p_end = c_start
    
    return np.concatenate([
        timeline[p_start:p_end, :],
        timeline[c_start:c_end, :]
    ], axis=1)

def _merge_sleep_stage_masks(sleep_stages: dict) -> dict[int, np.ndarray]:
    """Combines string-based sleep stage masks into target integer classes."""
    stage_map = {
        0: ['wake'],
        1: ['nonrem1', 'nonrem2'],  # Light Sleep
        2: ['nonrem3'],             # Deep Sleep
        3: ['rem']
    }
    
    merged = {}
    for label, keys in stage_map.items():
        mask = None
        for k in keys:
            if k in sleep_stages:
                mask = sleep_stages[k].copy() if mask is None else (mask | sleep_stages[k])
        if mask is not None:
            merged[label] = mask
            
    return merged

# ==========================================
# PUBLIC API
# ==========================================

def find_stable_blocks(stage_mask: np.ndarray, min_len_samp: int) -> list[tuple[int, int]]:
    """Finds contiguous blocks of 1s in a boolean array that are >= min_len_samp."""
    edges = np.diff(np.concatenate(([0], stage_mask, [0])))
    onsets = np.where(edges == 1)[0]
    offsets = np.where(edges == -1)[0]
    
    return [(on, off) for on, off in zip(onsets, offsets) if (off - on) >= min_len_samp]


def extract_arousal_windows(
    signals: np.ndarray, 
    arousals: np.ndarray, 
    full_spectral_timeline: np.ndarray, # <-- Now passed in to match sleep stages!
    fs: int, 
    pre_sec: int = 10,
    post_sec: int = 5,
    ctx_sec: int = 300,
    hop_len: int = 50,
    n_fft: int = 256,
    neg_ratio: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts high-resolution EEG windows and their low-resolution STFT context for Arousals."""
    win_samp = (pre_sec + post_sec) * fs
    ctx_samp = ctx_sec * fs
    
    out_signals, out_contexts, out_labels = [], [], []
    required_ctx_steps = int((ctx_sec - (n_fft / fs)) / (hop_len / fs)) + 1
    
    # 1. Extract Positive Windows (Arousal Onsets)
    onsets = np.where(np.diff(arousals) == 1)[0] + 1 
    offset_samp = pre_sec * fs
    valid_pos_count = 0
    
    for onset in onsets:
        start_samp = onset - offset_samp
        end_samp = start_samp + win_samp
        
        if (start_samp - ctx_samp) >= 0 and end_samp <= len(arousals):
            if not np.any(arousals[start_samp:end_samp] == -1): # No artifacts
                out_signals.append(signals[:, start_samp:end_samp])
                out_contexts.append(_get_historical_context(full_spectral_timeline, start_samp, hop_len, required_ctx_steps))
                out_labels.append(1)
                valid_pos_count += 1
                
    # 2. Extract Negative Windows (Pure Sleep)
    target_negatives = valid_pos_count * neg_ratio
    safe_margin = 15 * fs
    attempts, extracted_negatives = 0, 0
    
    while extracted_negatives < target_negatives and attempts < target_negatives * 10:
        attempts += 1
        start_samp = np.random.randint(ctx_samp, len(arousals) - win_samp)
        end_samp = start_samp + win_samp
        
        if start_samp - safe_margin >= 0 and end_samp + safe_margin < len(arousals):
            # Ensure the window AND the surrounding buffer are purely 0
            if np.all(arousals[start_samp - safe_margin : end_samp + safe_margin] == 0):
                out_signals.append(signals[:, start_samp:end_samp])
                out_contexts.append(_get_historical_context(full_spectral_timeline, start_samp, hop_len, required_ctx_steps))
                out_labels.append(0)
                extracted_negatives += 1

    return {
        "eeg_windows": out_signals,
        "context_windows": out_contexts,
        "labels": out_labels
    }


def extract_sleep_stage_windows(
    full_spectral_timeline: np.ndarray, 
    sleep_stages: dict, 
    fs: int, 
    hop_length: int, 
    win_sec: int = 30
) -> tuple[list, list]:
    """Extracts consecutive [Past, Current, Future] context windows for sleep staging."""
    win_samp = win_sec * fs
    all_features, all_labels = [], []
    
    merged_stages = _merge_sleep_stage_masks(sleep_stages)
    
    for stage_int, combined_mask in merged_stages.items():
        blocks = find_stable_blocks(combined_mask, win_samp)
        
        for on, off in blocks:
            # Step by win_samp to avoid overlapping the target windows
            for start_samp in range(on + win_samp, off - (2 * win_samp) + 1, win_samp):
                context_doublet = _get_doublet_context(full_spectral_timeline, start_samp, win_samp, hop_length)
                
                all_features.append(context_doublet)
                all_labels.append(stage_int)
                
    return {
        "sleep_features": all_features,
        "labels": all_labels
    }
