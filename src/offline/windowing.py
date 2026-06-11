import numpy as np 

def label_window(arousal_win: np.ndarray) -> int:
    n = len(arousal_win)
    if n == 0:
        return -1

    if np.sum(arousal_win == 1) / n > 0.1:  # e.g., >3 seconds in a 30s window
        return 1
    if np.sum(arousal_win == 0) / n > 0.5:
        return 0
    return -1

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
    n_fft = 256
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
    
    # Define a safe margin (e.g., 15 seconds)
    safe_margin = 15 * fs

    while extracted_negatives < target_negatives and attempts < max_attempts:
        attempts += 1
        i0 = np.random.randint(ctx_samp, len(arousals) - win_samp)
        i1 = i0 + win_samp
        
        # Check if we have room for the safe margin
        if i0 - safe_margin >= 0 and i1 + safe_margin < len(arousals):
            # NEW: Ensure the 15s window AND the 15s buffer on either side are purely 0
            if np.all(arousals[i0 - safe_margin : i1 + safe_margin] == 0):
                out_signals.append(signals[:, i0:i1])
                
                # Pull matching context slice
                end_step_idx = i0 // hop_length
                start_step_idx = end_step_idx - required_ctx_steps
                ctx_feats = full_spectral_timeline[start_step_idx:end_step_idx, :]
                out_contexts.append(ctx_feats)
                
                out_labels.append(0)
                extracted_negatives += 1

    return out_signals, out_contexts, out_labels


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


