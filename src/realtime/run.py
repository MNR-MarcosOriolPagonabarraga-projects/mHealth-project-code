import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

from src.config import PreprocessConfig, TrainArousalsConfig, TrainSleepStagesConfig
from src.networks.arousal_net import ArousalNet
from src.networks.sleep_stage_net import SleepStageNet

from .provider import SampleProvider
from .dsp import CausalStatefulFilter, Downsampler, zscore_normalize
from .process import StreamProcessor

def load_model(model_class: torch.nn.Module, path: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_class.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model_class.eval()
    return model_class

def run_pipeline(bin_path: Path, arousal_out: Path, sleep_out: Path):
    cfg = PreprocessConfig()
    arousal_cfg = TrainArousalsConfig()
    sleep_cfg = TrainSleepStagesConfig()
    num_channels = len(cfg.channels)

    print("[*] Initializing Models...")
    arousal_model = load_model(ArousalNet(), arousal_cfg.out_path)
    sleep_model = load_model(SleepStageNet(), sleep_cfg.out_path)

    print("[*] Initializing DSP Pipelines...")
    # Instantiate 1 stateful filter and 1 downsampler PER channel
    filters = [
        CausalStatefulFilter(cfg.original_fs, cfg.bp_low_hz, cfg.bp_high_hz, cfg.notch_freq_hz, cfg.notch_q) 
        for _ in range(num_channels)
    ]
    downsamplers = [Downsampler(cfg.downsample_factor) for _ in range(num_channels)]
    
    provider = SampleProvider(bin_path, num_channels)
    processor = StreamProcessor(cfg)
    
    arousal_preds, sleep_preds = [], []

    print(f"[*] Beginning Real-Time Stream from {bin_path.name}...")
    
    for raw_sample in tqdm(provider, desc="Processing Stream", unit="samp"):
        ds_sample = np.zeros(num_channels, dtype=np.float32)
        valid_ds = False
        
        # 1. Process sample per channel (filters + downsampler)
        for c in range(num_channels):
            filt_val = filters[c].process(raw_sample[c])
            ds_val = downsamplers[c].push(filt_val)
            
            if ds_val is not None:
                ds_sample[c] = ds_val
                valid_ds = True
                
        if valid_ds:
            processor.push(ds_sample)

            if processor.sample_counter >= 100000:
                print(f"\n[!] Reached 1000s limit ({processor.sample_counter} samples). Breaking early for test...")
                break

            # 3. Check for 5-second inference trigger
            if processor.sample_counter % (5 * cfg.fs) == 0 and processor.sample_counter > 0:
                current_time_s = processor.sample_counter / cfg.fs

                with torch.no_grad():
                    # --- Arousal Inference ---
                    t_arr = zscore_normalize(processor.arousal_temporal.get_ordered())
                    c_arr = zscore_normalize(processor.arousal_context.get_ordered())
                    
                    t_tensor = torch.tensor(t_arr).unsqueeze(0)
                    c_tensor = torch.tensor(c_arr).unsqueeze(0)
                    a_logit = arousal_model(t_tensor, c_tensor).item()

                    arousal_preds.append({"timestamp_s": current_time_s, "arousal_logit": a_logit})

                    # --- Sleep Stage Inference ---
                    s_arr = zscore_normalize(processor.extract_sleep_tensor())
                    s_tensor = torch.tensor(s_arr).unsqueeze(0)
                    s_logits = sleep_model(s_tensor).squeeze().numpy()

                    sleep_preds.append({
                        "timestamp_s": current_time_s,
                        "wake": s_logits[0],
                        "light_sleep": s_logits[1],
                        "deep_sleep": s_logits[2],
                        "rem": s_logits[3]
                    })

    pd.DataFrame(arousal_preds).to_csv(arousal_out, index=False)
    pd.DataFrame(sleep_preds).to_csv(sleep_out, index=False)
    print(f"[+] Emulation complete. Outputs saved to {arousal_out} and {sleep_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bin", type=str, help="Path to raw binary test data")
    args = parser.parse_args()

    run_pipeline(
        bin_path=Path(args.bin),
        arousal_out=Path("outputs/py_arousal_preds.csv"),
        sleep_out=Path("outputs/py_sleep_preds.csv")
    )