# Sleep Stage Detector

Real-time sleep staging and arousal detection from dual-channel EEG (C3-M2, C4-M1), trained on the [PhysioNet Challenge 2018](https://physionet.org/content/challenge-2018/1.0.0/) dataset and deployable as a lightweight C/Zig inference pipeline.

Two PyTorch models run in parallel every 5 seconds:

| Model | Task | Output |
|-------|------|--------|
| **SleepStageNet** | 4-class sleep staging | Wake, Light Sleep, Deep Sleep, REM |
| **ArousalNet** | Binary arousal detection | Arousal logit |

---

## Architecture

```
Raw EEG (200 Hz, 2 ch)
  → notch + bandpass + downsample (100 Hz)
  → STFT band-power features (every 0.5 s)
  → every 5 s: normalize buffers → model inference
```

The offline training path and the online streaming path share the same DSP parameters, defined in `src/config.py` (`PreprocessConfig`). A Python realtime emulator (`src/realtime/`) mirrors the Zig deploy runtime so you can validate parity before flashing to a device.

See `diagrams/pipeline.dot` for the per-sample processing flowchart.

---

## Project layout

```
sleep_stage_detector/
├── src/
│   ├── config.py              # Shared preprocessing & training defaults
│   ├── networks/              # PyTorch model definitions
│   ├── offline/               # Dataset building, batch DSP, training helpers
│   └── realtime/              # Streaming DSP + inference (Python reference)
├── scripts/
│   ├── build_dataset.py       # Raw WFDB → processed .npz
│   ├── export_and_quantize.py # PyTorch → ONNX → INT8
│   ├── build_test_bin.py      # WFDB → raw .bin for streaming tests
│   └── train/                 # Training entry points
├── deploy/                    # Zig runtime + onnx2c-generated C models
├── models/                    # Trained checkpoints & quantized ONNX
├── notebooks/                 # Exploratory analysis
└── tests/                     # Model verification & parity checks
```

---

## Prerequisites

- **Python 3.10+** with CUDA-capable GPU recommended for training
- **Zig 0.11** (for the deploy runtime; see `deploy/Dockerfile`)
- **onnx2c** — converts quantized ONNX models to standalone C (installed in the deploy Docker image)
- **wfdb** — needed only for `scripts/build_test_bin.py` (`pip install wfdb`)

The root `Dockerfile` targets an internal `ml_base:latest` image for the Python training environment. The deploy toolchain lives in `deploy/Dockerfile`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install onnxruntime wfdb   # export/quantization and .bin conversion
```

Run all Python commands from the repository root so imports resolve correctly.

---

## Data

Download the Challenge 2018 training set (~1 GB):

```bash
./fetch_data.sh
# → data/raw/<patient_id>/
```

Each patient folder contains WFDB signal and annotation files. The pipeline reads EEG channels **C3-M2** and **C4-M1** only.

---

## Workflow

### 1. Build processed datasets

```bash
python scripts/build_dataset.py
```

Processes every patient under `data/raw/`, splits 80/20 (seed 42), and writes:

```
data/processed/arousals/arousals_{train,test}.npz
data/processed/sleep_stage/sleep_stages_{train,test}.npz
```

Or via DVC (note: the DVC output path uses `arousal/` while the script writes to `arousals/`):

```bash
dvc repro
```

### 2. Train models

```bash
python scripts/train/train_sleep_stage.py
python scripts/train/train_arousals.py
```

Checkpoints are saved to `models/sleep_stage/sleep_stage_detector.pt` and `models/arousals/arousal_detector.pt`. Hyperparameters and class weights live in `TrainSleepStagesConfig` and `TrainArousalsConfig` inside `src/config.py`.

### 3. Export and quantize

```bash
python scripts/export_and_quantize.py --task both   # arousal | sleep | both
```

Exports each checkpoint to INT8 ONNX (`*_int8.onnx`) using calibration data from the training split.

### 4. Generate C models (deploy)

Inside the deploy Docker environment, convert ONNX to C with onnx2c and copy the generated `.c`/`.h` files into `deploy/models/`. Pre-generated models are already checked in under `deploy/models/`.

Build the Zig binary:

```bash
cd deploy
zig build -Doptimize=ReleaseFast
# → deploy/zig-out/bin/sleep_stager
```

### 5. Run streaming inference

Convert a WFDB recording to raw binary (interleaved f32, ch0 then ch1 per sample):

```bash
python scripts/build_test_bin.py data/raw/<patient_id>
```

**Python reference pipeline:**

```bash
python -m src.realtime.run outputs/<recording>.bin
# → outputs/py_arousal_preds.csv, outputs/py_sleep_preds.csv
```

**Zig deploy pipeline:**

```bash
./deploy/zig-out/bin/sleep_stager outputs/<recording>.bin
# → outputs/arousal_preds.csv, outputs/sleep_preds.csv
```

Compare outputs:

```bash
python scripts/compare_model_output.py
```

Both pipelines currently stop after 1000 s of data (a test harness limit).

---

## Configuration

All signal-processing constants are centralized in `PreprocessConfig`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| Sample rate | 200 Hz → 100 Hz | `downsample_factor=2` |
| EEG channels | C3-M2, C4-M1 | Index map in `channels` dict |
| Bandpass | 0.5–40 Hz | 4th-order Butterworth |
| Notch | 60 Hz | Power-line rejection |
| STFT hop | 50 samples (0.5 s) | Drives feature update rate |
| Sleep window | 30 s | Spectral context tensor |
| Arousal window | 10 s pre + 5 s post | Raw EEG + 60 s spectral context |
| Inference interval | 5 s | 500 samples at 100 Hz |

Training defaults (epochs, learning rate, class balancing) are in the same file under `TrainSleepStagesConfig` and `TrainArousalsConfig`.

---

## Models

**SleepStageNet** — 1D CNN over a `(20 × 60)` band-power spectrogram (10 frequency bands × 2 EEG channels, past + current 30 s windows). Four output logits.

**ArousalNet** — Dual-branch architecture: a multi-scale temporal CNN on raw EEG plus a context branch on rolling band-power history. Single output logit with focal loss during training.

---

## Notebooks

Exploratory work lives in `notebooks/`:

- `Exploring_signals.ipynb` — raw signal inspection
- `detect_sleep_stages.ipynb` — staging experiments
- `training_context.ipynb` / `training_processed.ipynb` — training analysis

---

## Testing

```bash
python tests/verify_models.py          # deterministic PyTorch forward pass
python tests/plot_logits.py            # visualize logit dumps
python tests/models_error.py           # parity error analysis
```

The C model smoke test is in `tests/test_c_model.c` (compiles against generated headers in `deploy/models/`).
