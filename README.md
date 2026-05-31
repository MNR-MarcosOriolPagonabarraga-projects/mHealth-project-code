# mHealth – PhysioNet Challenge 2018 (sleep arousal)

This repository contains exploratory and preprocessing workflows for sleep-related biosignals used in **PhysioNet / Computing in Cardiology Challenge 2018**, “You Snooze You Win” (classification of arousals during sleep from polysomnography recordings). Challenge materials and citations are documented on PhysioNet: [https://physionet.org/content/challenge-2018/1.0.0/](https://physionet.org/content/challenge-2018/1.0.0/).

The work was prototyped in **`notebooks/`**; reusable code lives in two flat modules under **`src/`**:

- **`process.py`** — constants, MAT/HDF5 loading, 30 s window extraction, filters, PSD export, batch CLI, and matplotlib exploration helpers.
- **`network.py`** — PyTorch **`SleepArousalNet`** (temporal + optional spectral branch) and a small **`PreprocessedWindowDataset`** for `*_preprocessed.npz` files.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `notebooks/preprocessing.ipynb` | Driver: `DATA_DIR` / `OUT_DIR` and calls into `process`. |
| `notebooks/Exploring_signals.ipynb` | Driver: patient stem path and `run_exploration_notebook_style`. |
| `src/process.py` | Preprocessing pipeline and exploration plots. |
| `src/network.py` | Torch model and dataset for preprocessed windows. |
| `pyproject.toml` | Metadata; `pip install -e .` exposes top-level modules `process` and `network`. |
| `requirements.txt` | Core scientific stack (no PyTorch). |

Notebook cells can prepend **`src`** to **`sys.path`** when you skip installation; **`pip install -e .`** is still the recommended workflow.

---

## What the preprocessing pipeline does

For each eligible subject under `DATA_DIR` (matching `**/tr03-*.mat`, excluding `-arousal.mat` companions):

1. Loads the waveform matrix (`val`) from the MATLAB-compatible `.mat` file and arousal annotations from `-arousal.mat` via **h5py**.
2. Keeps EEG **C3-M2** and **C4-M1**, Butterworth band-pass **0.5–40 Hz**, **50 Hz** notch.
3. **30 s** windows at **200 Hz** (**6000** samples).
4. Window label **0** or **1** by majority arousal samples; ambiguous windows skipped.
5. Welch PSD retained in **0.5–40 Hz** (about **159** bins with defaults).

Outputs **`signals`** `(N, 2, 6000)`, **`psd`** `(N, 2, 159)`, **`labels`** `(N,)`, plus metadata arrays.

Exploration helpers in **`process.py`** save **`{recording_id}_overview.png`** and **`{recording_id}_zoom_arousal.png`** unless you extend the plotting API.

---

## PyTorch baseline (`network.py`)

**`SleepArousalNet`** consumes batches shaped like the `.npz` exports:

- `signals`: `(batch, 2, 6000)`
- optional `psd`: `(batch, 2, 159)` — set `use_psd_branch=False` for time-domain only.

**`PreprocessedWindowDataset`** loads numpy arrays eagerly; **`__getitem__`** builds tensors only when queried (and requires Torch).

Importing **`network`** succeeds without installing PyTorch, but **`SleepArousalNet`**, tensor paths in **`PreprocessedWindowDataset`**, and **`logits_to_proba_arousal`** raise **`ImportError`** until you install **`torch`** (use **`pip install -e '.[training]'`**).

Install Torch with:

```bash
pip install -e '.[training]'
```

---

## Development setup

Python **3.10+**.

```bash
cd /path/to/mHealth-project-code
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
# optional GPU/CPU Torch:
pip install -e '.[training]'
```

Core dependencies only:

```bash
pip install -r requirements.txt
```

After editable install:

```python
import process
import network
```

Or run Jupyter with the **`sys.path`** bootstrap already present in **`notebooks/`**.

---

## Data placement

```
Data/
  0005/
    tr03-0005.mat
    tr03-0005-arousal.mat
```

Repository **`.gitignore`** already ignores **`data*`** patterns.

---

## Command-line usage

From the repository root:

```bash
PYTHONPATH=src python src/process.py --data-dir Data --out-dir Processed
PYTHONPATH=src python src/process.py explore Data/0005/tr03-0005
```

Use the **`explore`** subcommand for plots; positional **`base_path`** is the recording stem (**no** `.mat` suffix). Omit **`--out-dir`** to write each **`_preprocessed.npz`** beside the companion **`.mat`**.

Editable install (**`pip install -e .`**) publishes top-level modules **`process`** / **`network`** into the environment (`import process` from notebooks still works alongside the **`PYTHONPATH=src`** pattern above).

---

## API sketch

```python
from pathlib import Path
import process
from network import SleepArousalNet, PreprocessedWindowDataset

signals, arousals = process.load_signals_and_arousals("Data/0005/tr03-0005")

summary = process.run_batch(Path("Data"), Path("Processed"), cfg=process.PreprocessConfig())
process.print_batch_summary(summary)

process.run_exploration_notebook_style(Path("Data/0005/tr03-0005"))

model = SleepArousalNet()
ds = PreprocessedWindowDataset("Processed/tr03-0005_preprocessed.npz")
```

---

## How this project evolved

1. Exploration and preprocessing were Jupyter-first.
2. Logic was flattened into **`src/process.py`** to avoid package boilerplate while keeping imports simple for notebooks.
3. **`src/network.py`** encodes the expected tensor layouts for the next modelling step without pulling Torch into lightweight preprocessing installs.

Contributors should keep heavy numerics/visuals in **`process.py`** and model code in **`network.py`**.

---

## License and citation

Honor PhysioNet’s data-use agreement and cite the Challenge 2018 resource when publishing results derived from those recordings.
