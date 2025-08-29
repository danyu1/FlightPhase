# 🏃‍♂️ FlightPhase — WorldAthletics Scraper & Hierarchical LSTM Forecaster

> End-to-end pipeline to **scrape** collegiate T&F performances from World Athletics, **cache & preprocess** them, and **forecast next-season peaks** with a hierarchical LSTM that understands **per-season sequences** and **per-event heads**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/placeholder/flightphase-hero.gif" alt="Demo" width="760">
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.x-EE4C2C"></a>
  <a href="#"><img alt="Selenium" src="https://img.shields.io/badge/selenium-ready-43B02A"></a>
  <a href="#"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-optional-76B900"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

---

## ✨ Highlights

- **Fast, polite scraping** of TFRRS “All Performances” pages (w/ hard 5s caps, blocked assets, explicit waits).
- **Noise-free logs**: silences Chrome/TFLite native STDERR spam.
- **Caching for speed**: compute once → save as `.parquet` / `.pkl` (safe fallback) + `.npz` tensors.
- **Smart preprocessing**: per-family (athlete×gender×event-family) **season sequences** & **peak labels**.
- **Hierarchical LSTM**:  
  - LSTM#1 encodes **within-season** marks (value, day gaps, wind) → season embedding  
  - SeasonLabel embedding (Indoor/Outdoor) is concatenated  
  - LSTM#2 encodes **across-season** history (up to *K* seasons) → family embedding  
  - **Per-event heads** produce forecasts in z-space; de-standardized per event.
- **Tidy training**: time-aware splits, per-event normalization, early stopping, progress bars via `tqdm`.
- **User-friendly prediction CLI**: interactive prompts, minimal typing, instant forecasts.

> **Goal:** “Given my marks this season, what’s my **peak** next season (same label) likely to be?”

---

## 🧠 Model at a Glance (Mermaid)

```mermaid
flowchart LR
  subgraph Season
    direction LR
    X[(marks: value, Δdays, wind)]
    label1[[a season (sequence of marks)]]
  end
  X --> LSTM1((Season LSTM))
  Lab[SeasonLabel Embedding] -->|concat| CAT[Concat]
  LSTM1 --> CAT
  subgraph Fam
    direction LR
    label2[[Family history (K seasons)]]
    CAT --> LSTM2((Across-Season LSTM))
  end
  LSTM2 --> FEmb[Family Embedding]
  FEmb -->|per event| Heads{Event-Specific Heads}
  Heads --> ZPred[Peak (z-space)]
  ZPred -->|de-normalize by event| Pred[Peak (native units)]
```

## 📦 Project Structure

```
.
├─ ScrapeFromWorldAthletics_Fast.py        # scrape → structured .npy (or .csv)
├─ ForecastHGF-LSTM.ipynb         # training notebook (prep + train + metrics)
├─ utils_cache_io.py              # parquet/pickle-safe I/O helpers
├─ models_HierGenderFamilies/
│  ├─ Men/
│  │  ├─ model.pth               # trained weights
│  │  └─ meta.json               # training metadata (events, z-norms, dims, etc.)
│  └─ Women/
├─ PredictionScript.py            # interactive CLI: “put in my marks → next season peak”
└─ README.md
```

## 🚀 Quickstart

### 0) Prereqs

```bash
# Windows PowerShell example
python -m venv .venv
.venv\Scripts\activate

pip install -U pip wheel
pip install numpy pandas beautifulsoup4 lxml requests tqdm
pip install selenium webdriver-manager
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or CPU wheel
# Parquet optional (we auto-fallback to pickle if missing)
pip install pyarrow fastparquet || echo "Parquet engines are optional"
```

> Chrome is required for Selenium (Chromium/Edge also okay). WebDriver is auto-managed.

---

### 1) Scrape (fast, skip slow pages)

```bash
python ScrapeFromWorldAthletics_Fast.py -i WorldAthletics_all_ncaa_urls.csv -o WorldAthletics_performances_fast.npy --limit 500
```

> 5s hard cap per URL. Blocks heavy assets (png/jpg/mp4/woff/...).

---

### 2) Precompute & Cache (run once)

```python
from utils_cache_io import save_table, load_table, save_dict_table, load_dict_table, save_seqs_npz
from datetime import datetime
import os, json, pandas as pd, numpy as np

# 1) Load marks (from .npy or .csv → DataFrame)
df = load_df("WorldAthletics_performances_fast.npy")   # your helper that parses times/distances/dates
peaks_all = build_family_season_peaks(df)
seqs, lens, evt_used = build_mark_sequences_by_family(df, seqlen=32)

# 2) Save cache (parquet if available, else pickle)
paths = cache_paths("artifacts")
save_table(df,        paths["df"])
save_table(peaks_all, paths["peaks"])
save_dict_table(lens,     paths["lens"], "orig_len")
save_dict_table(evt_used, paths["evt"],  "event")
save_seqs_npz(seqs, paths["seqs"])

with open(paths["manifest"], "w") as f:
    json.dump({"created_utc": datetime.utcnow().isoformat()+"Z"}, f, indent=2)
```

---

### 3) Train (Hierarchical LSTM)

```python
# In notebook or script
args = build_defaults()
df         = load_table(paths["df"])
peaks_all  = load_table(paths["peaks"])
lens       = load_dict_table(paths["lens"])
evt_used   = load_dict_table(paths["evt"])
seqs       = load_seqs_npz(paths["seqs"])

# Build windows → split → DataLoaders → train
# (See notebook cell "Training Loop" – includes tqdm bars + early stopping)
```

> X: [N, K, L, F] (F=3 → value, Δdays, wind)

---

### 4) Predict (Interactive CLI)

```bash
python PredictionScript.py
```

> Prompts:
> - Model dir (e.g., `models_HierGenderFamilies/Men`)
> - Season label: Indoors or Outdoors
> - Event name
> - Paste marks:
```
2025-03-29, 13.31, +0.2
2025-04-17, 12.80, +2.2
2025-04-26, 12.77, +2.6
2025-05-09, 13.25, +3.3
```

> Output:
```
Predicted NEXT season (Outdoors) peak for Triple Jump: 13.54m
```

---

## 🛠️ Configuration (edit in-file)

- `SEQLEN_MARKS`, default `32`
- `K_SEASONS`, default `4`
- `HID_SEASON`, `HID_ACROSS`
- `EMB_LABEL` (SeasonLabel emb dim)
- Batch size / LR / Epochs / Early stopping
- Min train samples per event

---

## 🧪 Tips & Troubleshooting

<details><summary><b>Parquet / ArrowKeyError</b></summary>

```pgsql
ArrowKeyError: A type extension with name pandas.period already defined
```

> Fix: fallback to pickle

```bash
set FP_FORCE_PICKLE=1      # Windows
export FP_FORCE_PICKLE=1   # macOS/Linux
```

</details>

<details><summary><b>CUDA device-side assert / mysterious GPU errors</b></summary>

```bash
set CUDA_LAUNCH_BLOCKING=1      # Windows
export CUDA_LAUNCH_BLOCKING=1   # macOS/Linux
```

</details>

<details><summary><b>state_dict key mismatch when loading model</b></summary>

> Ensure model dim, emb size, and number of event heads match training config.

</details>

<details><summary><b>Selenium stalls / long page waits</b></summary>

> Restart the run. Block more asset types. Shard scraping jobs if needed.

</details>

---

## 🧩 Example Snippets

### Minimal prediction (programmatic)

```python
from PredictionScript import load_model_from_dir, build_inputs_for_window
from datetime import datetime
import torch, numpy as np

model, meta, evt_list, mu_by_event, sd_by_event, seqlen, k_seasons = load_model_from_dir(
    r"models_HierGenderFamilies\Men", device="cpu"
)

marks = [
  (datetime(2025,3,29), 13.31, +0.2),
  (datetime(2025,4,17), 12.80, +2.2),
  (datetime(2025,4,26), 12.77, +2.6),
  (datetime(2025,5,9),  13.25, +3.3),
]

X, Lk, LAB, E = build_inputs_for_window("Outdoors", "Triple Jump", marks, seqlen, k_seasons, evt_list, "cpu")
with torch.no_grad():
    z = model(X, Lk, LAB, E).item()
pred = z * sd_by_event["Triple Jump"] + mu_by_event["Triple Jump"]
print(pred)
```

---

### Training loop skeleton

```python
for gender in sorted(df["Gender"].unique()):
    # build windows → split (train<=2023, val==2024) → dataloaders
    # init model from dims in data → train with AdamW + L1Loss
    # early stop by val L1; save best .pth + meta.json
    ...
```

---

## 🧭 Data & Labels

We predict the next-season peak for the **same season label** as the most recent one (Indoor→Indoor, etc.), using z-normalized peaks per event and restoring via train-time μ/σ.

---

## 🧾 License

MIT — do cool things, attribute when you can. Please be a good citizen to WorldAthletics (polite delays, low QPS, cache aggressively).
