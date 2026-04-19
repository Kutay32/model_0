# Privacy-Preserving Learning for Breast Mammography

Capstone codebase: **centralized CLIP + UNet++** training and a browser demo on CBIS-DDSM-style data, plus **Harmonia Vision** — federated mass segmentation (Flower / FedAvg) with optional CKKS-style aggregation via TenSEAL.

> **Non-clinical research software.** Not for diagnostic use.

**Slides / narrative context:** [`docs/PRESENTATION_CONTEXT.md`](docs/PRESENTATION_CONTEXT.md)  
**Harmonia Vision (FL, Hub, Docker):** [`harmonia_vision/README.md`](harmonia_vision/README.md)

| | |
|---|---|
| **Title** | Privacy-Preserving Federated Learning System for Breast Tumor Segmentation Using Deep Neural Networks |
| **Program** | B.Sc. Software Engineering — Faculty of Engineering and Natural Sciences |
| **Institution** | Istinye University |
| **Date** | January 2026 |
| **Supervisor** | Asst. Prof. Wadhah Zeyad Tareq |


---

## Repository structure

| Area | Role |
|------|------|
| [`mammography/`](mammography/) | CBIS I/O, manifests, CLIP / UNet++ / multitask training |
| [`api/`](api/) | FastAPI inference + static mount for [`web/`](web/) |
| [`harmonia_vision/`](harmonia_vision/) | FL data pipeline, Flower server/client, benchmark, Harmonia Hub |
| [`scripts/`](scripts/) | PowerShell helpers for training, demo, preprocessing, Hub |
| [`requirements-*.txt`](requirements-torch-gpu.txt) | Split deps: GPU Torch, training, demo app |

**Outputs stay local (not committed):** raw CBIS trees, `mammography/cache/`, `harmonia_processed*/`, `harmonia_checkpoints/`, weight files (`*.pth`, etc.), logs. See [`.gitignore`](.gitignore). Obtain CBIS-DDSM separately and run preprocessing / manifest steps on your machine.

---

## Prerequisites

- **Python 3.10+** (3.11+ recommended for Harmonia Vision)
- **NVIDIA GPU** optional but expected for mammography training; use CUDA-matched PyTorch wheels
- **CBIS-DDSM** extract with `csv/` and `jpeg/` (and DICOMs as provided by your bundle)

---

## Environment (repo root)

Create a venv at the repo root (`.venv/` is assumed by the scripts below):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
# GPU PyTorch first (edit file if your CUDA major differs)
.\.venv\Scripts\pip install -r requirements-torch-gpu.txt
.\.venv\Scripts\pip install -r requirements-train.txt
.\.venv\Scripts\pip install -r requirements-app.txt
```

Harmonia modules additionally use [`harmonia_vision/requirements.txt`](harmonia_vision/requirements.txt):

```powershell
.\.venv\Scripts\pip install -r harmonia_vision\requirements.txt
```

**GPU check**

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

If CUDA is unavailable after install, reinstall the GPU wheel stack from `requirements-torch-gpu.txt` (avoid `+cpu` builds for training).

---

## Stack A — Mammography (centralized)

### 1. Manifests + training

Point `CBIS_ROOT` at the dataset root, then use the bundled script (manifests go to `mammography/cache/`, checkpoints to `mammography/checkpoints/`):

```powershell
$env:CBIS_ROOT = "C:\path\to\cbis-ddsm"
.\scripts\train_cbis_mammogram.ps1
```

**Full pipeline** (same idea, alternate entry): `.\scripts\run_full_training.ps1`  
**GPU only** (manifests already built): `.\scripts\run_mammo_gpu_train.ps1`

**Manual steps** (equivalent to what the scripts run):

```powershell
$env:CBIS_ROOT = "C:\path\to\cbis-ddsm"
.\.venv\Scripts\python mammography\build_manifest.py --data-root $env:CBIS_ROOT --out-dir mammography\cache
.\.venv\Scripts\python mammography\train_unetplusplus.py --manifest mammography\cache\manifest_segmentation.csv --checkpoint-dir mammography\checkpoints
.\.venv\Scripts\python mammography\train_clip_mammogram.py --manifest mammography\cache\manifest_classification.csv --checkpoint-dir mammography\checkpoints
```

Optional joint model: `mammography\train_multitask.py` (see script `--help`).

### 2. Browser demo + API

Requires trained weights under `mammography/checkpoints/` (e.g. `best_clip_classifier.pth`, `best_unetplusplus_model.pth`).

```powershell
$env:MODEL_ROOT = "$PWD\mammography\checkpoints"
.\.venv\Scripts\uvicorn api.main:app --host 127.0.0.1 --port 8080
```

- UI: **http://127.0.0.1:8080/**  
- Health: **http://127.0.0.1:8080/api/health**

Shortcut: **`.\scripts\run_demo.ps1`**

| Piece | Location |
|-------|----------|
| Backend | [`api/main.py`](api/main.py) |
| Frontend | [`web/index.html`](web/index.html) + [`web/static/`](web/static/) |

---

## Stack B — Harmonia Vision (federated / benchmark)

From the **repository root**, set `PYTHONPATH` so `harmonia_vision` imports resolve:

```powershell
$env:PYTHONPATH = $PWD   # repo root
```

**Preprocess CBIS** into per-client `.npy` trees (IID or non-IID):

```powershell
python -m harmonia_vision.data_pipeline --dataset-root $env:CBIS_ROOT --out-root harmonia_processed --split-mode iid
```

**Smoke / full CBIS** automation (PowerShell): `.\scripts\preprocess_cbis_full.ps1` — adjust paths inside the script for your machine.

**Centralized or simulated FL** (same processed folder, different `--mode`):

```powershell
python -m harmonia_vision.benchmark --mode centralized --data-root harmonia_processed --preset standard --checkpoint harmonia_checkpoints\centralized.pth
```

**Harmonia Hub** (start/stop jobs, logs, checkpoints, quick mask preview): **http://127.0.0.1:8765**

```powershell
.\scripts\run_harmonia_hub.ps1
```

| Variable | Purpose |
|----------|---------|
| `PYTHONPATH` | Must include repo root for `harmonia_vision` |
| `HARMONIA_HUB_DATA_ROOT` | Default processed data path for the Hub form |
| `HARMONIA_HUB_TOKEN` | Optional bearer token for Hub Start/Stop |
| `HARMONIA_PHE` | `1` to exercise encrypted aggregation path (TenSEAL CKKS) on server/client |

Flower **server** / **client** processes, Docker Compose, TLS dev certs, and CKKS notes are documented in [`harmonia_vision/README.md`](harmonia_vision/README.md).

---

## Script index (`scripts/`)

| Script | Purpose |
|--------|---------|
| `run_demo.ps1` | Mammography FastAPI demo on port 8080 |
| `train_cbis_mammogram.ps1` | Manifests + UNet++ + CLIP training |
| `run_full_training.ps1` | Full mammography training pipeline |
| `run_mammo_gpu_train.ps1` | Training only (skip manifest build) |
| `preprocess_cbis_full.ps1` | Full CBIS → `harmonia_processed_cbis_full/` via `data_pipeline` |
| `run_harmonia_hub.ps1` | Harmonia Hub on port 8765 |
| `run_cbis_long_train_visible.ps1` | Long centralized run (paths inside file) |
| `check_gpu.py` | Quick Torch/CUDA probe |
| `preprocess_cbis.py` | Optional CBIS → `.npy` + `manifest_dataset.csv` export (see script docstring) |
| `remote_*.ps1` / `remote_linux_setup_and_train.sh` | Optional remote helpers |

---

## Course report vs this repo

If you attach the academic PDF, keep it under something like `docs/` and link it from slides. Any MRI-focused narrative in the PDF is separate from the **mammography + CBIS-DDSM** metrics and demos implemented here.

---

## License

Capstone materials belong to the authors and Istinye University unless otherwise stated. Third-party libraries retain their original licenses.
