# Capstone: Privacy-Preserving Federated Learning for Breast Tumor Segmentation

## This repository (mammography)

The **live pipeline** trains and serves **CLIP + UNet++** on **CBIS-DDSM**-style mammogram patches. Checkpoints and the web demo expect weights under `mammography/checkpoints/` (`best_clip_classifier.pth`, `best_unetplusplus_model.pth`).

**Slide / talk references (web):** [`docs/PRESENTATION_CONTEXT.md`](docs/PRESENTATION_CONTEXT.md)

| | |
|---|---|
| **Title** | Privacy-Preserving Federated Learning System for Breast Tumor Segmentation Using Deep Neural Networks |
| **Program** | B.Sc. Software Engineering — Faculty of Engineering and Natural Sciences |
| **Institution** | Istinye University |
| **Date** | January 2026 |
| **Supervisor** | Asst. Prof. Wadhah Zeyad Tareq |
| **Students** | Kutay ORALLI (220911791), Eyüpcan IŞIKGÖR (210911023), Berk GÜNBERK (220911759) |

---

## GPU (NVIDIA CUDA)

Your venv should use **PyTorch + CUDA** wheels (not `+cpu`). Check:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

If `cuda` is `False`, reinstall GPU builds:

```powershell
.\.venv\Scripts\pip uninstall -y torch torchvision
.\.venv\Scripts\pip install -r requirements-torch-gpu.txt
```

RTX 3080 Ti works with **CUDA 12.4** builds (`cu124`). Lower batch sizes in the training scripts if you hit out-of-memory.

---

## Presentation demo (browser UI + API)

| Piece | Location |
|-------|----------|
| **Backend** | [`api/main.py`](api/main.py) — FastAPI, loads CLIP + segmentation weights from `mammography/checkpoints/*.pth` |
| **Frontend** | [`web/index.html`](web/index.html) — upload PNG/JPEG, see class probabilities + mask overlay |
| **Extra deps** | [`requirements-app.txt`](requirements-app.txt) |

**Install (from repo root `model_0/`):**

```powershell
.\.venv\Scripts\pip install -r requirements-train.txt
.\.venv\Scripts\pip install -r requirements-app.txt
```

**Run the demo** (requires trained checkpoints in `mammography/checkpoints/`):

```powershell
$env:MODEL_ROOT = "$PWD\mammography\checkpoints"
.\.venv\Scripts\uvicorn api.main:app --host 127.0.0.1 --port 8080
```

Open **http://127.0.0.1:8080/** · API health: **http://127.0.0.1:8080/api/health**

Shortcut: **`.\scripts\run_demo.ps1`** (sets `MODEL_ROOT` and starts Uvicorn on port 8080).

---

## Training (CBIS-DDSM mammography)

1. Download or locate **CBIS-DDSM** (folder must contain `csv/` and `jpeg/`).
2. Set `CBIS_ROOT` to that folder, then build manifests and train:

```powershell
$env:CBIS_ROOT = "C:\path\to\cbis-ddsm"
.\scripts\train_cbis_mammogram.ps1
```

Or step-by-step:

```powershell
.\.venv\Scripts\python mammography\build_manifest.py --data-root $env:CBIS_ROOT --out-dir mammography\cache
.\.venv\Scripts\python mammography\train_unetplusplus.py --manifest mammography\cache\manifest_segmentation.csv --checkpoint-dir mammography\checkpoints
.\.venv\Scripts\python mammography\train_clip_mammogram.py --manifest mammography\cache\manifest_classification.csv --checkpoint-dir mammography\checkpoints
```

**Full training shortcut** (same as `train_cbis_mammogram.ps1`, requires `CBIS_ROOT`):

```powershell
$env:CBIS_ROOT = "C:\path\to\cbis-ddsm"
.\scripts\run_full_training.ps1
```

If manifests already exist, you can run GPU training only: **`.\scripts\run_mammo_gpu_train.ps1`**

---

## Repository layout (essentials)

| Path | Purpose |
|------|---------|
| [`mammography/`](mammography/) | CBIS I/O, manifests, `train_clip_mammogram.py`, `train_unetplusplus.py` |
| [`mammography/cache/`](mammography/cache/) | Generated `manifest_*.csv` (after `build_manifest.py`) |
| `mammography/checkpoints/` | `best_clip_classifier.pth`, `best_unetplusplus_model.pth` (created by training) |
| [`api/`](api/) | FastAPI inference |
| [`web/`](web/) | Static UI |
| [`scripts/`](scripts/) | Demo and training helpers |

### Course PDF (optional)

If you keep the academic report as a file, place it under e.g. `docs/full_project_report.pdf` and link it from your slides; the narrative in the PDF may discuss MRI baselines while this repo implements **mammography** metrics and the demo.

---

## License

Capstone materials belong to the authors and Istinye University unless otherwise stated. Third-party code follows its original licenses.
