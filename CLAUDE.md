# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Privacy-preserving deep learning for breast mammography with two distinct stacks:

1. **mammography/** — Centralized CLIP + UNet++ training and FastAPI inference on CBIS-DDSM
2. **harmonia_vision/** — Federated learning (Flower) with optional CKKS encryption for breast mass segmentation

## Environment Setup

```powershell
# GPU check
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Install GPU PyTorch (CUDA 12.4 for RTX 3080 Ti)
.\.venv\Scripts\pip install -r requirements-torch-gpu.txt

# Install training deps
.\.venv\Scripts\pip install -r requirements-train.txt

# Install app deps (for demo)
.\.venv\Scripts\pip install -r requirements-app.txt
```

**PYTHONPATH** must be set to repo root when running Harmonia Vision modules:
```powershell
$env:PYTHONPATH="C:\path\to\model_0"
```

## Common Commands

### Mammography Pipeline (Centralized)

```powershell
# Full training (manifest build + UNet++ + CLIP)
$env:CBIS_ROOT = "C:\path\to\cbis-ddsm"
.\scripts\run_full_training.ps1

# GPU training only (skip manifest build)
.\scripts\run_mammo_gpu_train.ps1

# Build manifests only
.\.venv\Scripts\python mammography\build_manifest.py --data-root $env:CBIS_ROOT --out-dir mammography\cache

# Demo web UI (requires trained checkpoints)
$env:MODEL_ROOT = "$PWD\mammography\checkpoints"
.\.venv\Scripts\uvicorn api.main:app --host 127.0.0.1 --port 8080
```

### Harmonia Vision (Federated)

```powershell
# Preprocess CBIS for FL (IID split)
python -m harmonia_vision.data_pipeline --dataset-root /data/CBIS-DDSM --out-root harmonia_processed --split-mode iid

# Preprocess with non-IID split (benign-heavy vs malignant-heavy clients)
python -m harmonia_vision.data_pipeline --dataset-root /data/CBIS-DDSM --out-root harmonia_processed --split-mode non_iid

# Centralized benchmark
python -m harmonia_vision.benchmark --mode centralized --data-root harmonia_processed --preset standard --checkpoint harmonia_checkpoints/centralized.pth

# Federated simulation (same data, mode flag differs)
python -m harmonia_vision.benchmark --mode fl_iid --data-root harmonia_processed --preset standard

# FL with real processes (server + 2 clients)
python -m harmonia_vision.server --host 0.0.0.0 --port 8080 --rounds 5 --min-clients 2
python -m harmonia_vision.client --server 127.0.0.1:8080 --data-root harmonia_processed --client-key client_a
python -m harmonia_vision.client --server 127.0.0.1:8080 --data-root harmonia_processed --client-key client_b

# With CKKS encryption
HARMONIA_PHE=1 python -m harmonia_vision.server --phe --rounds 5 --min-clients 2
HARMONIA_PHE=1 python -m harmonia_vision.client --server 127.0.0.1:8080 --data-root harmonia_processed --client-key client_a --phe

# Hub dashboard (browser UI for training control)
.\scripts\run_harmonia_hub.ps1
```

## Architecture

### Mammography Stack

```
mammography/
├── cbis_io.py          # CBIS-DDSM CSV parsing, path resolution, manifest building
├── datasets.py         # PyTorch datasets: MammogramSegDataset, MammogramMultiTaskDataset, MammogramClipDataset
├── build_manifest.py  # Converts raw CBIS → CSV manifests (deprecated, use scripts/preprocess_cbis.py)
├── train_clip_mammogram.py      # CLIP fine-tuning for 3-way classification (normal/benign/malignant)
├── train_unetplusplus.py        # UNet++ segmentation training
├── train_multitask.py          # Joint segmentation + classification
├── multitask_unetpp.py         # MultiTaskUNetPP model (UNet++ with classification head)
└── checkpoints/       # best_clip_classifier.pth, best_unetplusplus_model.pth

api/
├── main.py            # FastAPI app, loads .pth checkpoints
└── inference.py       # Inference logic for CLIP + segmentation

scripts/
├── preprocess_cbis.py  # Offline CBIS preprocessing (preferred over build_manifest.py)
└── run_*.ps1         # Training/demo shortcuts
```

### Harmonia Vision Stack

```
harmonia_vision/
├── data_pipeline.py    # CBIS → DICOM windowing → breast crop → OR-merge masks → 256² .npy per client
├── dataset_npy.py     # PyTorch dataset for .npy images/masks
├── model.py           # UNet + HybridSegmentationLoss (BCE + Dice) + dice_iou_from_logits
├── server.py          # Flower gRPC server with FedAvg + optional CKKS aggregation
├── client.py          # Flower NumPyClient, 5 local epochs/round
├── benchmark.py       # Centralized vs FL simulation on local .npy data; patient-level hold-out split
├── crypto_phe.py      # TenSEAL CKKS encryption helpers
├── hub_app.py         # Harmonia Hub web app (FastAPI, starts/stops benchmark jobs)
└── hub_inference.py   # Quick mask preview for Hub

scripts/
└── generate_dev_tls.py  # Dev-only TLS certificates
```

**Key architectural distinction**: `harmonia_vision/benchmark.py` simulates FL in a single process (same `data_pipeline.py` output, different `--mode`); real FL uses `server.py` + `client.py` with Flower gRPC.

## Data Flow

```
CBIS-DDSM raw (csv/, jpeg/)
    ↓ (harmonia_vision/data_pipeline.py OR scripts/preprocess_cbis.py)
Preprocessed .npy / CSV manifests
    ↓
Training → checkpoints/*.pth
    ↓
Inference: api/main.py → web/index.html
```

For Harmonia FL, preprocessing outputs `client_a/` and `client_b/` folders, then `benchmark.py` or Flower server/client consume them.

## Key Configuration

| Variable | Purpose |
|----------|---------|
| `CBIS_ROOT` | Raw CBIS-DDSM folder (training pipeline) |
| `MODEL_ROOT` | `mammography/checkpoints/` (demo API) |
| `PYTHONPATH` | Repo root (Harmonia Vision imports) |
| `HARMONIA_PHE` | Enable CKKS encryption (1=on) |
| `HARMONIA_HUB_DATA_ROOT` | Default data for Hub (else `harmonia_processed_cbis_full/`) |
| `HARMONIA_HUB_TOKEN` | Optional auth for Hub Start/Stop |
