# Harmonia Vision

Privacy-oriented federated learning stack for **breast mass segmentation** on CBIS-DDSM-style exports: a **U-Net** (PyTorch + Albumentations) trains with **Flower (FedAvg)**, optional **TenSEAL CKKS** encrypted uploads, and **Docker Compose** wiring for one server and two hospital clients on an internal bridge network (a practical stand-in for a WireGuard-only mesh on a VPS).

> **Note on PHE:** [TenSEAL](https://github.com/OpenMined/TenSEAL) implements leveled schemes like **CKKS/BFV**. The classic **Paillier** cryptosystem is *not* provided by TenSEAL; this project uses **CKKS** ciphertext addition to approximate a PHE aggregation path. Plaintext FedAvg runs when PHE is disabled or TenSEAL is missing.

## Layout

| Path | Role |
|------|------|
| `data_pipeline.py` | CBIS-DDSD mass-only ingest, DICOM windowing, breast crop, OR-merge masks, 256² `.npy`, IID vs non-IID splits |
| `model.py` | `UNet` + hybrid **BCE + Dice** loss, Adam `1e-4` |
| `server.py` | Flower gRPC server, `FedAvg`, optional CKKS aggregation + communication accounting |
| `client.py` | `NumPyClient`, 5 local epochs per round, optional encrypted uploads |
| `benchmark.py` | Centralized baseline vs two-client FedAvg simulation (metrics + rough comm MB) |
| `docker-compose.yml` | `harmonia-server`, `harmonia-client-a`, `harmonia-client-b` on `harmonia_net` |
| `Dockerfile` | Python 3.11 + `requirements.txt` |
| `tls_util.py` | PEM loading for gRPC TLS |
| `scripts/generate_dev_tls.py` | Dev-only CA + server cert for local TLS smoke tests |
| `hub_app.py` + `web/harmonia/` | **Harmonia Hub** — browser UI to start/stop training, logs, checkpoints, quick mask preview |

## Harmonia Hub (training dashboard)

Local web app on **this machine** to start/stop `harmonia_vision.benchmark` jobs, tail logs, list `harmonia_checkpoints/*.pth`, and run a **quick mask preview** (non-clinical).

```bash
pip install pillow fastapi uvicorn python-multipart
export PYTHONPATH=/path/to/model_0   # Windows PowerShell: $env:PYTHONPATH="C:\path\to\model_0"
uvicorn harmonia_vision.hub_app:app --host 127.0.0.1 --port 8765
```

Open **http://127.0.0.1:8765** in your browser.

- **Default data root** for the form: `HARMONIA_HUB_DATA_ROOT` env, else **`harmonia_processed_cbis_full/`** (full CBIS mass export, IID). The smaller **`harmonia_processed_smoke/`** folder is only for quick tests (`--max-groups`).
- **Optional auth** for Start/Stop: set `HARMONIA_HUB_TOKEN` on the server and paste the same value in **Hub token** in the UI (sent as `Authorization: Bearer …`).

Bind to `127.0.0.1` only unless you trust your network. This is not hardened for the public internet.

**Windows:** `.\scripts\run_harmonia_hub.ps1` from the repo root (uses `.venv`).

## Environment

- Python 3.11+ recommended
- Install deps from this folder:

```bash
pip install -r requirements.txt
```

- Set `PYTHONPATH` to the repository root so `harmonia_vision` imports resolve:

**Windows (PowerShell)**

```powershell
$env:PYTHONPATH="C:\path\to\model_0"
```

**Linux / macOS**

```bash
export PYTHONPATH=/path/to/model_0
```

## Data preprocessing

Point `--dataset-root` at an extracted CBIS-DDSM tree that includes `csv/dicom_info.csv` and `mass_*.csv` manifests. The pipeline **ignores** calcification CSVs and only ingests `mass_*.csv`.

**IID split (50/50 patients):**

```bash
python -m harmonia_vision.data_pipeline --dataset-root /data/CBIS-DDSM --out-root /out/harmonia_iid --split-mode iid
```

**Non-IID split (benign-heavy vs malignant-heavy clients):**

```bash
python -m harmonia_vision.data_pipeline --dataset-root /data/CBIS-DDSM --out-root /out/harmonia_non_iid --split-mode non_iid
```

Outputs per client:

```
/out/.../client_a/images/*.npy
/out/.../client_a/masks/*.npy
/out/.../client_b/images/*.npy
/out/.../client_b/masks/*.npy
manifest_{iid|non_iid}.csv
```

### Smoke test / dry-run (CSV + path validation only)

If your repo only contains `csv/` (no full `jpeg/` tree yet), you can still validate manifests and counts without writing images:

```bash
python -m harmonia_vision.data_pipeline --dataset-root /path/to/parent/of/csv --split-mode iid --dry-run
```

Use `--max-groups N` to process only the first *N* lesion groups when running a full export.

## Federated training (local processes)

**Terminal 1 — server**

```bash
python -m harmonia_vision.server --host 0.0.0.0 --port 8080 --rounds 5 --min-clients 2
```

**Terminal 2 — client A**

```bash
python -m harmonia_vision.client --server 127.0.0.1:8080 --data-root /out/harmonia_iid --client-key client_a
```

**Terminal 3 — client B**

```bash
python -m harmonia_vision.client --server 127.0.0.1:8080 --data-root /out/harmonia_iid --client-key client_b
```

### Optional CKKS encrypted uploads

Enable on the **server** (needs TenSEAL):

```bash
HARMONIA_PHE=1 python -m harmonia_vision.server --phe --rounds 5 --min-clients 2
```

Enable matching **clients**:

```bash
HARMONIA_PHE=1 python -m harmonia_vision.client --server 127.0.0.1:8080 --data-root /out/harmonia_iid --client-key client_a --phe
```

The server decrypts with its secret context and averages using the same sample-weighting rule as FedAvg (`sum_k w_k n_k / sum_k n_k`).

## Docker Compose

From this directory:

```bash
# Preprocessed data on host: ./processed_data/client_{a,b}/{images,masks}
export HARMONIA_PROCESSED_HOST_PATH=/absolute/path/to/processed
docker compose up --build
```

The compose file binds an internal **bridge** `harmonia_net` analogous to a private overlay; for a DigitalOcean VPS, run **WireGuard** on the host so only VPN peers reach the published gRPC port, or keep the port unpublished and tunnel via WG.

## gRPC TLS (TLS 1.2+ / 1.3 via OpenSSL)

**Server** — pass all three PEM paths (or set env vars). Order matches Flower: **CA**, **server certificate**, **server private key**.

```bash
python -m harmonia_vision.server --host 0.0.0.0 --port 8080 \
  --tls-ca ./dev_tls/ca.pem --tls-cert ./dev_tls/server.pem --tls-key ./dev_tls/server.key
```

Environment equivalents: `HARMONIA_TLS_CA_PATH`, `HARMONIA_TLS_SERVER_CERT`, `HARMONIA_TLS_SERVER_KEY`. Omit all three for **plaintext** gRPC (default).

**Client** — trust the same CA file used on the server:

```bash
python -m harmonia_vision.client --server 127.0.0.1:8080 --data-root /out/harmonia_iid --client-key client_a \
  --tls-ca ./dev_tls/ca.pem
```

Or set `HARMONIA_TLS_ROOT_CA` to that path. When `--tls-ca` / env is unset, the client uses **insecure** gRPC (matches a plaintext server).

**Dev certificates** (local only; not for production):

```bash
python -m harmonia_vision.scripts.generate_dev_tls --out-dir ./dev_tls
```

## Benchmarks (single machine)

**Presets** (default `--preset standard`): `quick` (2×2 loops), `standard` (5×5), `long` (20×10). Override with `--rounds` and/or `--local-epochs`.

**Eval split** — By default the benchmark holds out a **patient-level** validation set (`--val-patient-fraction`, default `0.15`; seed `--split-seed`). Reported DSC/IoU/`eval_loss` are on that hold-out set. Use `--no-val-split` only for debugging (metrics then overlap training data, like the early baseline).

**Wait for preprocess, then train** — The pipeline writes `manifest_{iid|non_iid}.csv` at the **end**. To avoid training on a partial export, poll for that file and then launch the benchmark:

```bash
python -m harmonia_vision.wait_for_preprocess --data-root /out/harmonia_iid -- --mode centralized \
  --data-root /out/harmonia_iid --preset long --checkpoint harmonia_checkpoints/cbis_full_centralized_long.pth
```

Rapor / tez notu (eski kısıt): *Değerlendirme eğitimle aynı loader üzerinden (ayrı test seti yok); genelleme için ileride validation/test veya hasta bazlı hold-out eklenmesi iyi olur.* — Bu sürümde hasta bazlı hold-out varsayılan olarak açıktır; ayrı bir dış test seti veya çoklu seed ile raporlama ileride eklenebilir.

```bash
python -m harmonia_vision.benchmark --mode centralized --data-root /out/harmonia_iid --preset standard
python -m harmonia_vision.benchmark --mode centralized --data-root /out/harmonia_iid --preset long
python -m harmonia_vision.benchmark --mode fl_iid --data-root /out/harmonia_iid --preset quick
```

**Checkpoints** — saves `state_dict` + run metadata (`.pth`), and a JSON sidecar (default: `PATH.metrics.json` next to the checkpoint):

```bash
python -m harmonia_vision.benchmark --mode centralized --data-root /out/harmonia_iid --preset long \
  --checkpoint harmonia_checkpoints/centralized_last.pth
```

**Comparing smoke vs full CBIS** — use distinct checkpoint names so Hub and metrics stay clear, e.g. `harmonia_checkpoints/smoke_centralized.pth` (data: `harmonia_processed_smoke`) vs `harmonia_checkpoints/cbis_full_centralized.pth` (data: `harmonia_processed_cbis_full`).

Load later in Python:

```python
import torch
from harmonia_vision.model import UNet
ckpt = torch.load("harmonia_checkpoints/centralized_last.pth", map_location="cpu")
model = UNet(in_channels=1, base_ch=32)
model.load_state_dict(ckpt["state_dict"])
```

`fl_non_iid` uses the same simulator as `fl_iid`; the **dataset** difference comes from preprocessing (`non_iid` manifest).

Metrics printed: **DSC**, **IoU**, **eval_loss** (hold-out or full overlap), last **train_loss** (centralized only), estimated **communication MB** (param tensor size heuristic), wall **training time**, and paths when `--checkpoint` is set.

## License

Use and extend under your project’s license.
