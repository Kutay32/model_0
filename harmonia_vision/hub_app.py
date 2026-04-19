"""
Harmonia Hub — local dashboard to monitor Harmonia training, start/stop jobs, list checkpoints,
and run quick mask previews.

Run from repository root:
  pip install fastapi uvicorn python-multipart pillow
  uvicorn harmonia_vision.hub_app:app --host 127.0.0.1 --port 8765

Optional auth for mutating routes:
  set HARMONIA_HUB_TOKEN=your-secret
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "harmonia"
CHECKPOINTS_DIR = ROOT / "harmonia_checkpoints"
# Full CBIS mass pipeline output (IID split). Smoke tests use harmonia_processed_smoke/.
DEFAULT_PROCESSED_DATA = ROOT / "harmonia_processed_cbis_full"

from harmonia_vision.hub_runner import hub
from harmonia_vision.hub_inference import eval_random_npy_samples, infer_mask_png


def _optional_token(request: Request) -> None:
    token = os.environ.get("HARMONIA_HUB_TOKEN", "").strip()
    if not token:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _default_data_root() -> str:
    return os.environ.get("HARMONIA_HUB_DATA_ROOT", str(DEFAULT_PROCESSED_DATA))


class TrainStartBody(BaseModel):
    mode: str = Field("centralized", pattern="^(centralized|fl_iid|fl_non_iid)$")
    preset: str = Field("standard", pattern="^(quick|standard|long)$")
    data_root: str = Field(default_factory=_default_data_root)
    batch_size: int = Field(4, ge=1, le=64)
    rounds: int | None = None
    local_epochs: int | None = None
    checkpoint: str | None = None
    metrics_json: str | None = None
    val_patient_fraction: float = Field(0.15, ge=0.0, le=0.5)
    split_seed: int = Field(42, ge=0)
    no_val_split: bool = False
    device: str = Field("auto", pattern="^(auto|cuda|cpu)$")
    no_progress: bool = False


class InferRandomNpyBody(BaseModel):
    checkpoint: str
    data_root: str | None = Field(default_factory=_default_data_root)
    n: int = Field(15, ge=1, le=256)
    seed: int | None = None


app = FastAPI(title="Harmonia Hub", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "service": "harmonia-hub", "infer_random_npy": True}


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    return {
        "project_root": str(ROOT),
        "checkpoints_dir": str(CHECKPOINTS_DIR),
        "default_data_root": _default_data_root(),
        "full_cbis_data_root": str(DEFAULT_PROCESSED_DATA),
        "smoke_data_root": str(ROOT / "harmonia_processed_smoke"),
        "auth_required": bool(os.environ.get("HARMONIA_HUB_TOKEN", "").strip()),
        # Set by current hub; if missing, the running server is an older process — restart uvicorn.
        "infer_random_npy": True,
    }


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return hub.status()


@app.get("/api/logs")
def api_logs(tail: int = 400) -> dict[str, Any]:
    return {"lines": hub.logs_tail(tail)}


@app.post("/api/train/start")
def api_train_start(body: TrainStartBody, _: None = Depends(_optional_token)) -> dict[str, Any]:
    if hub.is_running():
        raise HTTPException(409, detail="Training already running.")
    cmd = [
        sys.executable,
        "-m",
        "harmonia_vision.benchmark",
        "--mode",
        body.mode,
        "--data-root",
        body.data_root,
        "--preset",
        body.preset,
        "--batch-size",
        str(body.batch_size),
    ]
    if body.rounds is not None:
        cmd.extend(["--rounds", str(body.rounds)])
    if body.local_epochs is not None:
        cmd.extend(["--local-epochs", str(body.local_epochs)])
    if body.checkpoint:
        cmd.extend(["--checkpoint", body.checkpoint])
    if body.metrics_json:
        cmd.extend(["--metrics-json", body.metrics_json])
    if body.no_val_split:
        cmd.append("--no-val-split")
    else:
        cmd.extend(["--val-patient-fraction", str(body.val_patient_fraction)])
    cmd.extend(["--split-seed", str(body.split_seed)])
    cmd.extend(["--device", body.device])
    if body.no_progress:
        cmd.append("--no-progress")
    try:
        return hub.start(cmd)
    except RuntimeError as e:
        raise HTTPException(409, detail=str(e)) from e


@app.post("/api/train/stop")
def api_train_stop(_: None = Depends(_optional_token)) -> dict[str, Any]:
    return hub.stop()


@app.get("/api/artifacts")
def api_artifacts() -> dict[str, Any]:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for p in sorted(CHECKPOINTS_DIR.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True):
        st = p.stat()
        metrics_path = p.with_suffix(".metrics.json")
        m: dict[str, Any] | None = None
        if metrics_path.is_file():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                m = {"error": "could not read metrics"}
        items.append(
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": st.st_size,
                "mtime_utc": st.st_mtime,
                "metrics": m,
            }
        )
    return {"checkpoints_dir": str(CHECKPOINTS_DIR), "items": items}


def _resolve_checkpoint(checkpoint: str) -> Path:
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        alt = CHECKPOINTS_DIR / Path(checkpoint).name
        if alt.is_file():
            ckpt_path = alt
    if not ckpt_path.is_file():
        raise HTTPException(404, detail=f"Checkpoint not found: {checkpoint}")
    return ckpt_path


@app.post("/api/infer/random-npy")
@app.post("/api/infer/random_npy")  # alias (same handler)
def api_infer_random_npy(body: InferRandomNpyBody) -> dict[str, Any]:
    """Evaluate the checkpoint on n random paired .npy samples (processed client_a/b layout)."""
    ckpt_path = _resolve_checkpoint(body.checkpoint)
    data_root = (body.data_root or _default_data_root()).strip()
    if not data_root:
        raise HTTPException(400, detail="data_root is empty.")
    try:
        out = eval_random_npy_samples(
            data_root,
            ckpt_path,
            n=body.n,
            seed=body.seed,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, detail=str(e)) from e
    return {"ok": True, **out}


@app.post("/api/infer/preview")
async def api_infer_preview(
    file: UploadFile = File(...),
    checkpoint: str = Form(...),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Expected an image (PNG/JPEG).")
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, detail="Image too large (max 20 MB).")
    ckpt_path = _resolve_checkpoint(checkpoint)
    try:
        out = infer_mask_png(data, ckpt_path)
        return {"ok": True, **out}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, detail=str(e)) from e


@app.get("/")
def index():
    index_path = WEB / "index.html"
    if not index_path.is_file():
        return JSONResponse(
            {"message": "Harmonia UI missing. Create web/harmonia/index.html"},
            status_code=404,
        )
    return FileResponse(index_path)


if WEB.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")
