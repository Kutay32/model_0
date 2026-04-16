"""
FastAPI backend for the breast lesion CLIP + U-Net demo.
Run from project root: uvicorn api.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.inference import get_pipeline

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

app = FastAPI(
    title="Breast Lesion AI Demo",
    description="CLIP classification + U-Net segmentation (capstone demo)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    os.environ.setdefault(
        "MODEL_ROOT", str(ROOT / "mammography" / "checkpoints")
    )
    try:
        get_pipeline()
    except FileNotFoundError as e:
        # Allow server to start; /api/health reports not ready
        app.state.load_error = str(e)
        return
    app.state.load_error = None


@app.get("/api/health")
def health():
    err = getattr(app.state, "load_error", None)
    if err:
        return JSONResponse(
            {"status": "degraded", "error": err, "models_loaded": False},
            status_code=503,
        )
    return {"status": "ok", "models_loaded": True}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    err = getattr(app.state, "load_error", None)
    if err:
        raise HTTPException(503, detail=f"Models not loaded: {err}")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Expected an image file (PNG/JPEG).")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(400, detail="Image too large (max 15 MB).")
    try:
        return get_pipeline().predict(data)
    except Exception as e:
        raise HTTPException(500, detail=str(e)) from e


# Static UI (must be after API routes)
if WEB.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(WEB / "static")),
        name="static",
    )


@app.get("/")
def index():
    index_path = WEB / "index.html"
    if not index_path.is_file():
        return JSONResponse(
            {"message": "UI not found. Create web/index.html"},
            status_code=404,
        )
    return FileResponse(index_path)
