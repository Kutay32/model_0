"""Load Harmonia U-Net checkpoint and run a quick 256×256 grayscale inference on an uploaded image."""

from __future__ import annotations

import base64
import csv
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .model import UNet, dice_iou_from_logits

# Mammogram CLIP — same prompts as `mammography/train_clip_mammogram.py` / `api/inference.py`
_CLIP_TEXT_PROMPTS = [
    "A screening mammogram with no suspicious lesion, normal breast tissue",
    "A mammogram showing a benign breast lesion or mass",
    "A mammogram showing a malignant breast lesion or mass",
]
_CLIP_CLASS_KEYS = ("normal", "benign", "malignant")

_clip_bundle: dict[str, Any] | None = None  # lazy: model, preprocess, text_features, device_id


def load_model_from_checkpoint(ckpt_path: str | Path, device: torch.device) -> UNet:
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = UNet(in_channels=1, base_ch=32).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _resolve_clip_weights_path() -> Path | None:
    for env in ("MODEL_ROOT", "HARMONIA_CLIP_ROOT"):
        r = os.environ.get(env, "").strip()
        if r:
            p = Path(r) / "best_clip_classifier.pth"
            if p.is_file() and _clip_state_dict_is_finite(p):
                return p
    root = Path(__file__).resolve().parent.parent
    candidate = root / "mammography" / "checkpoints" / "best_clip_classifier.pth"
    if candidate.is_file() and _clip_state_dict_is_finite(candidate):
        return candidate
    return None


def _clip_state_dict_is_finite(ckpt_path: Path) -> bool:
    """Skip CLIP if the saved weights contain NaN/Inf (corrupt or incompatible checkpoint)."""
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    if not isinstance(sd, dict):
        return False
    for v in sd.values():
        if not hasattr(v, "float"):
            continue
        t = v.float()
        if torch.isnan(t).any() or torch.isinf(t).any():
            return False
    return True


def _load_stem_pathology_map(data_root: Path) -> dict[str, str]:
    """Map stem (e.g. P_00001_LEFT_CC) -> BENIGN/MALIGNANT from manifest CSV if present."""
    out: dict[str, str] = {}
    for name in ("manifest_iid.csv", "manifest_non_iid.csv"):
        p = data_root / name
        if not p.is_file():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ip = (row.get("image") or "").strip()
                if not ip:
                    continue
                stem = Path(ip).stem
                patho = (row.get("pathology") or "").strip().upper()
                if patho in ("BENIGN", "MALIGNANT"):
                    out[stem] = patho
        if out:
            break
    return out


def _get_clip_bundle(device: torch.device) -> tuple[Any, Any, torch.Tensor] | None:
    """Lazy-load CLIP once per process (ViT-B/32 + mammogram text prompts)."""
    global _clip_bundle
    weights = _resolve_clip_weights_path()
    if weights is None:
        return None
    try:
        import clip  # type: ignore[import-untyped]
    except ImportError:
        return None
    dev_id = str(device)
    if _clip_bundle is not None and _clip_bundle.get("device") == dev_id and _clip_bundle.get("weights") == str(weights):
        return _clip_bundle["model"], _clip_bundle["preprocess"], _clip_bundle["text_features"]

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    w = torch.load(weights, map_location=device, weights_only=True)
    clip_model.load_state_dict(w)
    clip_model.eval()
    text_tokens = clip.tokenize(_CLIP_TEXT_PROMPTS).to(device)
    with torch.no_grad():
        tf = clip_model.encode_text(text_tokens)
        text_features = tf / tf.norm(dim=-1, keepdim=True)
    _clip_bundle = {
        "device": dev_id,
        "weights": str(weights),
        "model": clip_model,
        "preprocess": preprocess,
        "text_features": text_features,
    }
    return clip_model, preprocess, text_features


@torch.inference_mode()
def _clip_class_probs_gray(x01: np.ndarray, device: torch.device) -> dict[str, float] | None:
    """Softmax probs over normal/benign/malignant for a single-channel 0–1 float array (H×W)."""
    bundle = _get_clip_bundle(device)
    if bundle is None:
        return None
    clip_model, preprocess, text_features = bundle
    u8 = np.clip(x01 * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(u8, mode="L").convert("RGB")
    img_t = preprocess(pil).unsqueeze(0).to(device)
    img_feat = clip_model.encode_image(img_t)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale * (img_feat @ text_features.T)
    probs = torch.softmax(logits[0], dim=0).detach().cpu().numpy()
    out = {k: float(probs[i]) for i, k in enumerate(_CLIP_CLASS_KEYS)}
    if any(not np.isfinite(v) for v in out.values()):
        return None
    return out


@torch.no_grad()
def infer_mask_png(
    image_bytes: bytes,
    ckpt_path: str | Path,
    device: torch.device | None = None,
) -> dict[str, str | float]:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(ckpt_path, dev)

    im = Image.open(io.BytesIO(image_bytes)).convert("L")
    im = im.resize((256, 256), Image.Resampling.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(dev)
    logits = model(t)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask_u8 = (prob > 0.5).astype(np.uint8) * 255
    overlay = np.stack([x * 255, x * 255, np.clip(x * 255 + mask_u8 * 0.35, 0, 255)], axis=-1).astype(np.uint8)
    pil_mask = Image.fromarray(mask_u8, mode="L")
    pil_overlay = Image.fromarray(overlay, mode="RGB")

    def b64_png(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "mask_png_b64": b64_png(pil_mask),
        "overlay_png_b64": b64_png(pil_overlay),
        "mean_prob": float(prob.mean()),
    }


def _prep_x_npy(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    if x.ndim == 3:
        x = np.mean(x, axis=-1)
    return np.clip(x, 0.0, 1.0)


def _prep_y_npy(arr: np.ndarray) -> np.ndarray:
    y = arr.astype(np.float32)
    if y.ndim == 3:
        y = np.mean(y, axis=-1)
    return np.clip(y, 0.0, 1.0)


def collect_paired_npy_paths(data_root: str | Path) -> list[tuple[Path, Path]]:
    """All (image.npy, mask.npy) pairs under client_a and client_b."""
    root = Path(data_root).resolve()
    pairs: list[tuple[Path, Path]] = []
    for client in ("client_a", "client_b"):
        img_dir = root / client / "images"
        mask_dir = root / client / "masks"
        if not img_dir.is_dir():
            continue
        for img_p in sorted(img_dir.glob("*.npy")):
            mp = mask_dir / img_p.name
            if mp.is_file():
                pairs.append((img_p, mp))
    return pairs


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gray_input_b64(x: np.ndarray) -> str:
    """x: H×W float 0–1 → grayscale PNG."""
    u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return _pil_to_b64_png(Image.fromarray(u8, mode="L"))


def _binary_mask_b64(mask01: np.ndarray) -> str:
    """mask01: H×W float 0–1 → binary L PNG."""
    u8 = (mask01 >= 0.5).astype(np.uint8) * 255
    return _pil_to_b64_png(Image.fromarray(u8, mode="L"))


def _gt_overlay_b64(x: np.ndarray, y: np.ndarray) -> str:
    """Ground-truth overlay: green tint on mask (x, y float 0–1)."""
    m = (y >= 0.5).astype(np.uint8) * 255
    overlay = np.stack(
        [
            np.clip(x * 255.0 - m * 0.2, 0, 255),
            np.clip(x * 255.0 + m * 0.4, 0, 255),
            np.clip(x * 255.0 - m * 0.1, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    return _pil_to_b64_png(Image.fromarray(overlay, mode="RGB"))


def _mask_overlay_b64_from_gray_and_prob(x: np.ndarray, prob: np.ndarray) -> tuple[str, str]:
    """x, prob: H×W float 0–1."""
    mask_u8 = (prob > 0.5).astype(np.uint8) * 255
    overlay = np.stack(
        [x * 255, x * 255, np.clip(x * 255 + mask_u8 * 0.35, 0, 255)],
        axis=-1,
    ).astype(np.uint8)
    pil_mask = Image.fromarray(mask_u8, mode="L")
    pil_overlay = Image.fromarray(overlay, mode="RGB")
    return _pil_to_b64_png(pil_mask), _pil_to_b64_png(pil_overlay)


@torch.no_grad()
def eval_random_npy_samples(
    data_root: str | Path,
    ckpt_path: str | Path,
    n: int = 15,
    seed: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Pick n random paired `.npy` samples from processed data_root, run the checkpoint once, return per-image metrics + PNGs.
    """
    pairs = collect_paired_npy_paths(data_root)
    if not pairs:
        raise FileNotFoundError(f"No paired .npy under {data_root} (client_a|b/images + masks).")

    data_root_p = Path(data_root).resolve()
    stem_pathology = _load_stem_pathology_map(data_root_p)

    rng = np.random.default_rng(seed)
    n_requested = n
    n_take = min(max(n_requested, 1), len(pairs))
    idx = rng.choice(len(pairs), size=n_take, replace=False)
    chosen = [pairs[i] for i in idx]

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(ckpt_path, dev)
    clip_available = _get_clip_bundle(dev) is not None

    items: list[dict[str, Any]] = []
    sum_dice = 0.0
    sum_iou = 0.0
    for img_p, mask_p in chosen:
        x = _prep_x_npy(np.load(img_p))
        y = _prep_y_npy(np.load(mask_p))
        if x.shape != y.shape:
            y = np.asarray(Image.fromarray((y * 255).astype(np.uint8), mode="L").resize((x.shape[1], x.shape[0]), Image.Resampling.NEAREST)) / 255.0
            y = np.clip(y.astype(np.float32), 0.0, 1.0)

        xi = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(dev)
        yi = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(dev)
        logits = model(xi)
        dice, iou = dice_iou_from_logits(logits, yi)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred_mask_b64, pred_over_b64 = _mask_overlay_b64_from_gray_and_prob(x, prob)
        sum_dice += dice
        sum_iou += iou
        stem = img_p.stem
        patho_gt = stem_pathology.get(stem)
        clip_probs = _clip_class_probs_gray(x, dev) if clip_available else None
        row: dict[str, Any] = {
            "stem": stem,
            "client": "client_a" if "client_a" in str(img_p).replace("\\", "/") else "client_b",
            "pathology_gt": patho_gt,
            "dsc": dice,
            "iou": iou,
            "mean_prob": float(prob.mean()),
            "input_png_b64": _gray_input_b64(x),
            "mask_png_b64": pred_mask_b64,
            "overlay_png_b64": pred_over_b64,
            "gt_mask_png_b64": _binary_mask_b64(y),
            "gt_overlay_png_b64": _gt_overlay_b64(x, y),
        }
        if patho_gt == "BENIGN":
            row["gt_prob_benign"] = 1.0
            row["gt_prob_malignant"] = 0.0
        elif patho_gt == "MALIGNANT":
            row["gt_prob_benign"] = 0.0
            row["gt_prob_malignant"] = 1.0
        if clip_probs is not None:
            row["clip_probs"] = clip_probs
            row["prob_benign"] = clip_probs["benign"]
            row["prob_malignant"] = clip_probs["malignant"]
            row["prob_normal"] = clip_probs["normal"]
        items.append(row)

    k = len(items)
    out: dict[str, Any] = {
        "n_requested": n_requested,
        "n_evaluated": k,
        "data_root": str(Path(data_root).resolve()),
        "checkpoint": str(Path(ckpt_path).resolve()),
        "seed": seed,
        "mean_dsc": float(sum_dice / max(k, 1)),
        "mean_iou": float(sum_iou / max(k, 1)),
        "pathology_manifest": bool(stem_pathology),
        "clip_class_head": clip_available,
        "items": items,
    }
    if clip_available and items:
        pb = [float(it["prob_benign"]) for it in items if "prob_benign" in it]
        pm = [float(it["prob_malignant"]) for it in items if "prob_malignant" in it]
        if pb:
            out["mean_prob_benign_clip"] = float(sum(pb) / len(pb))
        if pm:
            out["mean_prob_malignant_clip"] = float(sum(pm) / len(pm))
    return out
