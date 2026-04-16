"""
Load CLIP + U-Net checkpoints from `mammography/checkpoints/` (or MODEL_ROOT) and run inference for the web demo.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

import albumentations as A
import clip
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Mammogram (CBIS) — must match `mammography/train_clip_mammogram.py`
TEXT_PROMPTS_MAMMO = [
    "A screening mammogram with no suspicious lesion, normal breast tissue",
    "A mammogram showing a benign breast lesion or mass",
    "A mammogram showing a malignant breast lesion or mass",
]
CLASS_NAMES = ["normal", "benign", "malignant"]


@dataclass
class ModelPaths:
    model_root: str
    clip_weights: str
    unet_weights: str
    unetplusplus_weights: str


def default_paths() -> ModelPaths:
    root = os.environ.get("MODEL_ROOT")
    if not root:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(os.path.dirname(here), "mammography", "checkpoints")
    root = os.path.abspath(root)
    return ModelPaths(
        model_root=root,
        clip_weights=os.path.join(root, "best_clip_classifier.pth"),
        unet_weights=os.path.join(root, "best_unet_model.pth"),
        unetplusplus_weights=os.path.join(root, "best_unetplusplus_model.pth"),
    )


class BreastLesionPipeline:
    def __init__(self, paths: ModelPaths | None = None):
        self.paths = paths or default_paths()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.clip_preprocess = None
        self.unet = None
        self._seg_arch = "unet"
        self._text_prompts: list[str] = TEXT_PROMPTS_MAMMO
        self._seg_transform = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def load(self) -> None:
        p = self.paths
        if not os.path.isfile(p.clip_weights):
            raise FileNotFoundError(f"Missing CLIP weights: {p.clip_weights}")

        clip_model, preprocess = clip.load("ViT-B/32", device=self.device)
        clip_model.load_state_dict(
            torch.load(p.clip_weights, map_location=self.device, weights_only=True)
        )
        clip_model.eval()
        self.clip_model = clip_model
        self.clip_preprocess = preprocess

        if os.path.isfile(p.unetplusplus_weights):
            unet = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None,
            ).to(self.device)
            unet.load_state_dict(
                torch.load(p.unetplusplus_weights, map_location=self.device, weights_only=True)
            )
            self._seg_arch = "unetplusplus"
            seg_path = p.unetplusplus_weights
        elif os.path.isfile(p.unet_weights):
            unet = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None,
            ).to(self.device)
            unet.load_state_dict(
                torch.load(p.unet_weights, map_location=self.device, weights_only=True)
            )
            self._seg_arch = "unet"
            seg_path = p.unet_weights
        else:
            raise FileNotFoundError(
                f"Missing segmentation weights: expected {p.unetplusplus_weights} or {p.unet_weights}"
            )
        unet.eval()
        self.unet = unet
        self._seg_weights_path = seg_path

        text_inputs = clip.tokenize(self._text_prompts).to(self.device)
        with torch.no_grad():
            tf = clip_model.encode_text(text_inputs)
            self.text_features = tf / tf.norm(dim=-1, keepdim=True)

    def _ensure_loaded(self) -> None:
        if self.clip_model is None:
            self.load()

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        self._ensure_loaded()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        # --- CLIP ---
        img_clip = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        img_feat = self.clip_model.encode_image(img_clip)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * (img_feat @ self.text_features.T)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        out: dict[str, Any] = {
            "classification": {
                "label": CLASS_NAMES[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(3)},
            },
            "image_size": {"width": w, "height": h},
            "segmentation_model": self._seg_arch,
        }

        # --- U-Net mask (upsample to original size) ---
        arr = np.array(image)
        aug = self._seg_transform(image=arr, mask=np.zeros((h, w), dtype=np.uint8))
        t = aug["image"].unsqueeze(0).to(self.device)
        logits_m = self.unet(t)
        prob = torch.sigmoid(logits_m)[0, 0]
        prob_up = F.interpolate(
            prob.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        mask_bin = (prob_up > 0.5).cpu().numpy().astype(np.uint8)

        overlay = self._blend_overlay(arr, mask_bin > 0)
        out["segmentation"] = {
            "dice_proxy": float(mask_bin.mean()),
            "mask_coverage": float(mask_bin.mean()),
        }
        out["images"] = {
            "overlay_png_base64": _png_b64(overlay),
            "mask_png_base64": _png_b64(
                np.stack([mask_bin * 255, mask_bin * 255, mask_bin * 255], axis=-1)
            ),
        }
        return out

    def _blend_overlay(
        self, rgb: np.ndarray, mask: np.ndarray, color=(255, 80, 120), alpha: float = 0.38
    ) -> np.ndarray:
        base = rgb.astype(np.float32)
        c = np.array(color, dtype=np.float32)
        m = mask.astype(np.float32)[..., None]
        blended = base * (1.0 - alpha * m) + c * (alpha * m)
        return np.clip(blended, 0, 255).astype(np.uint8)


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_pipeline: BreastLesionPipeline | None = None


def get_pipeline() -> BreastLesionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BreastLesionPipeline()
        _pipeline.load()
    return _pipeline
