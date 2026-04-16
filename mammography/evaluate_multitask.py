"""
Evaluate the multi-task UNet++ checkpoint.

Run from repo root::

    .\\.venv\\Scripts\\python mammography\\evaluate_multitask.py ^
        --manifest mammography\\cache\\manifest_segmentation.csv

Writes ``mammography/checkpoints/multitask_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mammography.datasets import MammogramMultiTaskDataset
from mammography.multitask_unetpp import MultiTaskUNetPP


# ── metrics ──────────────────────────────────────────────────────────
def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    p = pred.view(-1).float()
    t = target.view(-1).float()
    inter = (p * t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


def iou_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    p = pred.view(-1).float()
    t = target.view(-1).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + eps) / (union + eps))


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.view(-1).float()
    t = target.view(-1).float()
    return float((p == t).float().mean())


# ── main ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate multi-task UNet++.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", default="",
                    help="Path to .pth (default: mammography/checkpoints/best_multitask_unetpp.pth)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--split", default="val", help="Which split to evaluate on.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    ck_path = Path(args.checkpoint) if args.checkpoint else here / "checkpoints" / "best_multitask_unetpp.pth"
    if not ck_path.is_file():
        print(f"Missing checkpoint: {ck_path}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ── data ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    eval_df = df[df["split"] == args.split].copy()
    ds = MammogramMultiTaskDataset(eval_df, image_size=args.image_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating {len(ds)} samples from split='{args.split}'", flush=True)

    # ── model ────────────────────────────────────────────────────────
    model = MultiTaskUNetPP(
        encoder_name=args.encoder,
        encoder_weights=None,       # no imagenet at eval; weights from checkpoint
        in_channels=3,
        seg_classes=1,
    ).to(device)
    state = torch.load(ck_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ── evaluate ─────────────────────────────────────────────────────
    all_dice: list[float] = []
    all_iou: list[float] = []
    all_acc: list[float] = []
    all_patho: list[int] = []     # ground-truth label
    all_pred_cls: list[int] = []  # predicted label

    with torch.no_grad():
        for x, m, y in tqdm(loader, desc="eval"):
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            seg_logits, cls_logits = model(x)
            seg_pred = (torch.sigmoid(seg_logits) > 0.5).float()
            cls_pred = (cls_logits.squeeze(-1) > 0.0).long()

            for b in range(x.size(0)):
                all_dice.append(dice_coef(seg_pred[b], m[b]))
                all_iou.append(iou_coef(seg_pred[b], m[b]))
                all_acc.append(pixel_accuracy(seg_pred[b], m[b]))

            all_patho.extend(y.tolist())
            all_pred_cls.extend(cls_pred.cpu().tolist())

    # ── classification metrics ───────────────────────────────────────
    gt = np.array(all_patho)
    pr = np.array(all_pred_cls)
    tp = int(((gt == 1) & (pr == 1)).sum())
    tn = int(((gt == 0) & (pr == 0)).sum())
    fp = int(((gt == 0) & (pr == 1)).sum())
    fn = int(((gt == 1) & (pr == 0)).sum())
    cls_accuracy = (tp + tn) / max(len(gt), 1)
    cls_precision = tp / max(tp + fp, 1)
    cls_recall = tp / max(tp + fn, 1)
    cls_f1 = 2 * cls_precision * cls_recall / max(cls_precision + cls_recall, 1e-8)

    # ── per-class segmentation ───────────────────────────────────────
    dice_benign = [d for d, g in zip(all_dice, all_patho) if g == 0]
    dice_malig = [d for d, g in zip(all_dice, all_patho) if g == 1]

    metrics = {
        "checkpoint": str(ck_path.name),
        "split": args.split,
        "n_samples": len(ds),
        "device": str(device),
        "segmentation": {
            "mean_dice": round(float(np.mean(all_dice)), 5),
            "std_dice": round(float(np.std(all_dice)), 5),
            "mean_iou": round(float(np.mean(all_iou)), 5),
            "std_iou": round(float(np.std(all_iou)), 5),
            "mean_pixel_accuracy": round(float(np.mean(all_acc)), 5),
            "dice_benign_mean": round(float(np.mean(dice_benign)), 5) if dice_benign else None,
            "dice_malignant_mean": round(float(np.mean(dice_malig)), 5) if dice_malig else None,
        },
        "classification": {
            "accuracy": round(cls_accuracy, 5),
            "precision": round(cls_precision, 5),
            "recall": round(cls_recall, 5),
            "f1": round(cls_f1, 5),
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        },
    }

    # ── write ────────────────────────────────────────────────────────
    out_dir = ck_path.parent
    out_path = out_dir / "multitask_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
