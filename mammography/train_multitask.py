"""
Train multi-task UNet++ on CBIS-DDSM (segmentation + benign/malignant).

Run from repo root::

    .\\.venv\\Scripts\\python mammography\\train_multitask.py ^
        --manifest mammography\\cache\\manifest_segmentation.csv

Saves best checkpoint to ``mammography/checkpoints/best_multitask_unetpp.pth``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mammography.datasets import MammogramMultiTaskDataset
from mammography.losses import MultiTaskLoss
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


# ── main ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Train multi-task UNet++ (seg + cls).")
    ap.add_argument("--manifest", required=True, help="manifest_segmentation.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--cls-weight", type=float, default=0.5,
                    help="Weight λ for classification loss term (default 0.5).")
    ap.add_argument("--patience", type=int, default=10,
                    help="Early-stopping patience on validation Dice.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Cap training rows (debug).")
    ap.add_argument("--checkpoint-dir", default="",
                    help="Where to save best .pth (default: mammography/checkpoints).")
    args = ap.parse_args()

    # ── data ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    if args.max_samples > 0:
        train_df = train_df.head(args.max_samples).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        flush=True,
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = MammogramMultiTaskDataset(train_df, image_size=args.image_size, train=True)
    val_ds = MammogramMultiTaskDataset(val_df, image_size=args.image_size, train=False)

    nw = 0 if os.name == "nt" else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Train: {len(train_ds)} samples ({len(train_loader)} batches)  "
        f"Val: {len(val_ds)} samples ({len(val_loader)} batches)",
        flush=True,
    )

    # ── model ────────────────────────────────────────────────────────
    model = MultiTaskUNetPP(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        seg_classes=1,
    ).to(device)

    criterion = MultiTaskLoss(cls_weight=args.cls_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5, verbose=True,
    )

    ck_dir = Path(args.checkpoint_dir or (Path(__file__).resolve().parent / "checkpoints"))
    ck_dir.mkdir(parents=True, exist_ok=True)
    best_path = ck_dir / "best_multitask_unetpp.pth"
    log_path = ck_dir / "train_multitask_log.json"

    best_dice = 0.0
    patience_counter = 0
    history: list[dict] = []

    # ── training loop ────────────────────────────────────────────────
    t_start = time.time()
    for epoch in range(args.epochs):
        # --- train ---
        model.train()
        loss_accum = 0.0
        comp_accum: dict[str, float] = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs} train", leave=False)
        for x, m, y in pbar:
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            seg_logits, cls_logits = model(x)
            loss, details = criterion(seg_logits, m, cls_logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_accum += loss.item() * x.size(0)
            for k, v in details.items():
                comp_accum[k] = comp_accum.get(k, 0.0) + v * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n_train = max(len(train_ds), 1)
        loss_train = loss_accum / n_train
        comp_train = {k: v / n_train for k, v in comp_accum.items()}

        # --- validate ---
        model.eval()
        dices: list[float] = []
        ious: list[float] = []
        cls_correct = 0
        cls_total = 0
        with torch.no_grad():
            for x, m, y in tqdm(val_loader, desc="val", leave=False):
                x = x.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                seg_logits, cls_logits = model(x)
                seg_prob = torch.sigmoid(seg_logits)
                seg_pred = (seg_prob > 0.5).float()

                for b in range(x.size(0)):
                    dices.append(dice_coef(seg_pred[b], m[b]))
                    ious.append(iou_coef(seg_pred[b], m[b]))

                cls_pred = (cls_logits.squeeze(-1) > 0.0).long()
                cls_correct += (cls_pred == y).sum().item()
                cls_total += y.numel()

        mean_dice = sum(dices) / max(len(dices), 1)
        mean_iou = sum(ious) / max(len(ious), 1)
        cls_acc = cls_correct / max(cls_total, 1)

        scheduler.step(mean_dice)

        record = {
            "epoch": epoch + 1,
            "train_loss": round(loss_train, 5),
            **{f"train_{k}": round(v, 5) for k, v in comp_train.items()},
            "val_dice": round(mean_dice, 5),
            "val_iou": round(mean_iou, 5),
            "val_cls_acc": round(cls_acc, 5),
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"epoch {epoch+1}/{args.epochs}  "
            f"loss={loss_train:.4f}  "
            f"val_dice={mean_dice:.4f}  val_iou={mean_iou:.4f}  "
            f"val_cls_acc={cls_acc:.4f}",
            flush=True,
        )

        # --- checkpointing ---
        if mean_dice > best_dice:
            best_dice = mean_dice
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  [BEST] saved {best_path}  (best dice={best_dice:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={args.patience}).", flush=True)
                break

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min.  Best val Dice = {best_dice:.4f}", flush=True)

    # ── save log ─────────────────────────────────────────────────────
    summary = {
        "best_val_dice": round(best_dice, 5),
        "total_epochs": len(history),
        "elapsed_seconds": round(elapsed, 1),
        "args": vars(args),
        "history": history,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training log -> {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
