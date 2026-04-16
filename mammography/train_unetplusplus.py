"""
Train UNet++ on CBIS-DDSM lesion segmentation (cropped patch + ROI mask).

Run from repo root:
  .\\.venv\\Scripts\\python mammography\\train_unetplusplus.py --manifest mammography\\cache\\manifest_segmentation.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mammography.datasets import MammogramSegDataset


def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest_segmentation.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--max-samples", type=int, default=0, help="Cap training rows (debug).")
    ap.add_argument(
        "--checkpoint-dir",
        default="",
        help="Where to save best_unetplusplus_model.pth (default: mammography/checkpoints).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    if args.max_samples > 0:
        train_df = train_df.head(args.max_samples).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = MammogramSegDataset(train_df, image_size=args.image_size, train=True)
    val_ds = MammogramSegDataset(val_df, image_size=args.image_size, train=False)
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
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}", flush=True)

    model = smp.UnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ck_dir = Path(args.checkpoint_dir or (Path(__file__).resolve().parent / "checkpoints"))
    ck_dir.mkdir(parents=True, exist_ok=True)
    best_path = ck_dir / "best_unetplusplus_model.pth"
    best_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        loss_tr = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs} train", leave=False)
        for x, m in pbar:
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = dice_loss(logits, m) + bce(logits, m)
            loss.backward()
            opt.step()
            loss_tr += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        loss_tr /= max(len(train_ds), 1)

        model.eval()
        dices: list[float] = []
        with torch.no_grad():
            for x, m in tqdm(val_loader, desc="val", leave=False):
                x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
                prob = torch.sigmoid(model(x))
                pred = (prob > 0.5).float()
                for b in range(x.size(0)):
                    dices.append(dice_coef(pred[b], m[b]).item())
        mean_dice = float(sum(dices) / max(len(dices), 1))
        print(
            f"epoch {epoch+1}/{args.epochs}  train_loss={loss_tr:.4f}  val_dice={mean_dice:.4f}",
            flush=True,
        )
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), best_path)
            print(f"  saved {best_path}", flush=True)

    print("Best val dice:", best_dice, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
