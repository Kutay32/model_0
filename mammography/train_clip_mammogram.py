"""
Fine-tune OpenAI CLIP (ViT-B/32) on CBIS mammogram crops + normal full-field images.

Run from repo root:
  .\\.venv\\Scripts\\python mammography\\train_clip_mammogram.py --manifest mammography\\cache\\manifest_classification.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import clip
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mammography.datasets import MammogramClipDataset

TEXT_PROMPTS = [
    "A screening mammogram with no suspicious lesion, normal breast tissue",
    "A mammogram showing a benign breast lesion or mass",
    "A mammogram showing a malignant breast lesion or mass",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument(
        "--checkpoint-dir",
        default="",
        help="Saves best_clip_classifier.pth (default: mammography/checkpoints).",
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

    model, _pre = clip.load("ViT-B/32", device=device)

    for p in model.transformer.parameters():
        p.requires_grad = False
    for p in model.token_embedding.parameters():
        p.requires_grad = False
    for p in model.ln_final.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(
        list(model.visual.parameters()) + [model.logit_scale],
        lr=args.lr,
        weight_decay=0.01,
    )

    train_ds = MammogramClipDataset(train_df, image_size=args.image_size, train=True)
    val_ds = MammogramClipDataset(val_df, image_size=args.image_size, train=False)
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

    text_tokens = clip.tokenize(TEXT_PROMPTS).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    ck_dir = Path(args.checkpoint_dir or (Path(__file__).resolve().parent / "checkpoints"))
    ck_dir.mkdir(parents=True, exist_ok=True)
    best_path = ck_dir / "best_clip_classifier.pth"
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs} train", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            img_f = model.encode_image(x)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (img_f @ text_feat.T)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="val", leave=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                img_f = model.encode_image(x)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * (img_f @ text_feat.T)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
        print(f"epoch {epoch+1}/{args.epochs}  val_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print(f"  saved {best_path}", flush=True)

    if not best_path.is_file():
        torch.save(model.state_dict(), best_path)
        print(f"Saved final weights (no val improvement) -> {best_path}", flush=True)

    print("Best val acc:", best_acc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
