"""
U-Net for binary mass segmentation (1 x 256 x 256) with hybrid BCE + Dice loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net for single-channel input and single-class logits."""

    def __init__(self, in_channels: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(c4, c4 * 2)

        self.up4 = nn.ConvTranspose2d(c4 * 2, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        self.out = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.mid(self.pool4(x4))

        u4 = self.up4(x5)
        u4 = self._crop_concat(u4, x4)
        u4 = self.dec4(u4)
        u3 = self.up3(u4)
        u3 = self._crop_concat(u3, x3)
        u3 = self.dec3(u3)
        u2 = self.up2(u3)
        u2 = self._crop_concat(u2, x2)
        u2 = self.dec2(u2)
        u1 = self.up1(u2)
        u1 = self._crop_concat(u1, x1)
        u1 = self.dec1(u1)
        return self.out(u1)

    @staticmethod
    def _crop_concat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dh = skip.size(2) - up.size(2)
        dw = skip.size(3) - up.size(3)
        up = F.pad(up, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return torch.cat((skip, up), dim=1)


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        t = targets.float()
        inter = (probs * t).sum(dim=(1, 2, 3))
        denom = probs.pow(2).sum(dim=(1, 2, 3)) + t.pow(2).sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class HybridSegmentationLoss(nn.Module):
    """BCEWithLogits + Dice (handles class imbalance)."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def build_model(device: torch.device | str = "cpu") -> tuple[UNet, HybridSegmentationLoss, torch.optim.Adam]:
    model = UNet(in_channels=1, base_ch=32).to(device)
    loss_fn = HybridSegmentationLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, loss_fn, optim


@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    t = (targets >= 0.5).float()
    inter = (preds * t).sum()
    union = preds.sum() + t.sum() - inter
    iou = (inter + 1e-8) / (union + 1e-8)
    dice = (2.0 * inter + 1e-8) / (preds.sum() + t.sum() + 1e-8)
    return float(dice.item()), float(iou.item())
