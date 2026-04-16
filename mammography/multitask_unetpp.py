"""
Multi-task UNet++ — joint segmentation + benign/malignant classification.

Architecture
------------
- Shared encoder : ResNet-34 (ImageNet pretrained) via ``segmentation_models_pytorch``
- Seg decoder    : UNet++ nested dense decoder → 1-channel logits
- Cls head       : Global-Average-Pool on deepest encoder features
                   → FC(512→256) → ReLU → Dropout → FC(256→1) → raw logit

Both heads receive gradients; the encoder is trained end-to-end.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ClassificationHead(nn.Module):
    """Lightweight head on top of the encoder's deepest feature map."""

    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),          # single logit: P(malignant)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: encoder's deepest feature map (B, C, H, W) → (B, 1)."""
        x = self.pool(x).flatten(1)        # (B, C)
        return self.fc(x)                  # (B, 1)


class MultiTaskUNetPP(nn.Module):
    """UNet++ (``smp``) with an extra classification head.

    Forward returns ``(seg_logits, cls_logits)`` where:
    - ``seg_logits`` : ``(B, 1, H, W)`` — raw logits before sigmoid
    - ``cls_logits`` : ``(B, 1)``       — raw logit before sigmoid
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        seg_classes: int = 1,
        cls_dropout: float = 0.3,
    ):
        super().__init__()

        # --- segmentation backbone ---
        self.unetpp = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=seg_classes,
            activation=None,            # return raw logits
        )

        # The encoder's deepest feature dimension (last entry in
        # ``encoder.out_channels``); for resnet34 this is 512.
        encoder_out_channels: list[int] = self.unetpp.encoder.out_channels
        deepest_channels = encoder_out_channels[-1]

        # --- classification head ---
        self.cls_head = ClassificationHead(
            in_features=deepest_channels,
            dropout=cls_dropout,
        )

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) Encoder
        features: list[torch.Tensor] = self.unetpp.encoder(x)
        # features[0] = stem, features[1..4] = blocks, features[-1] = deepest

        # 2) Segmentation decoder (operates on all feature levels)
        decoder_out: torch.Tensor = self.unetpp.decoder(features)
        seg_logits: torch.Tensor = self.unetpp.segmentation_head(decoder_out)

        # 3) Classification head (operates on deepest encoder features)
        cls_logits: torch.Tensor = self.cls_head(features[-1])

        return seg_logits, cls_logits
