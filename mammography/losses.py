"""
Multi-task loss for joint segmentation + classification.

    L_total = L_BCE_seg + L_Dice_seg  +  λ · L_BCE_cls

Both segmentation losses operate on raw **logits** (no sigmoid required).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class MultiTaskLoss(nn.Module):
    """Combined segmentation (BCE + Dice) and classification (BCE) loss.

    Parameters
    ----------
    cls_weight : float
        Multiplier *λ* for the classification loss term.  Default 0.5
        gives the segmentation task twice the gradient budget.
    """

    def __init__(self, cls_weight: float = 0.5):
        super().__init__()
        self.seg_bce = nn.BCEWithLogitsLoss()
        self.seg_dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.cls_bce = nn.BCEWithLogitsLoss()
        self.cls_weight = cls_weight

    def forward(
        self,
        seg_logits: torch.Tensor,   # (B, 1, H, W)
        seg_target: torch.Tensor,   # (B, 1, H, W)
        cls_logits: torch.Tensor,   # (B, 1)
        cls_target: torch.Tensor,   # (B,) or (B, 1)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(total_loss, details_dict)``."""
        l_bce_seg = self.seg_bce(seg_logits, seg_target)
        l_dice = self.seg_dice(seg_logits, seg_target)

        # Ensure cls_target shape matches cls_logits
        if cls_target.dim() == 1:
            cls_target = cls_target.unsqueeze(1)
        cls_target = cls_target.float()

        l_cls = self.cls_bce(cls_logits, cls_target)

        total = l_bce_seg + l_dice + self.cls_weight * l_cls

        details = {
            "bce_seg": l_bce_seg.item(),
            "dice_seg": l_dice.item(),
            "bce_cls": l_cls.item(),
            "total": total.item(),
        }
        return total, details
