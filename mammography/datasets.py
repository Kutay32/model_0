from __future__ import annotations

import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class MammogramSegDataset(Dataset):
    """Binary lesion mask on cropped CBIS patch."""

    def __init__(self, df: pd.DataFrame, image_size: int = 256, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.train = train
        geo = [A.Resize(image_size, image_size)]
        if train:
            geo += [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02, scale_limit=0.05, rotate_limit=8, border_mode=0, p=0.5
                ),
            ]
        self.tf = A.Compose(
            geo
            + [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(row["mask_path"]).convert("L"))
        if mask.shape[:2] != image.shape[:2]:
            mask = np.array(
                Image.fromarray(mask).resize(
                    (image.shape[1], image.shape[0]),
                    resample=Image.NEAREST,
                )
            )
        mask = (mask > 0).astype(np.uint8)
        aug = self.tf(image=image, mask=mask)
        x = aug["image"]
        m = aug["mask"].float().unsqueeze(0)
        return x, m


_PATHO_TO_IDX = {"benign": 0, "malignant": 1}


class MammogramMultiTaskDataset(Dataset):
    """Segmentation mask + binary classification (benign=0, malignant=1).

    Each sample returns ``(image, mask, label)`` where *image* is a
    normalised RGB tensor, *mask* is a ``(1, H, W)`` binary float tensor
    and *label* is a scalar ``{0, 1}`` integer.
    """

    def __init__(self, df: pd.DataFrame, image_size: int = 256, train: bool = True):
        # Keep only rows with a valid pathology label
        df = df[df["pathology"].str.lower().isin(_PATHO_TO_IDX)].copy()
        self.df = df.reset_index(drop=True)
        self.train = train
        geo = [A.Resize(image_size, image_size)]
        if train:
            geo += [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02, scale_limit=0.05, rotate_limit=15, border_mode=0, p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ]
        self.tf = A.Compose(
            geo
            + [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(row["mask_path"]).convert("L"))
        if mask.shape[:2] != image.shape[:2]:
            mask = np.array(
                Image.fromarray(mask).resize(
                    (image.shape[1], image.shape[0]),
                    resample=Image.NEAREST,
                )
            )
        mask = (mask > 0).astype(np.uint8)
        aug = self.tf(image=image, mask=mask)
        x = aug["image"]
        m = aug["mask"].float().unsqueeze(0)
        label = _PATHO_TO_IDX[str(row["pathology"]).strip().lower()]
        return x, m, label


_LABEL_TO_IDX = {"normal": 0, "benign": 1, "malignant": 2}


class MammogramClipDataset(Dataset):
    """RGB images for CLIP fine-tuning (3-way)."""

    def __init__(self, df: pd.DataFrame, image_size: int = 224, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.train = train
        geo = [A.Resize(image_size, image_size)]
        if train:
            geo += [A.HorizontalFlip(p=0.5)]
        self.tf = A.Compose(
            geo
            + [
                A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        aug = self.tf(image=image)
        y = _LABEL_TO_IDX[str(row["label"]).lower()]
        return aug["image"], y
