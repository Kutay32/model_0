"""Load paired `.npy` images / masks for segmentation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


try:
    import albumentations as A

    _HAS_ALBU = True
except ImportError:
    A = None  # type: ignore[assignment]
    _HAS_ALBU = False


class MammogramNpyDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, train: bool = False) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.train = train
        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(self.images_dir.glob("*.npy")):
            m = self.masks_dir / img_path.name
            if m.is_file():
                self.samples.append((img_path, m))
        if not self.samples:
            raise FileNotFoundError(f"No paired .npy files in {images_dir} / {masks_dir}")
        self._aug = None
        if train and _HAS_ALBU and A is not None:
            self._aug = A.Compose([A.HorizontalFlip(p=0.5)])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.samples[idx]
        x = np.load(ip).astype(np.float32)
        y = np.load(mp).astype(np.float32)
        if x.ndim == 3:
            x = np.mean(x, axis=-1)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        if self._aug is not None:
            aug = self._aug(image=(x * 255.0).astype(np.uint8), mask=(y * 255.0).astype(np.uint8))
            x = aug["image"].astype(np.float32) / 255.0
            y = (aug["mask"].astype(np.float32) > 127.0).astype(np.float32)

        xi = torch.from_numpy(x).unsqueeze(0)
        yi = torch.from_numpy(y).unsqueeze(0)
        return xi, yi


def default_loader(data_root: str, client: str, train: bool = False) -> MammogramNpyDataset:
    base = os.path.join(data_root, client)
    return MammogramNpyDataset(os.path.join(base, "images"), os.path.join(base, "masks"), train=train)
