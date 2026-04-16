"""
Download CBIS-DDSM (Kaggle mirror) via kagglehub.

Requires a Kaggle account token (same as Kaggle CLI):
  set KAGGLE_USERNAME=...
  set KAGGLE_KEY=...

Or place ~/.kaggle/kaggle.json
"""
from __future__ import annotations

import argparse
import os
import shutil


def main() -> int:
    p = argparse.ArgumentParser(description="Download CBIS-DDSM with kagglehub.")
    p.add_argument(
        "--dataset",
        default="awsaf49/cbis-ddsm-breast-cancer-image-dataset",
        help="Kaggle dataset slug",
    )
    p.add_argument(
        "--link-dir",
        default="",
        help="Optional: copy/symlink extracted bundle into this folder for stable paths.",
    )
    args = p.parse_args()
    try:
        import kagglehub
    except ImportError as e:
        print("Install kagglehub: pip install kagglehub")
        raise SystemExit(1) from e

    path = kagglehub.dataset_download(args.dataset)
    path = os.path.abspath(path)
    print(path)

    if args.link_dir:
        dst = os.path.abspath(args.link_dir)
        if os.path.lexists(dst):
            print(f"Skip link-dir: already exists: {dst}")
        else:
            parent = os.path.dirname(dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            try:
                os.symlink(path, dst, target_is_directory=True)
                print(f"Symlinked -> {dst}")
            except OSError:
                shutil.copytree(path, dst)
                print(f"Copied -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
