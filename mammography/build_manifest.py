"""CLI: build manifest_segmentation.csv and manifest_classification.csv."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mammography.cbis_io import write_manifests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        required=True,
        help="Folder containing csv/ and jpeg/ (extracted CBIS-DDSM Kaggle bundle).",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: mammography/cache under repo).",
    )
    args = ap.parse_args()
    root = os.path.abspath(args.data_root)
    here = Path(__file__).resolve().parent
    out = args.out_dir or str(here / "cache")
    seg_p, cls_p = write_manifests(root, out)
    print("Wrote", seg_p)
    print("Wrote", cls_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
