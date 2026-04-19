"""
Wait until preprocessing finishes (manifest CSV is written at end of `data_pipeline`),
then run `python -m harmonia_vision.benchmark` with the given arguments.

Example (long centralized run after full CBIS preprocess):

  python -m harmonia_vision.wait_for_preprocess ^
    --data-root harmonia_processed_cbis_full ^
    -- ^
    --mode centralized --data-root harmonia_processed_cbis_full --preset long ^
    --checkpoint harmonia_checkpoints/cbis_full_centralized_long.pth
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Poll for manifest CSV, then run harmonia_vision.benchmark.",
    )
    p.add_argument("--data-root", required=True, help="Processed output root (manifest lives here).")
    p.add_argument(
        "--manifest",
        default=None,
        help="Explicit manifest path. Default: {data_root}/manifest_iid.csv",
    )
    p.add_argument("--poll-interval", type=float, default=15.0, help="Seconds between checks.")
    p.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Pass after '--', e.g. -- --mode centralized --data-root ... --preset long",
    )
    args = p.parse_args()

    root = Path(args.data_root).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else root / "manifest_iid.csv"

    argv = list(args.benchmark_args or [])
    if argv and argv[0] == "--":
        argv = argv[1:]

    print("=" * 72, flush=True)
    print("[wait_for_preprocess] Bu sürecin Python yorumlayıcısı (benchmark aynı exe ile çalışır):", flush=True)
    print(f"  sys.executable = {sys.executable}", flush=True)
    print(f"  sürüm          = {sys.version.split()[0]} ({platform.python_implementation()})", flush=True)
    print(f"  çalışma dizini = {os.getcwd()}", flush=True)
    print("=" * 72, flush=True)

    print(f"[wait_for_preprocess] Watching for {manifest}", flush=True)
    while not manifest.is_file():
        time.sleep(args.poll_interval)
        print(f"[wait_for_preprocess] Not ready yet, sleeping {args.poll_interval}s …", flush=True)

    print(f"[wait_for_preprocess] Found {manifest}, starting benchmark.", flush=True)
    cmd = [sys.executable, "-u", "-m", "harmonia_vision.benchmark", *argv]
    print("[wait_for_preprocess] Alt süreç komutu:", flush=True)
    print("  " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
