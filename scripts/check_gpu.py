#!/usr/bin/env python3
"""Quick CUDA check + tiny GPU workload. Run on the training machine (local or remote)."""
from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"PyTorch {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")
    if cuda:
        print(f"Device 0: {torch.cuda.get_device_name(0)}")
        print(f"Capability: {torch.cuda.get_device_capability(0)}")
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"VRAM free/total: {free // 1_048_576} / {total // 1_048_576} MiB")
        except Exception as e:
            print(f"(VRAM info unavailable: {e})")
    dev = torch.device("cuda" if cuda else "cpu")
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    c = a @ b
    torch.cuda.synchronize() if cuda else None
    print(f"Matmul ok, result mean: {float(c.mean()):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
