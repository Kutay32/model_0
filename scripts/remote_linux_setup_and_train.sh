#!/usr/bin/env bash
# Run ON the Linux server (after repo is on disk), from repo root or pass path:
#   chmod +x scripts/remote_linux_setup_and_train.sh
#   ./scripts/remote_linux_setup_and_train.sh /root/model_0
set -euo pipefail

REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

if [[ ! -f "requirements-train.txt" ]]; then
  echo "requirements-train.txt not found under $REPO" >&2
  exit 1
fi

PYTHON="${PYTHON:-python3}"
$PYTHON -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip wheel

# Base stack (CLIP from git needs git on server)
python -m pip install -r requirements-train.txt

# Linux + NVIDIA: CUDA 12.4 wheels (adjust cu121/cu118 on older drivers — see pytorch.org)
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

python scripts/check_gpu.py

echo ""
echo "=== Örnek: multi-task eğitim (manifest gerekli) ==="
echo "python mammography/train_multitask.py \\"
echo "  --manifest mammography/cache/manifest_segmentation.csv \\"
echo "  --epochs 40 --batch-size 8 --checkpoint-dir mammography/checkpoints"
echo ""
echo "Uzun koşular için: tmux new -s train  →  komutu orada çalıştır  →  Ctrl+B D ile çık"
