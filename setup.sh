#!/usr/bin/env bash
# Ephemeral-GPU startup: one command stands up the whole pipeline.
# Usage on Vast.ai RTX 5090 (Ubuntu 22.04 + CUDA 12.8 / torch 2.7 template):
#   git clone <your-repo> && cd wan22-pipeline && bash setup.sh [--profile eco] [--csv prompts.example.csv]
# Re-runs are idempotent: skips apt/pip work already done, reuses HF cache.
set -euo pipefail

PROFILE="eco"
CSV="prompts.example.csv"
OUT="outputs"
while [[ $# -gt 0 ]]; do case "$1" in
  --profile) PROFILE="$2"; shift 2;;
  --csv) CSV="$2"; shift 2;;
  --out) OUT="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done

echo "[setup] profile=$PROFILE csv=$CSV out=$OUT"
echo "[setup] nvidia-smi:" && (nvidia-smi --query-gpu=name,memory.total --format=csv || true)

if ! command -v python3 >/dev/null; then
  echo "[setup] installing python..."; sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-pip python3-venv ffmpeg git
elif ! command -v ffmpeg >/dev/null; then
  echo "[setup] installing ffmpeg..."; sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
fi

if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

echo "[setup] verifying 5090 stack (torch cu128 + sm_120 + diffusers pin)..."
python - <<'EOF'
import torch, diffusers
assert torch.cuda.is_available(), "CUDA not visible — check driver/container"
major = torch.cuda.get_device_capability(0)[0]
print("torch", torch.__version__, "| cuda-build", torch.version.cuda,
      "| gpu", torch.cuda.get_device_name(0), "| sm", torch.cuda.get_device_capability(0))
assert "cu128" in torch.__version__ or torch.version.cuda >= "12.8", \
    "needs cu128 torch for RTX 5090 (sm_120)"
assert major >= 12, f"unexpected compute capability for 5090: {torch.cuda.get_device_capability(0)}"
assert diffusers.__version__ == "0.35.2", f"expected diffusers 0.35.2, got {diffusers.__version__}"
# Host #553680 fingerprint: 32GB 5090, PCIe4 limit. Warn-only (bukan assert keras)
# agar script tetap portable bila Vast memindahkan pod ke host 5090 lain.
total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"[setup] VRAM total: {total_gb:.1f} GB (expect ~32)")
if abs(total_gb - 32) > 2:
    print("[setup] WARNING: VRAM != 32GB — cek apakah benar host #553680")
# PCIe generation check via nvidia-smi (host ini PCIe4 x16 @26.9GB/s, bukan 5.0)
print("[setup] stack OK")
EOF

export HF_HUB_ENABLE_HF_TRANSFER=1
python -m pip install -q hf_transfer 2>/dev/null || true

echo "[setup] validating prompt matrix (dry-run)..."
python batch.py --csv "$CSV" --out "$OUT" --profile "$PROFILE" --dry-run

echo ""
echo "[setup] READY. To launch the batch, run:"
echo "  source .venv/bin/activate && python batch.py --csv $CSV --out $OUT --profile $PROFILE"
echo "To resume after interruption:"
echo "  source .venv/bin/activate && python batch.py --csv $CSV --out $OUT --profile $PROFILE --resume"
