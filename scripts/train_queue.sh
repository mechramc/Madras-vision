#!/usr/bin/env bash
set -euo pipefail

echo "[$(date)] Starting Madras Vision SFT..."
docker exec madras-vision python scripts/train_sft.py --config configs/qlora_sft.yaml
echo "[$(date)] Madras Vision SFT complete."

echo "[$(date)] Starting Project B SFT..."
docker exec project-b python scripts/train_sft.py --config configs/qlora_sft.yaml
echo "[$(date)] Project B SFT complete."

echo "[$(date)] All training jobs finished."
