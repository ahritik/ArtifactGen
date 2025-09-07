#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/wgan_raw.yaml}
source venv/Scripts/activate
python -m src.preprocess --config "$CFG"
