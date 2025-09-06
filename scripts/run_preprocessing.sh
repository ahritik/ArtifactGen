#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/wgan_raw.yaml}
python -m src.preprocess --config "$CFG"
