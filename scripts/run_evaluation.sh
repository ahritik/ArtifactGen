#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/wgan_raw.yaml}
python -m src.eval.metrics_signal --config "$CFG"
python -m src.eval.metrics_feature --config "$CFG"
python -m src.eval.metrics_functional --config "$CFG"
