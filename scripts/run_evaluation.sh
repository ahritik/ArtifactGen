#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/wgan_raw.yaml}
python -m src.eval.metrics_signal --config "$CFG"
python -m src.eval.metrics_feature --config "$CFG"
python -m src.eval.metrics_functional --config "$CFG"
python -m src.eval.utility_classifier --config "$CFG" --ckpt-ddpm results/checkpoints/ddpm_unet_best.pth --ckpt-wgan results/checkpoints/wgan_generator_best.pth --n-synth-per-class 500
