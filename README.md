# ArtifactGEN: High-Fidelity Synthesis of EEG Artifacts

This repository implements a reproducible pipeline to synthesize multi-channel EEG artifact windows using two state-of-the-art generative paradigms:

- WGAN-GP with projection discriminator
- DDPM (1D U-Net + classifier-free guidance)

It targets the TUH EEG Artifact Corpus (TUAR) with subject-wise splits, robust preprocessing, and a comprehensive evaluation suite (signal-level, feature-space, and functional tasks like TRTS/TSTR and AugMix-style augmentation studies).

## Quickstart

1) Set up environment

- Python 3.10+
- Install dependencies

1) Prepare data

- Point `configs/*yaml` `data.dataset_root` to your local TUAR path
- Run preprocessing to populate `data/processed`

1) Train models

- Use provided configs to train WGAN-GP or DDPM

1) Evaluate

- Run signal, feature, and functional metrics

## Repo Layout

- configs/           YAML configs for experiments
- data/              Raw link + processed windows
- notebooks/         Exploration & visualization
- results/           Checkpoints, figures, manifest
- scripts/           Entrypoint bash scripts
- src/               Python source (preprocess, datasets, models, train, eval)


## Minimal Repro Steps

- Preprocess
  - scripts/run_preprocessing.sh configs/wgan_raw.yaml
- Train (WGAN example)
  - scripts/run_training.sh configs/wgan_raw.yaml
- Evaluate
  - scripts/run_evaluation.sh configs/wgan_raw.yaml

See `ENVIRONMENT.md` for pinned versions, `CITATIONS.bib` for references, and `LICENSE` for licensing. Replace example configs with your desired windows (1s/2s), filtering scheme (raw/filtered), and normalization strategies per model.

## Notes

- Subject-wise splits are enforced via metadata
- WGAN uses per-window min-max to [-1, 1] with min/max stored for inversion
- DDPM uses per-recording z-score normalization
- Privacy audit, Model/Data cards, and additional configs will be added alongside trained checkpoints
