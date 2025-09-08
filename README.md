# ArtifactGEN: High-Fidelity Synthesis of EEG Artifacts

This repository implements a reproducible pipeline to synthesize multi-channel EEG artifact windows using two state-of-the-art generative paradigms:

- WGAN-GP with projection discriminator
- DDPM (1D U-Net + classifier-free guidance)

It targets the TUH EEG Artifact Corpus (TUAR) with subject-wise splits, robust preprocessing, and a comprehensive evaluation suite (signal-level, feature-space, and functional tasks like TRTS/TSTR and AugMix-style augmentation studies).

## Project Status

✅ **Data Processing Complete**: TUAR dataset has been processed with subject-wise stratified splits (149 train / 32 val / 32 test subjects)

✅ **Exploration Analysis Done**: Comprehensive data exploration completed including:

- Label distribution analysis across 5 artifact classes (Muscle, Eye movement, Electrode, Chewing, Shiver)
- Channel-wise event frequency analysis
- Duration distribution analysis with recommended window lengths
- Dimensionality reduction visualizations (t-SNE, UMAP) of per-file artifact summaries
- Multi-label stratified splitting to ensure balanced representation

✅ **Training Infrastructure Ready**: Models configured and training scripts prepared for both WGAN-GP and DDPM architectures

✅ **Initial Training Runs Completed**: TensorBoard logs indicate successful training runs with GPU acceleration

## Associated Paper

The paper files are located in the `paper/` directory:

- Source: `paper/ArtifactGEN.tex`
- Bibliography: `paper/ArtifactGEN.bib`
- Style: `paper/neurips_2025.sty`
- Compiled PDF: `paper/ArtifactGEN.pdf`

## Quickstart

1) Set up environment

- Python 3.12+
- Install dependencies: `pip install -r requirements.txt`
- GPU support: CUDA 12.1+ for PyTorch acceleration, CuPy for MNE GPU operations

1) Prepare data

- Point `configs/*yaml` `data.dataset_root` to your local TUAR path
- Processed data already available in `data/processed/` including:
  - Subject-wise stratified splits (`suggested_splits_subjectwise_multilabel.csv`)
  - Class mappings (`class_map.csv`)
  - Pre-computed data statistics

1) Train models

- Use provided configs to train WGAN-GP or DDPM
- Models automatically detect and use GPU if available
- Monitor training with TensorBoard: `tensorboard --logdir results/tensorboard/`

1) Evaluate

- Run signal, feature, and functional metrics
- View results in `results/figures/` and `results/manifest.json`

## Repo Layout

- `configs/`           YAML configuration files for experiments (e.g., `ddpm_raw.yaml`, `wgan_raw.yaml`)
- `data/`              Data directory containing raw and processed datasets
  - `raw/`             Raw data files
  - `processed/`       Processed data including class mappings and split suggestions
- `notebooks/`         Jupyter notebooks for data exploration and visualization
  - `exploration.ipynb` Comprehensive TUAR dataset analysis and visualization
- `paper/`             Paper-related files for NeurIPS 2025 submission
  - `CITATIONS.bib`    Bibliography references
  - `neurips_2025.pdf` Compiled PDF of the paper
  - `neurips_2025.sty` LaTeX style file for NeurIPS formatting
  - `neurips_2025.tex` LaTeX source file for the paper
- `results/`           Output directory for model checkpoints, figures, and evaluation results
  - `checkpoints/`     Saved model weights
  - `figures/`         Generated plots and visualizations
  - `manifest.json`    Metadata about results
  - `tensorboard/`     Training logs and metrics
- `scripts/`           Bash scripts for running preprocessing, training, and evaluation
  - `run_preprocessing.sh` Script to preprocess raw data
  - `run_training.sh`      Script to train models
  - `run_evaluation.sh`    Script to evaluate trained models
- `src/`               Python source code
  - `dataset.py`       Dataset loading and preprocessing utilities
  - `preprocess.py`    Data preprocessing functions
  - `train.py`         Training scripts for WGAN and DDPM models
  - `eval/`            Evaluation modules
    - `metrics_feature.py`   Feature-space evaluation metrics
    - `metrics_functional.py` Functional evaluation metrics
    - `metrics_signal.py`     Signal-level evaluation metrics
  - `models/`          Model implementations
    - `ddpm.py`        Denoising Diffusion Probabilistic Model
    - `wgan.py`        Wasserstein GAN with Gradient Penalty
- `ENVIRONMENT.md`     Environment setup and dependency versions
- `LICENSE`            Project license
- `README.md`          This file
- `requirements.txt`   Python dependencies


## Current Features

### Data Processing

- **Subject-wise splits**: 213 total subjects split into 149 train / 32 val / 32 test
- **Multi-label stratification**: Ensures balanced representation of all 5 artifact classes
- **Window extraction**: Configurable window lengths (1s/2s) with overlap options
- **Normalization strategies**: Per-window min-max for WGAN, per-recording z-score for DDPM

### Model Architectures

- **WGAN-GP**: Projection discriminator, gradient penalty, spectral normalization
- **DDPM**: 1D U-Net with classifier-free guidance, configurable noise schedules
- **GPU acceleration**: Automatic CUDA detection for both PyTorch and MNE operations

### Evaluation Suite

- **Signal-level metrics**: Fidelity measures, reconstruction quality
- **Feature-space metrics**: Distribution matching, embedding comparisons
- **Functional metrics**: TRTS/TSTR evaluation, AugMix-style augmentation studies

## Minimal Repro Steps

- Preprocess: `scripts/run_preprocessing.sh configs/wgan_raw.yaml`
- Train (WGAN example): `scripts/run_training.sh configs/wgan_raw.yaml`
- Evaluate: `scripts/run_evaluation.sh configs/wgan_raw.yaml`

## Recent Updates

- **Python Version**: Updated to 3.12.11 for improved performance and compatibility
- **Data Exploration**: Complete TUAR dataset analysis with visualization notebooks
- **Training Infrastructure**: Configured for both WGAN-GP and DDPM with GPU support
- **Results Tracking**: TensorBoard integration for monitoring training progress
- **Documentation**: Updated setup instructions and project status

See `ENVIRONMENT.md` for pinned versions, `paper/CITATIONS.bib` for references, and `LICENSE` for licensing. Replace example configs with your desired windows (1s/2s), filtering scheme (raw/filtered), and normalization strategies per model.

## Notes

- **GPU Support**: The pipeline automatically detects and uses CUDA GPUs where possible. PyTorch models are moved to GPU, MNE filtering uses GPU acceleration if CuPy is installed, and DataLoaders use pinned memory for faster transfers.
- **Data Handling**: Subject-wise splits are enforced via metadata to prevent data leakage
- **Normalization**:
  - WGAN uses per-window min-max normalization to [-1, 1] with min/max values stored for inversion
  - DDPM uses per-recording z-score normalization
- **Models**:
  - WGAN-GP includes a projection discriminator for improved stability
  - DDPM uses a 1D U-Net architecture with classifier-free guidance
- **Evaluation**: Comprehensive metrics include signal fidelity, feature distribution matching, and functional performance on downstream tasks
- **Reproducibility**: All dependencies are pinned in `requirements.txt` and `ENVIRONMENT.md`
- **Future Additions**: Privacy audit, Model/Data cards, and additional configurations will be added alongside trained checkpoints

## Citation

If you use this work, please cite our PrePrint at arXiv:XXXX.XXXX (citation details to be added upon publication).
