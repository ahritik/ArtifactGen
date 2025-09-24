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

1. Set up environment

- Python 3.12+
- Install dependencies: `pip install -r requirements.txt`
- GPU support: CUDA 12.1+ for PyTorch acceleration, CuPy for MNE GPU operations

1. Prepare data

- Point `configs/*yaml` `data.dataset_root` to your local TUAR path
- Processed data already available in `data/processed/` including:
  - Subject-wise stratified splits (`suggested_splits_subjectwise_multilabel.csv`)
  - Class mappings (`class_map.csv`)
  - Pre-computed data statistics

1. Train models

- Use provided configs to train WGAN-GP or DDPM
- Models automatically detect and use GPU if available
- Monitor training with TensorBoard: `tensorboard --logdir results/tensorboard/`

1. Evaluate

- Run the full evaluation suite (signal, feature, functional, utility classifier):

  
  ```bash
  # zsh/bash
  ./scripts/run_evaluation.sh configs/ddpm_raw.yaml
  ```

- Or call individual evaluators:

  
  ```bash
  # Signal-level metrics (band-power errors, channel correlation, PSD distance)
  python -m src.eval.metrics_signal --config configs/ddpm_raw.yaml \
    --ckpt results/checkpoints/ddpm_unet_best.pth --model-kind ddpm --n 256

  # Feature-space metrics placeholder (MMD/PRD, t-SNE/UMAP):
  python -m src.eval.metrics_feature --config configs/ddpm_raw.yaml

  # Functional metrics placeholder (TRTS/TSTR, AugMix-style):
  python -m src.eval.metrics_functional --config configs/ddpm_raw.yaml

  # NEW: Classifier-based utility evaluation (train on real vs real+synthetic; eval on real test)
  python -m src.eval.utility_classifier --config configs/ddpm_raw.yaml \
    --ckpt-ddpm results/checkpoints/ddpm_unet_best.pth \
    --ckpt-wgan results/checkpoints/wgan_generator_best.pth \
    --n-synth-per-class 500 --epochs 10 --batch-size 64 --lr 1e-3
  ```

- Key outputs are written to `results/` as CSV and LaTeX tables ready for inclusion in the paper.

## Jupyter Notebooks

Explore and visualize results in the following notebooks (open with Jupyter Lab or VS Code):

- `notebooks/exploration.ipynb` — Data exploration
- `notebooks/visualization.ipynb` — Visualizations
- `notebooks/class_comparison.ipynb` — Class comparison analysis
- `notebooks/model_comparison.ipynb` — Model comparison analysis

## Where to Find Everything

- **Configs:** `configs/` (YAML files for model and data settings)
- **Processed Data & Metadata:** `data/processed/` (splits, class maps, statistics)
- **Checkpoints & Results:** `results/checkpoints/`, `results/generated/`, `results/manifest.json`, `results/split_summary.json`
- **TensorBoard Logs:** `results/tensorboard/`
- **Paper Source & Figures:** `paper/`
- **Scripts:** `scripts/` (for all main pipeline steps)
- **Source Code:** `src/` (Python modules for all core logic)
  - `src/eval/utility_classifier.py` — trains a small 1D CNN on `real`, `real+wgan`, and `real+ddpm`, then evaluates on the held-out real test set; writes `results/utility_classifier.csv` and `results/table_utility.tex`.

## Tips

- All scripts are compatible with PowerShell on Windows.
- Update config paths as needed for your local data.
- For troubleshooting, see comments in scripts and notebooks.

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

- **Signal-level metrics**: Welch band-power relative errors (δ/θ/α/β), channel-wise correlation, PSD L2 distance
- **Feature-space metrics**: Distribution matching (MMD/PRD), embedding comparisons (t-SNE/UMAP)
- **Functional metrics**: TRTS/TSTR evaluation, AugMix-style augmentation studies
- **Utility classifier (NEW)**: Train a small 1D CNN on (i) real-only, (ii) real+WGAN, (iii) real+DDPM; evaluate accuracy and macro-F1 on the real test set to measure whether synthetic data improve performance on real data.

#### Utility Classifier Protocol (what it does and why)

- We measure downstream utility by training an identical classifier on three training sets and always evaluating on the held-out real test set:
  1) `real` (baseline), 2) `real+wgan`, 3) `real+ddpm`.
- Synthetic windows are generated per class using the best checkpoints with explicit class conditioning (we pass `class_id=c` into the generator), so each artifact class is represented accurately.
- To avoid normalization confounds, both real and synthetic training windows are z-scored per window at classifier-training time.
- We report accuracy and macro-F1 (class-balanced) on the real test set.
- Reproducible outputs:
  - `results/utility_classifier.csv`
  - `results/table_utility.tex`

Run it directly:

```bash
python -m src.eval.utility_classifier \
  --config configs/ddpm_raw.yaml \
  --ckpt-ddpm results/checkpoints/ddpm_unet_best.pth \
  --ckpt-wgan results/checkpoints/wgan_generator_best.pth \
  --n-synth-per-class 500 --epochs 10 --batch-size 64 --lr 1e-3
```

## Minimal Repro Steps

- Preprocess: `scripts/run_preprocessing.sh configs/wgan_raw.yaml`
- Train (WGAN example): `scripts/run_training.sh configs/wgan_raw.yaml`
- Evaluate: `scripts/run_evaluation.sh configs/wgan_raw.yaml`

### Notable Evaluation Artifacts

- Signal metrics: `results/signal_metrics.csv`, LaTeX `results/table_bandpower.tex`
- Utility classifier: `results/utility_classifier.csv`, LaTeX `results/table_utility.tex`

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

## How to Run Everything

All main scripts are in the `scripts/` folder and can be run from PowerShell on Windows:

- **Preprocessing:**

  ```powershell
  ./scripts/run_preprocessing.sh
  ```
- **Training:**

  ```powershell
  ./scripts/run_training.sh
  ```
- **Evaluation:**

  ```powershell
  ./scripts/run_evaluation.sh
  
