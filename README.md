# Drywall QA Prompted Segmentation

A deep learning project for segmenting cracks and drywall joints in construction images using CLIPSeg (CLIP-based Segmentation) with text prompts.

## Project Overview

This project fine-tunes the CLIPSeg model to perform prompted segmentation for two primary tasks:
- **Crack Detection**: Identifying structural cracks, wall cracks, and surface cracks
- **Drywall Joint Detection**: Segmenting drywall seams and taping areas

The model uses text prompts (e.g., "wall crack", "segment drywall seam") to guide the segmentation process, enabling flexible, language-driven segmentation.

## Prerequisites

- Python 3.11
- [UV](https://docs.astral.sh/uv/) package manager

## Environment Setup

### 1. Install UV

If you don't have UV installed, install it using:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Navigate to Project

```bash
cd /path/to/10x
```

### 3. Create Virtual Environment with UV

UV will automatically create a virtual environment and install dependencies based on [pyproject.toml](pyproject.toml):

```bash
uv sync
```

This command:
- Creates a virtual environment in `.venv/` (if not already present)
- Installs Python 3.11 (specified in `.python-version`)
- Installs all dependencies from [pyproject.toml](pyproject.toml)

### 4. Activate Virtual Environment

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 5. Verify Installation

```bash
python --version  # Should show Python 3.11.x
```

## Project Structure

### Source Code (`src/`)

The [src/](src/) directory contains all the core Python modules:

- **[dataset.py](src/dataset.py)** - Dataset handling for both crack and drywall joint annotations
  - `DryWallQADataset`: Custom PyTorch dataset class
  - `create_combined_dataset()`: Combines crack and drywall datasets
  - Handles polygon (crack) and bounding box (drywall) annotations
  - Supports multiple text prompts per category for data augmentation

- **[load.py](src/load.py)** - Data loading utilities for VOC-format datasets
  - `get_dataset_files()`: Loads image paths and XML annotations
  - Parses VOC XML format for both datasets

- **[train.py](src/train.py)** - Model training pipeline
  - `setup_model()`: Initializes CLIPSeg model and processor
  - `configure_trainable_parameters()`: Sets up encoder/decoder learning rates
  - `train_one_epoch()`: Training loop with checkpointing
  - `evaluate()`: Validation loss computation
  - Supports differential learning rates (encoder: 1e-6, decoder: 1e-4)

- **[metrics.py](src/metrics.py)** - Evaluation metrics and visualization
  - `calculate_iou()`: Intersection over Union metric
  - `calculate_dice()`: Dice coefficient
  - `calculate_pixel_accuracy()`: Pixel-wise accuracy
  - `evaluate_model_with_metrics()`: Comprehensive evaluation on validation set
  - `visualize_predictions_with_metrics()`: Visual comparison of predictions

- **[process.py](src/process.py)** - Post-processing utilities
  - `process_predicted_mask()`: Cleans and thresholds predicted masks
  - Applies morphological operations for noise reduction

- **[visualize.py](src/visualize.py)** - Visualization tools
  - `visualize_samples()`: Display training samples with masks
  - `detailed_sample_analysis()`: Analyze mask statistics
  - `visualize_preds_and_masks()`: Compare predictions with ground truth

- **[main.py](src/main.py)** - Main training script (command-line interface)

- **[train_nb.ipynb](src/train_nb.ipynb)** - Interactive training notebook (see below)

## Training Notebook

The [src/train_nb.ipynb](src/train_nb.ipynb) notebook provides an end-to-end interactive workflow for training and evaluating the CLIPSeg model.

### Notebook Overview

**Purpose**: Complete pipeline from data loading to model evaluation with visualizations

**Workflow**:

1. **Data Loading** (Cells 1-3)
   - Imports all necessary modules from `src/`
   - Loads crack detection dataset (VOC format from `cracks.v1i.voc/`)
   - Loads drywall joint dataset (VOC format from `Drywall-Join-Detect.v1i.voc/`)
   - Creates combined training (5,984 samples) and validation (403 samples) datasets

2. **Data Visualization** (Cell 4)
   - Displays sample images with ground truth masks
   - Shows detailed mask statistics (coverage, dimensions)
   - Verifies data loading correctness

3. **Model Setup** (Cells 5-7)
   - Loads pre-trained CLIPSeg model with checkpoint
   - Configures trainable parameters (150M parameters)
   - Sets up differential learning rates:
     - Encoder: 1e-6 (fine-tuning CLIP features)
     - Decoder: 1e-4 (training segmentation head)
   - Creates data loaders with batch size 16

4. **Training Loop** (Cell 8)
   - Trains for 10 epochs with BCEWithLogitsLoss
   - Evaluates every 170 steps
   - Saves best model based on validation loss
   - Tracks training/validation loss curves
   - Uses AdamW optimizer with weight decay

5. **Test Set Evaluation** (Cells 9-10)
   - Creates test dataset (4 samples)
   - Visualizes predictions on test samples
   - Compares raw model output with ground truth

6. **Comprehensive Metrics** (Cell 11)
   - Evaluates entire validation set
   - Calculates metrics by category:
     - Overall performance
     - Crack-specific metrics
     - Drywall joint-specific metrics
   - Reports: mIoU, mDice, Pixel Accuracy (with standard deviations)
   - Generates bar charts comparing performance

7. **Detailed Sample Analysis** (Cells 12-13)
   - Visualizes 6 random validation samples
   - Shows side-by-side comparison:
     - Input image
     - Ground truth mask
     - Raw prediction
     - Processed prediction (post-processing applied)
   - Reports per-sample metrics
   - Analyzes impact of post-processing

### Key Features

- **Interactive Development**: Modify hyperparameters and re-run cells
- **Visual Feedback**: Inline plots show training progress and predictions
- **Metric Tracking**: Comprehensive evaluation with statistical analysis
- **Checkpoint Management**: Saves best models during training
- **Post-processing Analysis**: Compares raw vs. processed predictions

### Running the Notebook

```bash
# Make sure UV environment is activated
source .venv/bin/activate

# Start Jupyter
jupyter notebook src/train_nb.ipynb
```

Or use VS Code's built-in notebook support.

## Dependencies

All dependencies are managed via [pyproject.toml](pyproject.toml):

- **Core ML**: PyTorch 2.9.0, torchvision 0.24.0
- **Transformers**: Hugging Face transformers 4.57.1 (CLIPSeg)
- **Data Processing**: datasets 4.3.0, Pillow 12.0.0, opencv-python 4.12.0
- **ML Tools**: scikit-learn 1.7.2
- **Visualization**: matplotlib 3.10.7, seaborn 0.13.2
- **Jupyter**: ipykernel 7.0.1

## Quick Start

### Using the Notebook (Recommended)

```bash
# Activate environment
source .venv/bin/activate

# Launch notebook
jupyter notebook src/train_nb.ipynb

# Run cells sequentially to train and evaluate
```

### Using the CLI

```bash
# Activate environment
source .venv/bin/activate

# Run training script
python src/main.py
```

## Model Checkpoints

Trained models are saved in [checkpoints/](checkpoints/) with the format:
- `best_clipseg_drywall_model_v3.pth` - Best model checkpoint
- `clipseg_drywall_model_{epoch}.pth` - Epoch-specific checkpoints

Checkpoints include:
- Model state dict
- Optimizer state dict
- Training/validation loss
- Epoch number

## Datasets

The project uses two VOC-format datasets:
1. **Crack Detection** (`cracks.v1i.voc/`) - Polygon annotations
2. **Drywall Joint Detection** (`Drywall-Join-Detect.v1i.voc/`) - Bounding box annotations

Both datasets are expected in the project root directory.

## Performance

Based on validation set evaluation (403 samples):
- **Overall mIoU**: ~0.60-0.70 (good segmentation quality)
- **Crack Detection**: High precision for structural/wall cracks
- **Drywall Joints**: Effective seam and taping area segmentation

See [src/train_nb.ipynb](src/train_nb.ipynb) for detailed metric breakdowns.

## Troubleshooting

### UV sync fails
```bash
# Clean and reinstall
rm -rf .venv uv.lock
uv sync
```

### Module import errors
```bash
# Ensure you're in the project root and environment is activated
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA out of memory
Reduce batch size in training config (default: 16 � 8 or 4)
