# Agricultural Weed Segmentation UNet

Semantic segmentation of agricultural weeds and crops using a U-Net architecture. This repository contains a PyTorch implementation for tile-based training on a Sugarbeets dataset with RGB images and class annotations.

## Repository Structure

- `src/`
  - `dataset.py` - custom tile dataset loader with optional copy-paste augmentation
  - `model.py` - U-Net model definition for multi-class segmentation
  - `engine.py` - training, validation, and evaluation loop with accuracy, IoU, and F1 metrics
  - `experiments.py` - experiment runner for comparing learning rates, optimizers, and model capacity
- `data/`
  - `rgb/` - input RGB images
  - `annotations/` - ground truth mask images
  - `train.txt` / `test.txt` - split files listing annotation filenames
- `.gitignore` - ignores Python cache files and compiled bytecode

## Requirements

- Python 3.10+ (or compatible)
- PyTorch
- torchvision
- numpy
- opencv-python
- scikit-learn
- tqdm

Install dependencies with pip:

```bash
pip install torch torchvision numpy opencv-python scikit-learn tqdm
```

## Usage

Run the experiment script from the repository root:

```bash
python src/experiments.py
```

The script will:

1. Load training tiles from `data/train.txt`
2. Split them into training and validation subsets
3. Train the U-Net model for multiple experiments
4. Evaluate accuracy, mean IoU, and macro F1 score on validation data

## Data Format

- `data/rgb/rgb_XXXXX.png` - input images
- `data/annotations/XXXXX.png` - corresponding masks
- Mask colors:
  - Red `[255, 0, 0]` → class 1
  - Blue `[0, 0, 255]` → class 2
  - all other pixels → class 0

## Dataset and Augmentation

The dataset loader samples fixed-size tiles from full images:
- `tile_size=256`
- `overlap=0` by default
- optional copy-paste augmentation can paste object regions from one tile into another

## Model

The U-Net implementation uses:
- 4 downsampling blocks with `DoubleConv`
- 4 upsampling transpose convolutions
- final `1x1` convolution to predict `3` output classes

## Notes

- The training script uses a weighted cross-entropy loss to account for class imbalance
- The model runs on `cuda` if available, otherwise on CPU
<<<<<<< HEAD
- Designed for modest hardware like a GTX 1650 Super with 16GB RAM, using tile-based loading, smaller batch sizes, and lightweight U-Net blocks for optimization
=======
>>>>>>> d0fe1c063bd016cc38a4c89bf0beec01e8c36bb2
- You can change experiment parameters in `src/experiments.py`
