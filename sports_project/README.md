# Sports Image Classification (PyTorch, Simple Semester Project)

## Project Goal

This project builds a **simple, complete, and reproducible** image classification pipeline for a sports dataset (Kaggle sports classification).

It is designed for a school semester project with focus on:
- data handling and preprocessing
- clean implementation
- reproducible experiments
- clear evaluation and interpretation

## Why This Design

- **Main model:** ResNet-18 (pretrained)  
  Chosen because it is easy to explain, stable, and fast enough for local training.
- **Backup model:** EfficientNet-B0  
  Available with one config switch if you need a stronger alternative.
- **Transfer learning:** best trade-off for 100 classes and limited local compute.
- **Moderate augmentations:** consistent with EDA findings (brightness variation + geometric variation), without risky aggressive crops.
- **Imbalance handling:** weighted cross-entropy loss (simple and effective for moderate imbalance).

## EDA-Aligned Choices

Based on EDA:
- images are already 224x224 and mostly RGB
- no corrupted files and no exact duplicates
- moderate class imbalance
- brightness and pose variation exist

So this project uses:
- image size 224
- ImageNet normalization (transfer-learning default)
- moderate training augmentation:
  - horizontal flip
  - small rotation
  - slight color jitter
  - small translation
- deterministic validation/test transforms

> Note: EDA computed dataset-specific mean/std. For transfer learning, this project defaults to ImageNet mean/std. You can switch manually in `src/transforms.py`.

## Class Merging (Optional)

Class merging is controlled from **one place**: `config.py`.

Default mapping includes safe examples:
- figure skating men -> figure skating
- figure skating women -> figure skating
- one-leg football -> football

How it works:
- mapping dictionary: `DEFAULT_CLASS_MERGE_MAP`
- enable/disable per experiment with `enable_class_merge`
- script `merge_classes.py` can preview before/after class counts

## Project Structure

```text
sports_project/
  README.md
  requirements.txt
  train.py
  evaluate.py
  predict.py
  config.py
  merge_classes.py
  src/
    __init__.py
    data.py
    transforms.py
    model.py
    trainer.py
    metrics.py
    utils.py
    visualize.py
  outputs/
```

## Installation

From this folder:

```bash
pip install -r requirements.txt
```

## Dataset Layout Supported

Either:

```text
sports/
  class_name_1/
  class_name_2/
  ...
```

Or split layout (Kaggle style):

```text
sports/
  train/class_name_1/... 
  valid/class_name_1/...   (or val/)
  test/class_name_1/...
```

If split folders are missing, the code creates stratified train/val/test split.

## Experiments

Three minimal experiments are predefined in `config.py`:
1. `baseline_no_aug`
2. `moderate_aug`
3. `moderate_aug_with_merge`

## Train

Run from project root (`sports_project`):

```bash
python train.py --data-dir /path/to/sports --experiment baseline_no_aug
python train.py --data-dir /path/to/sports --experiment moderate_aug
python train.py --data-dir /path/to/sports --experiment moderate_aug_with_merge
```

Useful overrides:

```bash
python train.py --data-dir /path/to/sports --experiment moderate_aug --batch-size 32 --epochs 20
python train.py --data-dir /path/to/sports --experiment moderate_aug --model-name efficientnet_b0
python train.py --data-dir /path/to/sports --experiment moderate_aug --disable-unfreeze
```

## Evaluate

After training, evaluate again from saved split/checkpoint:

```bash
python evaluate.py --experiment-dir outputs/moderate_aug_YYYYMMDD_HHMMSS
```

## Predict One Image

```bash
python predict.py --experiment-dir outputs/moderate_aug_YYYYMMDD_HHMMSS --image-path /path/to/image.jpg --top-k 5
```

## What Gets Saved

Inside each run folder under `outputs/`:
- `config_used.json`
- `class_mapping.json`
- `checkpoints/best_model.pt`
- `splits/train.csv`, `splits/val.csv`, `splits/test.csv`
- `metrics/epoch_history.csv`
- `metrics/summary_metrics.json`
- `metrics/test_classification_report.txt`
- `metrics/test_classification_report.json`
- `metrics/test_confusion_matrix.csv`
- `plots/training_curves.png`
- `plots/test_confusion_matrix.png`

## How to Interpret Outputs

- `summary_metrics.json`: key final metrics (accuracy, macro precision/recall/F1, top-k, loss)
- `epoch_history.csv`: per-epoch train/val progress for trend analysis
- `training_curves.png`: overfitting/underfitting check
- `classification_report`: per-class precision/recall/F1
- `test_confusion_matrix`: which classes are most confused

## Default Hyperparameters (Local-Friendly)

- image size: 224
- batch size: 32
- epochs: 20
- patience: 5
- optimizer: Adam
- head LR: 1e-3
- fine-tune LR after unfreeze: 1e-4
- weighted cross-entropy for class imbalance

## Possible Presentation Talking Points

1. Why transfer learning beats training from scratch for 100 classes.
2. Why moderate augmentation is safer for sports images than aggressive crops.
3. Why weighted loss is enough for imbalance ratio around 2.9.
4. What changed between baseline, augmentation, and augmentation+merge experiments.
5. How confusion matrix reveals visually similar sports categories.
6. How the project ensures reproducibility (seed, config snapshot, saved splits, saved checkpoint).
