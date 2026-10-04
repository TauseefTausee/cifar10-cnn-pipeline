# cifar10-cnn-pipeline

End-to-end ML versioning with Git, DVC and DagsHub (MLOps Assignment 2, student ID 24i-8024).

A reproducible CNN pipline for CIFAR-10 image classification. The code is versioned with Git. The data, the trained model and the metrics are versioned with DVC and stored on a DagsHub remote.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| prepare | `src/prepare.py` | raw CIFAR-10 arrays in `data/raw/` |
| preprocess | `src/preprocess.py` | normalized train/val/test arrays in `data/processed/` |
| train | `src/train.py` | `models/model.pth` and `models/history.csv` |
| evaluate | `src/evaluate.py` | `metrics.json` and `confusion_matrix.png` |

All hyperparameters live in `params.yaml`. The stages are wired together in `dvc.yaml`.

## Reproduce

```bash
pip install -r requirements.txt
dvc repro
dvc metrics show
```

`dvc repro` runs only the stages whose code, data or parameters changed since the last run.
