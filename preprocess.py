"""Stage 2 - preprocess: normalize the raw images, split off a validation set, save to data/processed/."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"


def compute_stats(train_images):
    """Normalization statistics: per-channel mean and std of the training images."""
    scaled = train_images.astype(np.float32) / 255.0
    mean = scaled.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = scaled.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    return mean, std


def normalize(images, mean, std):
    """uint8 (N, H, W, 3) -> float16 (N, 3, H, W): scale pixels to [0, 1], then standardize."""
    x = images.astype(np.float32)
    x /= 255.0
    x -= mean
    x /= std
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.float16)


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    train_images = np.load(os.path.join(RAW_DIR, "train_images.npy"))
    train_labels = np.load(os.path.join(RAW_DIR, "train_labels.npy"))
    test_images = np.load(os.path.join(RAW_DIR, "test_images.npy"))
    test_labels = np.load(os.path.join(RAW_DIR, "test_labels.npy"))

    if train_images.shape[1] != params["image_size"]:
        raise ValueError(f"expected {params['image_size']}px images, got {train_images.shape[1]}px")

    x_train, x_val, y_train, y_val = train_test_split(
        train_images, train_labels,
        test_size=params["val_size"], random_state=params["seed"], stratify=train_labels,
    )

    mean, std = compute_stats(x_train)

    os.makedirs(OUT_DIR, exist_ok=True)
    splits = (("train", x_train, y_train), ("val", x_val, y_val), ("test", test_images, test_labels))
    for name, images, labels in splits:
        np.save(os.path.join(OUT_DIR, f"x_{name}.npy"), normalize(images, mean, std))
        np.save(os.path.join(OUT_DIR, f"y_{name}.npy"), labels)


if __name__ == "__main__":
    main()
