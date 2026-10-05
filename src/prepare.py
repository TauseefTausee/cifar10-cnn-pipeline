"""Stage 1 - prepare: download CIFAR-10 and save the raw arrays to data/raw/."""
import os

import numpy as np
from torchvision.datasets import CIFAR10

DOWNLOAD_DIR = ".cache/cifar10"  # torchvision download cache, ignored by Git and DVC
RAW_DIR = "data/raw"


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    for split, is_train in (("train", True), ("test", False)):
        dataset = CIFAR10(root=DOWNLOAD_DIR, train=is_train, download=True)
        images = np.asarray(dataset.data, dtype=np.uint8)      # (N, 32, 32, 3)
        labels = np.asarray(dataset.targets, dtype=np.int64)   # (N,)
        np.save(os.path.join(RAW_DIR, f"{split}_images.npy"), images)
        np.save(os.path.join(RAW_DIR, f"{split}_labels.npy"), labels)
        print(f"[prepare] {split}: images {images.shape}, labels {labels.shape}")
    print(f"[prepare] raw data saved to {RAW_DIR}/")


if __name__ == "__main__":
    main()
