"""Stage 4 - evaluate: score the trained CNN on the test set, write metrics.json and a confusion matrix plot."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from train import MODEL_DIR, SimpleCNN, load_split, run_eval

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(os.path.join(MODEL_DIR, "model.pth"), map_location=device)
    model = SimpleCNN(checkpoint["num_filters"], checkpoint["dropout_rate"], checkpoint["image_size"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    x_test, y_test = load_split("test")
    test_loss, test_acc, preds = run_eval(model, x_test, y_test, nn.CrossEntropyLoss(), device)

    metrics = {"test_loss": round(test_loss, 4), "test_accuracy": round(test_acc, 4), "test_samples": len(y_test)}
    with open("metrics.json", "w", newline="\n") as f:   # fixed line endings keep the file hash stable
        json.dump(metrics, f, indent=2)

    matrix = confusion_matrix(y_test.numpy(), preds.numpy())
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"CIFAR-10 confusion matrix (test accuracy {test_acc:.2%})")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=120)

    print(f"[evaluate] {metrics}")
    print("[evaluate] saved metrics.json and confusion_matrix.png")


if __name__ == "__main__":
    main()
