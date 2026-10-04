"""Stage 3 - train: build and train the CNN on data/processed/, save weights and history to models/."""
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

DATA_DIR = "data/processed"
MODEL_DIR = "models"


class SimpleCNN(nn.Module):
    """Three Conv -> BatchNorm -> ReLU -> MaxPool blocks, then Dense -> Dropout -> Output(10)."""

    def __init__(self, num_filters, dropout_rate, image_size=32, num_classes=10):
        super().__init__()
        f = num_filters

        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out),
                    nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(3, f), *block(f, 2 * f), *block(2 * f, 4 * f))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * f * (image_size // 8) ** 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_split(name):
    x = torch.from_numpy(np.load(os.path.join(DATA_DIR, f"x_{name}.npy")).astype(np.float32))
    y = torch.from_numpy(np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))).long()
    return x, y


@torch.no_grad()
def run_eval(model, x, y, loss_fn, device, batch_size=500):
    """Return (mean loss, accuracy, predicted labels) for one split."""
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(xb)
        preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds)
    return total_loss / len(x), (preds == y).float().mean().item(), preds


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    torch.manual_seed(params["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")

    model = SimpleCNN(params["num_filters"], params["dropout_rate"], image_size=x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
    loss_fn = nn.CrossEntropyLoss()
    print(f"[train] device={device}, train={len(x_train)}, val={len(x_val)}, params={params}")

    history, best_val_acc, best_state = [], 0.0, None
    for epoch in range(1, params["epochs"] + 1):
        start = time.time()
        model.train()
        order = torch.randperm(len(x_train))
        running_loss, correct = 0.0, 0
        for i in range(0, len(order), params["batch_size"]):
            idx = order[i:i + params["batch_size"]]
            xb, yb = x_train[idx], y_train[idx]
            flip = torch.rand(len(xb)) < 0.5            # light augmentation: random horizontal flip
            xb[flip] = xb[flip].flip(3)
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        scheduler.step()                                # decay the learning rate towards zero
        train_loss, train_acc = running_loss / len(x_train), correct / len(x_train)
        val_loss, val_acc, _ = run_eval(model, x_val, y_val, loss_fn, device)
        history.append([epoch, round(train_loss, 4), round(train_acc, 4), round(val_loss, 4), round(val_acc, 4)])
        print(f"[train] epoch {epoch}/{params['epochs']}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({time.time() - start:.0f}s)")

        if val_acc > best_val_acc:                      # keep the weights with the best validation accuracy
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"model_state": best_state, "num_filters": params["num_filters"],
                "dropout_rate": params["dropout_rate"], "image_size": x_train.shape[-1]},
               os.path.join(MODEL_DIR, "model.pth"))
    with open(os.path.join(MODEL_DIR, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(history)
    print(f"[train] best val_acc={best_val_acc:.4f}, saved {MODEL_DIR}/model.pth and {MODEL_DIR}/history.csv")


if __name__ == "__main__":
    main()
