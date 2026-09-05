"""
Trains ELACNN on the ELA-map dataset, logging every run to MLflow and
saving the best checkpoint to models/ -- this is the "Train CNN (PyTorch,
MLflow)" box in the architecture diagram, and models/best_model.pt is the
"Model artifact (versioned)" the serving path loads next.

Run:
  python src/train.py
  mlflow ui --backend-store-uri file:./mlruns   # to browse runs
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import make_splits, ROOT_DIR
from model import ELACNN

MODELS_DIR = os.path.join(ROOT_DIR, "models")
MLFLOW_DB = os.path.join(ROOT_DIR, "mlflow.db")


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    acc = (preds == labels).float().mean().item()

    # Simple AUC via rank statistic (no sklearn dependency needed).
    pos = probs[labels == 1].numpy()
    neg = probs[labels == 0].numpy()
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        auc = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, labels).item()
    return {"loss": loss, "accuracy": acc, "auc": auc}


def train(epochs: int = 25, batch_size: int = 16, lr: float = 3e-4, seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds = make_splits(seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ELACNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(MODELS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("document-forgery-ela-cnn")

    best_val_auc = -1.0
    best_path = os.path.join(MODELS_DIR, "best_model.pt")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "train_size": len(train_ds), "val_size": len(val_ds),
            "model": "ELACNN", "input": "ELA-heatmap-128x128",
        })

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            scheduler.step()
            train_loss = running_loss / len(train_ds)
            val_metrics = evaluate(model, val_loader, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
            }, step=epoch)

            print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.3f}  "
                  f"val_auc={val_metrics['auc']:.3f}")

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                torch.save(model.state_dict(), best_path)

        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_artifact(best_path)

    print(f"\nBest val AUC: {best_val_auc:.3f}  -> saved to {best_path}")
    return best_val_auc


if __name__ == "__main__":
    train()
