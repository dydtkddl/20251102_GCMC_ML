# -*- coding: utf-8 -*-
# ActiveLearning/AL_Trainer.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os


class ActiveTrainer:
    """
    Unified Active Learning Trainer
    - Supports early stopping
    - MC-Dropout uncertainty prediction
    - Deterministic inference for final testing
    - Save/load for reuse
    """

    def __init__(self, model, optimizer, loss_fn,
                 epochs=300, patience=30,
                 device="cpu", outdir=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.outdir = outdir
        if outdir:
            os.makedirs(outdir, exist_ok=True)

    # ───────────────────────────────
    # Training
    # ───────────────────────────────
    def fit(self, train_dl, val_dl=None, verbose=True):
        best_loss, patience_ctr, best_state = float("inf"), 0, None
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for xb, yb, _ in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(xb)

            avg_train_loss = total_loss / len(train_dl.dataset)

            # Validation loss
            if val_dl is not None:
                self.model.eval()
                total_val, n_val = 0, 0
                with torch.no_grad():
                    for xb, yb, _ in val_dl:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        preds = self.model(xb)
                        total_val += self.loss_fn(preds, yb).item() * len(xb)
                        n_val += len(xb)
                val_loss = total_val / max(1, n_val)
            else:
                val_loss = avg_train_loss

            if verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                print(f"[Epoch {epoch:03d}] Train={avg_train_loss:.6f} | Val={val_loss:.6f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= self.patience:
                break

        if best_state:
            self.model.load_state_dict(best_state)

        if self.outdir:
            torch.save(self.model.state_dict(), os.path.join(self.outdir, "final_model.pth"))

    # ───────────────────────────────
    # Deterministic Prediction (no dropout)
    # ───────────────────────────────
    def predict_deterministic(self, X_tensor):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor.to(self.device)).cpu().numpy().flatten()
        return preds

    # ───────────────────────────────
    # MC-Dropout Prediction (for uncertainty)
    # ───────────────────────────────
    def predict_with_uncertainty(self, X_tensor, n_sim=20):
        self.model.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_sim):
                preds.append(self.model(X_tensor.to(self.device)).cpu().numpy())
        arr = np.array(preds)
        return arr.mean(axis=0).flatten(), arr.std(axis=0).flatten()

    # ───────────────────────────────
    # Unified Prediction API
    # ───────────────────────────────
    def predict(self, X_tensor, uncertainty=False, n_sim=20):
        if not uncertainty:
            return self.predict_deterministic(X_tensor), None
        else:
            return self.predict_with_uncertainty(X_tensor, n_sim=n_sim)

    # ───────────────────────────────
    # Evaluation metrics
    # ───────────────────────────────
    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False)
        }

    # ───────────────────────────────
    # Save & Load utilities
    # ───────────────────────────────
    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.outdir or ".", "final_model.pth")
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved → {path}")

    def load_model(self, path=None):
        if path is None:
            path = os.path.join(self.outdir or ".", "final_model.pth")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✅ Model loaded from {path}")
