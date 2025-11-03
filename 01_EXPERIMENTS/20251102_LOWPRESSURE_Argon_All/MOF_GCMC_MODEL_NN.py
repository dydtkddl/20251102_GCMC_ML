# -*- coding: utf-8 -*-
"""
MOF_NN_MODEL.py (Extended, Log-aware + Auto-scaling)
────────────────────────────────────────────
Residual Neural Network Trainer for MOF-GCMC regression.

Features:
- Configurable residual MLP
- Automatic log-transform & scaling (fit_scaler=True)
- Early stopping
- CSV logging (train/val loss)
- Metrics & prediction saving
"""

import os
import json
import time
import logging
from tqdm import tqdm
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ───────────────────────────────────────────────
def get_activation(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    elif name == "silu": return nn.SiLU()
    elif name == "gelu": return nn.GELU()
    elif name == "mish": return nn.Mish()
    else: raise ValueError(f"Unsupported activation: {name}")

# ───────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim, drop, act):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), act, nn.Dropout(drop),
            nn.Linear(dim, dim), act, nn.Dropout(drop)
        )
    def forward(self, x): return x + self.block(x)

class GCMCModel(nn.Module):
    def __init__(self, dim, h1, h2, drop, act):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(dim, h1), act)
        self.res1 = ResidualBlock(h1, drop, act)
        self.mid = nn.Sequential(nn.Linear(h1, h2), act)
        self.res2 = ResidualBlock(h2, drop, act)
        self.out = nn.Linear(h2, 1)
    def forward(self, x):
        x = self.input(x)
        x = self.res1(x)
        x = self.mid(x)
        x = self.res2(x)
        return self.out(x)

# ───────────────────────────────────────────────
class MOFModelTrainer:
    def __init__(self,
                 model_type: str = "nn",
                 model_params: Optional[Dict] = None,
                 outdir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 device: Optional[str] = None):
        self.model_type = model_type.lower()
        self.params = model_params or {}
        self.outdir = outdir
        self.logger = logger or self._make_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model().to(self.device)
        self.criterion = nn.MSELoss()

        # Log transform & scaling options
        self.low_features = self.params.get("low_features", [])
        self.apply_log_to_low = self.params.get("apply_log_to_low", False)
        self.fit_scaler = self.params.get("fit_scaler", False)
        self.num_threads = self.params.get("num_threads", 4)

        # CPU thread control
        torch.set_num_threads(self.num_threads)

        # Scalers (will be fitted internally if fit_scaler=True)
        self.scaler_X = None
        self.scaler_y = None

    # ───────────────────────────────────────────────
    def _make_logger(self):
        logger = logging.getLogger("MOFModelTrainer_NN")
        if not logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            h = logging.StreamHandler()
            h.setFormatter(fmt)
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger

    def _init_model(self):
        p = self.params
        act = get_activation(p.get("activation", "relu"))
        return GCMCModel(
            dim=p.get("input_dim"),
            h1=p.get("hidden_dim1", 128),
            h2=p.get("hidden_dim2", 64),
            drop=p.get("dropout", 0.1),
            act=act
        )

    # ───────────────────────────────────────────────
    def _apply_log_to_lowcols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log-transform to designated low-pressure features."""
        if not self.apply_log_to_low or len(self.low_features) == 0:
            return df
        df_mod = df.copy()
        for col in self.low_features:
            if col in df_mod.columns:
                df_mod[col] = np.log(np.clip(df_mod[col], 1e-12, None))
        return df_mod

    # ───────────────────────────────────────────────
    def _prepare_scalers(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit scalers internally (after log-transform)."""
        X_mod = self._apply_log_to_lowcols(X_train)
        self.scaler_X = StandardScaler().fit(X_mod)
        self.scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
        self.logger.info(f"[SCALE] StandardScaler fitted on log-adjusted X_train.")

    # ───────────────────────────────────────────────
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train model with early stopping, auto-scaling, and log transform."""
        if self.fit_scaler or (self.scaler_X is None or self.scaler_y is None):
            self._prepare_scalers(X_train, y_train)

        # Log-transform
        X_train_mod = self._apply_log_to_lowcols(X_train)
        X_val_mod = self._apply_log_to_lowcols(X_val) if X_val is not None else None

        # Scale
        X_train_scaled = self.scaler_X.transform(X_train_mod)
        y_train_scaled = self.scaler_y.transform(y_train.values.reshape(-1, 1)).ravel()

        if X_val_mod is not None:
            X_val_scaled = self.scaler_X.transform(X_val_mod)
            y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        else:
            X_val_scaled, y_val_scaled = None, None

        # Tensor
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)

        if X_val_scaled is not None:
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        else:
            X_val_t, y_val_t = None, None

        # ── Optimizer & loop ─────────────────────────
        lr = self.params.get("lr", 1e-3)
        epochs = self.params.get("epochs", 500)
        patience = self.params.get("patience", 30)
        batch_size = self.params.get("batch_size", 64)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_loss = float("inf")
        patience_ctr = 0
        best_state = None
        history = []

        self.logger.info(f"[TRAIN] NN | Epochs={epochs} | Batch={batch_size} | LR={lr} | Threads={self.num_threads}")
        self.logger.info(f"[INFO] Log-transform applied to: {self.low_features if self.apply_log_to_low else 'None'}")

        t0 = time.time()
        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            idx = torch.randperm(len(X_train_t))
            total_loss = 0.0
            for i in range(0, len(X_train_t), batch_size):
                xb = X_train_t[idx[i:i+batch_size]]
                yb = y_train_t[idx[i:i+batch_size]]
                optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)

            train_loss = total_loss / len(X_train_t)
            val_loss = None
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_t)
                    val_loss = self.criterion(val_pred, y_val_t).item()

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

            # Early stopping
            target_loss = val_loss if val_loss is not None else train_loss
            if target_loss < best_loss:
                best_loss = target_loss
                best_state = self.model.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                self.logger.info(f"⏹ Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.train_time = time.time() - t0
        self.logger.info(f"[TIME] Training took {self.train_time:.2f}s")

        # Save training log
        if self.outdir:
            os.makedirs(self.outdir, exist_ok=True)
            pd.DataFrame(history).to_csv(
                os.path.join(self.outdir, "train_log.csv"), index=False, encoding="utf-8-sig"
            )

    # ───────────────────────────────────────────────
    def predict(self, X_test):
        X_mod = self._apply_log_to_lowcols(X_test)
        X_proc = self.scaler_X.transform(X_mod)
        X_t = torch.tensor(X_proc, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_t).cpu().numpy().ravel()
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred

    # ───────────────────────────────────────────────
    def evaluate(self, X_test, y_true):
        y_pred = self.predict(X_test)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))) * 100
        metrics = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE(%)": mape}
        self.logger.info(f"[EVAL] R2={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")

        if self.outdir:
            with open(os.path.join(self.outdir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    # ───────────────────────────────────────────────
    def save_predictions(self, X_test, y_true):
        y_pred = self.predict(X_test)
        pred_df = X_test.copy()  # 원본 그대로
        pred_df["y_true"] = y_true.values
        pred_df["y_pred"] = y_pred
        if self.outdir:
            path = os.path.join(self.outdir, "predictions.csv")
            pred_df.to_csv(path, index=False, encoding="utf-8-sig")
            self.logger.info(f"[SAVE] predictions → {path}")
        return pred_df

    # ───────────────────────────────────────────────
    def save_predictions_full(self,
                              X_full: pd.DataFrame,
                              y_full: pd.Series,
                              train_idx: Union[list, np.ndarray],
                              test_idx: Union[list, np.ndarray],
                              id_col: str = "filename"):
        """Save model predictions for full dataset (Train/Test annotated)."""
        X_full = X_full.reset_index(drop=True)
        y_full = y_full.reset_index(drop=True)

        trained_features = list(X_full.columns.difference([id_col]))
        X_pred_input = X_full[trained_features].copy()
        y_pred_full = self.predict(X_pred_input)

        df_out = X_full.copy()
        df_out["y_true"] = y_full.values
        df_out["y_pred"] = y_pred_full
        df_out["Split"] = "Unknown"
        df_out.loc[np.array(train_idx, dtype=int), "Split"] = "Train"
        df_out.loc[np.array(test_idx, dtype=int), "Split"] = "Test"

        if id_col in df_out.columns:
            cols = [id_col] + [c for c in df_out.columns if c != id_col]
            df_out = df_out[cols]

        if self.outdir:
            os.makedirs(self.outdir, exist_ok=True)
            out_path = os.path.join(self.outdir, "predictions_full.csv")
            df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
            self.logger.info(f"[SAVE] full predictions → {out_path}")

        return df_out
