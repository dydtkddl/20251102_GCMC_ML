# -*- coding: utf-8 -*-
"""
MOF_GCMC_MODEL.py
────────────────────────────────────────────
Tree-based model trainer for MOF-GCMC regression tasks.

Supports:
- RandomForestRegressor
- GradientBoostingRegressor
- CatBoostRegressor

Scaling should be provided externally (e.g., StandardScaler, MinMaxScaler).
"""

import os
import json
import time
import logging
from tqdm import tqdm
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# Optional CatBoost
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


class MOFModelTrainer:
    def __init__(self,
                 model_type: str = "rf",
                 model_params: Optional[Dict] = None,
                 scaler_X=None,
                 scaler_y=None,
                 outdir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Parameters
        ----------
        model_type : {"rf", "gbm", "cat"}
            Model type to use.
        model_params : dict, optional
            Hyperparameters for model.
        scaler_X, scaler_y : sklearn scaler, optional
            Pre-fitted or externally provided scalers.
        outdir : str, optional
            Path to save outputs (metrics, feature importances, etc.)
        logger : logging.Logger, optional
            Logging handler.
        """
        self.model_type = model_type.lower()
        self.params = model_params or {}
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.outdir = outdir
        self.logger = logger or self._make_logger()
        self.model = self._init_model()

    # ───────────────────────────────────────────────
    def _make_logger(self):
        logger = logging.getLogger("MOFModelTrainer")
        if not logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            h = logging.StreamHandler()
            h.setFormatter(fmt)
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger

    # ───────────────────────────────────────────────
    def _init_model(self):
        kind = self.model_type
        p = self.params

        if kind == "rf":
            base = dict(n_estimators=800, n_jobs=-1, random_state=42)
            base.update(p)
            return RandomForestRegressor(**base)
        elif kind == "gbm":
            base = dict(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
            base.update(p)
            return GradientBoostingRegressor(**base)
        elif kind == "cat":
            if not _HAS_CATBOOST:
                raise ImportError("❌ CatBoost not installed. Run: pip install catboost")
            base = dict(iterations=500, depth=8, learning_rate=0.05,
                        loss_function="RMSE", random_seed=42, verbose=False)
            base.update(p)
            return CatBoostRegressor(**base)
        else:
            raise ValueError(f"Unsupported model type '{kind}'")

    # ───────────────────────────────────────────────
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit model with optional scaling."""
        X_proc = self.scaler_X.transform(X_train) if self.scaler_X else X_train
        y_proc = self.scaler_y.transform(y_train.values.reshape(-1, 1)).ravel() if self.scaler_y else y_train

        self.logger.info(f"[TRAIN] Model={self.model_type.upper()} | Samples={len(X_train)} | Features={X_train.shape[1]}")
        t0 = time.time()
        self.model.fit(X_proc, y_proc)
        self.train_time = time.time() - t0
        self.logger.info(f"[TIME] Training took {self.train_time:.2f} s")

    # ───────────────────────────────────────────────
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict values with inverse scaling if needed."""
        X_proc = self.scaler_X.transform(X_test) if self.scaler_X else X_test
        y_pred = self.model.predict(X_proc)
        if self.scaler_y:
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred

    # ───────────────────────────────────────────────
    def evaluate(self, X_test: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """Compute regression metrics."""
        y_pred = self.predict(X_test)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))) * 100.0

        metrics = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE(%)": mape}
        self.logger.info(f"[EVAL] R2={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")

        if self.outdir:
            os.makedirs(self.outdir, exist_ok=True)
            with open(os.path.join(self.outdir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    # ───────────────────────────────────────────────
    def feature_importance(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Extract feature importance or permutation importance."""
        if hasattr(self.model, "feature_importances_"):
            imp = pd.Series(self.model.feature_importances_, index=X_test.columns)
        elif _HAS_CATBOOST and isinstance(self.model, CatBoostRegressor):
            imp = pd.Series(self.model.get_feature_importance(type="PredictionValuesChange"),
                            index=X_test.columns)
        else:
            self.logger.info("Using permutation importance (slow)...")
            r = permutation_importance(self.model, X_test, y_test, n_repeats=5, random_state=42)
            imp = pd.Series(r.importances_mean, index=X_test.columns)

        imp_df = imp.sort_values(ascending=False).reset_index()
        imp_df.columns = ["Feature", "Importance"]

        if self.outdir:
            imp_df.to_csv(os.path.join(self.outdir, "feature_importance.csv"), index=False, encoding="utf-8-sig")
        return imp_df

    # ───────────────────────────────────────────────
    def save_predictions(self, X_test: pd.DataFrame, y_true: pd.Series):
        y_pred = self.predict(X_test)
        pred_df = X_test.copy()
        pred_df["y_true"] = y_true.values
        pred_df["y_pred"] = y_pred
        if self.outdir:
            pred_path = os.path.join(self.outdir, "predictions.csv")
            pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
            self.logger.info(f"[SAVE] predictions → {pred_path}")
        return pred_df
        # ───────────────────────────────────────────────
    def save_predictions_full(self,
                              X_full: pd.DataFrame,
                              y_full: pd.Series,
                              train_idx: Union[list, np.ndarray],
                              test_idx: Union[list, np.ndarray],
                              id_col: str = "filename"):
        """
        Save model predictions for the FULL dataset (train + test).

        Automatically excludes non-feature columns (like filename) 
        before applying scaler/model, then restores them afterward.
        """

        # --- Identify columns used in training ---
        trained_features = None
        if self.scaler_X and hasattr(self.scaler_X, "feature_names_in_"):
            trained_features = list(self.scaler_X.feature_names_in_)
        else:
            trained_features = list(X_full.columns.difference([id_col]))

        # --- Predict only on valid features ---
        X_pred_input = X_full[trained_features].copy()
        y_pred_full = self.predict(X_pred_input)

        # --- Construct result DataFrame ---
        df_out = X_full.copy()
        df_out["y_true"] = y_full.values
        df_out["y_pred"] = y_pred_full

        # --- Split info ---
        df_out["Split"] = "Unknown"
        df_out.loc[train_idx, "Split"] = "Train"
        df_out.loc[test_idx, "Split"] = "Test"

        # --- Ensure id_col is first ---
        if id_col in df_out.columns:
            cols = [id_col] + [c for c in df_out.columns if c != id_col]
            df_out = df_out[cols]
        else:
            self.logger.warning(f"[WARN] id_col '{id_col}' not found in X_full")

        # --- Save file ---
        if self.outdir:
            os.makedirs(self.outdir, exist_ok=True)
            out_path = os.path.join(self.outdir, "predictions_full.csv")
            df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
            self.logger.info(f"[SAVE] full predictions → {out_path}")

        return df_out

