# -*- coding: utf-8 -*-
"""
engine.single.case.qt_from_prediction.py
────────────────────────────────────────────
Use predicted mid-pressure (e.g., 1bar y_pred)
as a quantile sampling feature for re-training
a new model (e.g., CAT, RF, GBM).
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from MOF_GCMC_DATALOADER import load_mof_dataset
from MOF_GCMC_SAMPLER import GCMCSampler
from MOF_GCMC_MODEL import MOFModelTrainer

# ───────────────────────────────────────────────
def setup_logger(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging started → {log_path}")


# ───────────────────────────────────────────────
def load_experiment_result(base_dir, temp, lowp, outp, model,
                           train_ratio, mode, qt_frac, seed):
    """Load metrics.json & predictions_full.csv"""
    combo = f"{lowp}_to_{outp}"
    path = os.path.join(
        base_dir,
        temp,
        combo,
        model.upper(),
        f"trainratio_{train_ratio:.2f}".replace(".", "_"),
        mode,
        f"qtfrac_{qt_frac:.2f}".replace(".", "_"),
        f"seed_{seed}"
    )

    metrics_path = os.path.join(path, "metrics.json")
    pred_path = os.path.join(path, "predictions_full.csv")

    if not os.path.exists(metrics_path) or not os.path.exists(pred_path):
        raise FileNotFoundError(f"❌ Missing required files in {path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    df_pred = pd.read_csv(pred_path)

    logging.info(f"✅ Loaded prior model results: {path}")
    logging.info(f"📈 R2={metrics['R2']:.4f} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f}")
    return metrics, df_pred


# ───────────────────────────────────────────────
def merge_with_base_dataset(df_pred, df_base, meta):
    """Merge predictions_full.csv with base GCMC dataset."""
    id_col = meta["meta_columns"][0]
    target_col = meta["output_features"][0]

    df_merged = pd.merge(df_pred, df_base, how="left", left_on="filename", right_on=id_col)

    drop_cols = ["y_true", "Split", id_col]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns], errors="ignore")

    logging.info(f"✅ Merged dataset shape: {df_merged.shape}")
    return df_merged


# ───────────────────────────────────────────────
def run_single_case(args):
    np.random.seed(args.seed)
    combo = f"{args.lowp}_to_{args.outp}"

    # ─── Output Directory ───
    out_dir = os.path.join(
        args.out_root,
        args.temp,
        combo,
        f"{args.target_model.upper()}_QTfrom_{args.source_model.upper()}",
        f"trainratio_{args.train_ratio:.2f}".replace(".", "_"),
        args.mode,
        f"qtfrac_{args.qt_frac:.2f}".replace(".", "_"),
        f"seed_{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)
    setup_logger(os.path.join(out_dir, "logs.txt"))

    logging.info(f"🚀 Starting QT-based transfer training: {args.target_model.upper()} from {args.source_model.upper()}")

    # ─── Load Source Experiment ───
    metrics_prev, df_pred = load_experiment_result(
        base_dir=args.out_root,
        temp=args.temp,
        lowp=args.lowp,
        outp=args.outp,
        model=args.source_model,
        train_ratio=args.train_ratio,
        mode=args.mode,
        qt_frac=args.qt_frac,
        seed=args.seed
    )

    # ─── Load Base Dataset ───
    data_path = f"../../00_GCMC/00_1st_collect/{args.temp}_merged_dataset.exclude.broken_cif.csv"
    df_base, meta = load_mof_dataset(
        csv_path=data_path,
        input_features=[
            "LCD", "PLD", "LFPD", "cm3_g", "ASA_m2_cm3", "ASA_m2_g",
            "NASA_m2_cm3", "NASA_m2_g", "AV_VF", "AV_cm3_g", "NAV_cm3_g", "Has_OMS"
        ],
        lowp_features=[args.lowp],
        output_features=[args.outp]
    )

    # ─── Merge with Base ───
    df_new = merge_with_base_dataset(df_pred, df_base, meta)

    # ─── Set qt_col = predicted 1bar (or selected pressure) ───
    qt_col = "y_pred"
    if qt_col not in df_new.columns:
        raise KeyError(f"❌ {qt_col} column not found in merged dataset")

    logging.info(f"📊 Using {qt_col} as quantile column for sampling")

    # ─── Run Quantile Sampler ───
    sampler = GCMCSampler(
        sampler_type="qt_then_rd",
        qt_col=qt_col,
        use_log=False,
        n_bins=100,
        qt_frac=args.qt_frac,
        train_ratio=args.train_ratio,
        gamma=0.5,
        seed_base=args.seed,
        outdir=out_dir
    )

    result = sampler.fit(df_new)
    sampler.summary(result, df_new)
    train_idx, test_idx = result["train_idx"], result["test_idx"]

    # ─── Split Dataset ───
    df_train = df_new.iloc[train_idx]
    df_test = df_new.iloc[test_idx]
    X_train = df_train.drop(columns=["y_pred"])
    y_train = df_train["y_pred"]
    X_test = df_test.drop(columns=["y_pred"])
    y_test = df_test["y_pred"]

    # ─── Scaling ───
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))

    # ─── Model Config ───
    params = dict(
        iterations=800,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=args.seed,
        verbose=False,
         thread_count=4

    )

    trainer = MOFModelTrainer(
        model_type=args.target_model,
        model_params=params,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        outdir=out_dir
    )

    # ─── Train & Evaluate ───
    trainer.fit(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.feature_importance(X_test, y_test)
    trainer.save_predictions(X_test, y_test)
    trainer.save_predictions_full(
        X_full=df_new.drop(columns=[]),
        y_full=df_new["y_pred"],
        train_idx=train_idx,
        test_idx=test_idx,
        id_col="filename"
    )

    # ─── Save metrics ───
    dfm = pd.DataFrame([{**vars(args), **metrics}])
    dfm.to_csv(os.path.join(out_dir, "metrics.csv"), index=False, encoding="utf-8-sig")

    logging.info(f"✅ Finished QT-based retraining. Results saved → {out_dir}")


# ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain model using predicted y_pred as quantile sampler input.")
    parser.add_argument("--temp", required=True)
    parser.add_argument("--lowp", required=True)
    parser.add_argument("--outp", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["struct", "struct_with_input"], required=True)
    parser.add_argument("--qt_frac", type=float, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--out_root", default="./RUN_FULL_EXP")

    parser.add_argument("--source_model", choices=["NN", "CAT", "RF", "GBM"], required=True,
                        help="Model used to generate predictions_full.csv")
    parser.add_argument("--target_model", choices=["CAT", "RF", "GBM"], default="CAT",
                        help="Model to train on sampled data")
    args = parser.parse_args()
    run_single_case(args)
