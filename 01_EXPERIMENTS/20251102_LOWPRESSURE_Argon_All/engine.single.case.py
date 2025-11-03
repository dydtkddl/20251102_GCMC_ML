# -*- coding: utf-8 -*-
"""
Single GCMC ML Case Runner
────────────────────────────────────────────
Runs one configuration (T, lowp→outp, seed, sampling mode, qt_frac)
and saves results under a hierarchical directory.

Output structure:
RUN_FULL_EXP/
 └── 293K/
     └── 0.01_to_5/
         └── CAT/
             ├── struct/
             │   └── qtfrac_0_20/
             │       └── seed_2025/
             │           ├── metrics.csv
             │           ├── predictions.csv
             │           ├── predictions_full.csv
             │           ├── feature_importance.csv
             │           └── logs.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from MOF_GCMC_DATALOADER import load_mof_dataset
from MOF_GCMC_SAMPLER import GCMCSampler
from MOF_GCMC_MODEL import MOFModelTrainer


# ───────────────────────────────────────────────
def setup_logger(log_path: str):
    """File + console logging setup"""
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
def run_single_case(args):
    np.random.seed(args.seed)
    # ─── Hierarchical run directory ───
    combo = f"{args.lowp}_to_{args.outp}"
    run_dir = os.path.join(
        args.out_root,                              # ./RUN_FULL_EXP
        args.temp,                                  # 273K / 293K / 313K
        combo,                                      # e.g. HENRY_to_1
        args.model.upper(),                         # CAT / RF / GBM
        f"trainratio_{args.train_ratio:.2f}".replace(".", "_"),  # trainratio_0_50
        args.mode,                                  # struct / struct_with_input
        f"qtfrac_{args.qt_frac:.2f}".replace(".", "_"),           # qtfrac_0_25
        f"seed_{args.seed}"                         # seed_2025
    )
    os.makedirs(run_dir, exist_ok=True)
    setup_logger(os.path.join(run_dir, "logs.txt"))

    logging.info(f"🚀 Starting: {args.temp} | {combo} | mode={args.mode} | qt_frac={args.qt_frac:.2f} | seed={args.seed}")

    # ─── Load dataset ───
    data_path = f"../{args.temp}_merged_dataset.exclude.broken_cif.csv"
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found: {data_path}")
        return
# args.mode는 struct / lowr / lowq / lowqr50_50 / lowqr10_90 등등
    df, meta = load_mof_dataset(
        csv_path=data_path,
        input_features=[
            "LCD", "PLD", "LFPD", "cm3_g", "ASA_m2_cm3", "ASA_m2_g",
            "NASA_m2_cm3", "NASA_m2_g", "AV_VF", "AV_cm3_g", "NAV_cm3_g", "Has_OMS"
        ],
        lowp_features=[args.lowp] if args.mode != "struct" else [],
        output_features=[args.outp]
    )
    target_col = meta["output_features"][0]
    id_col = meta["meta_columns"][0]

    # ─── Sampler selection ───
    if args.mode == "struct_with_input":
        sampler_type = "qt_then_rd"
        qt_col = args.lowp
    else:
        sampler_type = "random_struct"
        qt_col = None

    sampler = GCMCSampler(
        sampler_type=sampler_type,
        qt_col=qt_col,
        use_log=True,
        n_bins=100,
        qt_frac=args.qt_frac,
        train_ratio=args.train_ratio,
        gamma=0.5,
        seed_base=args.seed,
        outdir=run_dir

    )
    
    result = sampler.fit(df)
    sampler.summary(result, df)

    train_idx, test_idx = result["train_idx"], result["test_idx"]

    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    X_train = df_train.drop(columns=[id_col, target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[id_col, target_col])
    y_test = df_test[target_col]

    # ─── Scaling ───
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))

    # ─── Model parameters ───
    params = {
        "iterations": 1000,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": args.seed,
        "verbose": False,
         "thread_count": 4
    }

    trainer = MOFModelTrainer(
        model_type=args.model,
        model_params=params,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        outdir=run_dir
    )

    # ─── Train + Evaluate ───
    trainer.fit(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.feature_importance(X_test, y_test)

    # ─── Save predictions ───
    trainer.save_predictions(X_test, y_test)
    trainer.save_predictions_full(
        X_full=df.drop(columns=[target_col]),
        y_full=df[target_col],
        train_idx=train_idx,
        test_idx=test_idx,
        id_col=id_col
    )

    # ─── Save summary ───
    dfm = pd.DataFrame([{**vars(args), **metrics}])
    dfm.to_csv(os.path.join(run_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    logging.info(f"✅ Finished case. Results saved at {run_dir}")


# ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", required=True, help="Temperature label (e.g., 273K)")
    parser.add_argument("--lowp", required=True, help="Low-pressure input feature")
    parser.add_argument("--outp", required=True, help="Target output pressure feature")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["struct", "struct_with_input"], required=True)
    parser.add_argument("--qt_frac", type=float, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--out_root", default="./RUN_FULL_EXP")
    parser.add_argument("--model", default="cat", choices=["rf", "gbm", "cat"])
    args = parser.parse_args()
    run_single_case(args)
