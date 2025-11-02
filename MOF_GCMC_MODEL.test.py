# -*- coding: utf-8 -*-
"""
Full GCMC ML Experiment Pipeline
────────────────────────────────────────────
Multiple Datasets (273K, 293K, 313K)
Multiple lowp/output feature combinations
Multiple model types (rf, gbm, cat)
qt_frac sweep: 0.0 → 0.8 (0.1 step)
Train ratio fixed (default 0.5)

Output hierarchy:
RUN_FULL_EXP/TEMP/LOWP_to_OUTPUT/MODEL/qtfrac_*/...
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from MOF_GCMC_DATALOADER import load_mof_dataset
from MOF_GCMC_SAMPLER import GCMCSampler
from MOF_GCMC_MODEL import MOFModelTrainer


# ───────────────────────────────
# Main experiment routine
# ───────────────────────────────
def run_full_experiment(
    temps=["273K", "293K", "313K"],
    lowp_list=["HENRY", "0.01", "0.05", "0.1", "0.5"],
    output_list=["1", "5", "15"],
    models=["rf", "gbm", "cat"],
    qt_fracs=np.arange(0.0, 0.81, 0.1),
    train_ratio=0.5,
    seed_base=2025,
    out_root="./RUN_FULL_EXP"
):
    os.makedirs(out_root, exist_ok=True)

    for T in temps:
        data_path = f"./{T}_merged_dataset.exclude.broken_cif.csv"
        if not os.path.exists(data_path):
            print(f"⚠️ Skipping {T} (file not found: {data_path})")
            continue

        print(f"\n📂 Processing dataset: {T}")

        for lowp in lowp_list:
            for outp in output_list:
                combo_name = f"{lowp}_to_{outp}"
                combo_dir = os.path.join(out_root, T, combo_name)
                os.makedirs(combo_dir, exist_ok=True)

                print(f"\n🧩 Combination: {combo_name}")

                # ─── Dataset load ───
                df, meta = load_mof_dataset(
                    csv_path=data_path,
                    input_features=[
                        "LCD","PLD","LFPD","cm3_g","ASA_m2_cm3","ASA_m2_g",
                        "NASA_m2_cm3","NASA_m2_g","AV_VF","AV_cm3_g","NAV_cm3_g","Has_OMS"
                    ],
                    lowp_features=[lowp],
                    output_features=[outp]
                )
                target_col = meta["output_features"][0]
                id_col = meta["meta_columns"][0]

                for model_type in models:
                    model_dir = os.path.join(combo_dir, model_type.upper())
                    os.makedirs(model_dir, exist_ok=True)
                    results = []

                    print(f"🔧 Model: {model_type.upper()}")

                    for qt_frac in qt_fracs:
                        print(f"    ▶ train_ratio={train_ratio:.2f}, qt_frac={qt_frac:.2f}")

                        # ─── Sampler ───
                        sampler = GCMCSampler(
                            sampler_type="qt_then_rd",
                            qt_col=lowp,
                            use_log=True,
                            n_bins=125,
                            qt_frac=qt_frac,
                            train_ratio=train_ratio,
                            gamma=0.3,
                            seed_base=seed_base,
                            outdir=None
                        )
                        result = sampler.fit(df)
                        train_idx, test_idx = result["train_idx"], result["test_idx"]
                        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

                        # ─── Feature split ───
                        X_train = df_train.drop(columns=[id_col, target_col])
                        y_train = df_train[target_col]
                        X_test = df_test.drop(columns=[id_col, target_col])
                        y_test = df_test[target_col]

                        # ─── Scaling ───
                        scaler_X = StandardScaler().fit(X_train)
                        scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))

                        # ─── Model params ───
                        if model_type == "rf":
                            params = {"n_estimators": 1000, "max_depth": None, "n_jobs": -1}
                        elif model_type == "gbm":
                            params = {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 5}
                        elif model_type == "cat":
                            params = {
                                "iterations": 1000, "depth": 8, "learning_rate": 0.05,
                                "loss_function": "RMSE", "random_seed": seed_base, "verbose": False
                            }
                        else:
                            raise ValueError(f"Unsupported model: {model_type}")

                        # ─── Trainer ───
                        run_dir = os.path.join(model_dir, f"qtfrac_{qt_frac:.2f}".replace(".", "_"))
                        os.makedirs(run_dir, exist_ok=True)

                        trainer = MOFModelTrainer(
                            model_type=model_type,
                            model_params=params,
                            scaler_X=scaler_X,
                            scaler_y=scaler_y,
                            outdir=run_dir
                        )

                        trainer.fit(X_train, y_train)
                        metrics = trainer.evaluate(X_test, y_test)
                        trainer.feature_importance(X_test, y_test)
                        trainer.save_predictions(X_test, y_test)

                        results.append({
                            "temperature": T,
                            "lowp": lowp,
                            "output": outp,
                            "model": model_type,
                            "train_ratio": train_ratio,
                            "qt_frac": qt_frac,
                            "train_size": len(X_train),
                            "test_size": len(X_test),
                            **metrics
                        })

                    # ─── Save summary ───
                    df_result = pd.DataFrame(results)
                    df_result.to_csv(os.path.join(model_dir, "qtfrac_summary.csv"), index=False, encoding="utf-8-sig")
                    print(f"✅ Saved summary → {model_dir}/qtfrac_summary.csv")

    print("\n🎯 All datasets + models + feature combos processed successfully!")


# ─────────────────────────────
# Entry point
# ─────────────────────────────
if __name__ == "__main__":
    run_full_experiment()
