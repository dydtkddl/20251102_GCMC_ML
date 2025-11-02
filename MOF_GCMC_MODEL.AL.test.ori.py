# -*- coding: utf-8 -*-
"""
Active Learning Test with load_mof_dataset (Monitoring + Uncertainty Stats)
────────────────────────────────────────────
Dataset : 313K_merged_dataset.exclude.broken_cif.csv
Target  : 0.01bar → 15bar uptake mapping
Sampling: initial_ratio=0.01, target_ratio=0.8
Model   : Residual MLP (ActiveTrainer)
Output  : ./RUN_AL_313K_001to15/
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from MOF_GCMC_DATALOADER import load_mof_dataset
from MOF_GCMC_MODEL_AL import ActiveSampler
from MOF_GCMC_SAMPLER_AL import ActiveTrainer


# ════════════════════════════════════════════════════════
# 💡 Global Config
# ════════════════════════════════════════════════════════
CONFIG = {
    "T": 313,
    "LOWP": 0.01,
    "HIGHP": 15,
    "DATA_DIR": "./",
    "OUT_BASE": "./RUN_AL_",

    # Active Learning control
    "INITIAL_RATIO": 0.01,
    "TARGET_RATIO": 0.8,
    "INITIAL_METHOD": "hybrid",  # "random" | "quantile" | "hybrid"
    "SAMPLES_PER_ITER": 30,
    "RD_FRAC": 0.0,
    "QT_FRAC": 0.2,
    "NUM_BINS": 70,
    "GAMMA": 0.3,
    "SEED": 2025,

    # Model / training
    "HIDDENDIM1": 128,
    "HIDDENDIM2": 128,
    "DROPOUT_RATE": 0.05,
    "ACTIVATION": "gelu",   # "relu", "silu", "gelu", "mish"
    "EPOCHS": 400,
    "PATIENCE": 30,
    "BATCH_SIZE": 64,
    "LR": 1e-3,
    "VAL_SPLIT": 0.1,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

OUT_DIR = f"{CONFIG['OUT_BASE']}{CONFIG['T']}K_{CONFIG['LOWP']}to{CONFIG['HIGHP']}/"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(CONFIG["DATA_DIR"], f"{CONFIG['T']}K_merged_dataset.exclude.broken_cif.csv")


# ════════════════════════════════════════════════════════
# Dataset + Model Definition
# ════════════════════════════════════════════════════════
import torch.nn as nn
import torch.optim as optim


class GCMCDataset(Dataset):
    def __init__(self, X, Y_log, LOW_log):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y_log, dtype=torch.float32).unsqueeze(1)
        self.LOW = torch.tensor(LOW_log, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.LOW[i]


def get_activation(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    elif name == "silu": return nn.SiLU()
    elif name == "gelu": return nn.GELU()
    elif name == "mish": return nn.Mish()
    else: raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, dim, drop, act):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), act, nn.Dropout(drop),
            nn.Linear(dim, dim), act, nn.Dropout(drop)
        )

    def forward(self, x):
        return x + self.block(x)


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


# ════════════════════════════════════════════════════════
# Load dataset
# ════════════════════════════════════════════════════════
df, meta = load_mof_dataset(
    csv_path=DATA_PATH,
    input_features=[
        "LCD","PLD","LFPD","cm3_g","ASA_m2_cm3","ASA_m2_g",
        "NASA_m2_cm3","NASA_m2_g","AV_VF","AV_cm3_g","NAV_cm3_g","Has_OMS"
    ],
    lowp_features=[CONFIG["LOWP"]],
    output_features=[CONFIG["HIGHP"]],
)

print(f"✅ Loaded dataset: {len(df)} rows ({CONFIG['LOWP']} → {CONFIG['HIGHP']})")

target_col = meta["output_features"][0]
id_col = meta["meta_columns"][0]
low_col = meta["lowp_features"][0]

# ─── 데이터 구성 ───────────────────────────────────────
X_all = df[meta["input_features"] + meta["lowp_features"]].values.astype(np.float32)
Y_all = df[target_col].values.astype(np.float32)
LOW_all = df[low_col].values.astype(np.float32)

LOW_log = np.log(np.clip(LOW_all, 1e-12, None))
Y_log = np.log(np.clip(Y_all, 1e-12, None))
X_all[:, -1] = LOW_log

scaler_X = StandardScaler().fit(X_all)
X_scaled = scaler_X.transform(X_all)
idx_all = np.arange(len(X_scaled))


# ════════════════════════════════════════════════════════
# Initialize Sampler + Trainer
# ════════════════════════════════════════════════════════
sampler = ActiveSampler(
    rd_frac=CONFIG["RD_FRAC"],
    qt_frac=CONFIG["QT_FRAC"],
    num_bins=CONFIG["NUM_BINS"],
    gamma=CONFIG["GAMMA"],
    seed=CONFIG["SEED"]
)

n_init = int(CONFIG["INITIAL_RATIO"] * len(X_scaled))
init_idx = sampler.initial_sampling(LOW_log, idx_all, total_size=n_init, method=CONFIG["INITIAL_METHOD"])
idx_labeled = init_idx
idx_unlabeled = np.setdiff1d(idx_all, idx_labeled)

print(f"🎯 Initial labeled={len(idx_labeled)}, unlabeled={len(idx_unlabeled)}")

act = get_activation(CONFIG["ACTIVATION"])
model = GCMCModel(X_scaled.shape[1], CONFIG["HIDDENDIM1"], CONFIG["HIDDENDIM2"],
                  CONFIG["DROPOUT_RATE"], act)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
loss_fn = nn.MSELoss()
trainer = ActiveTrainer(model, optimizer, loss_fn,
                        epochs=CONFIG["EPOCHS"], patience=CONFIG["PATIENCE"],
                        device=CONFIG["DEVICE"], outdir=OUT_DIR)


# ════════════════════════════════════════════════════════
# Active Learning Loop with Uncertainty Stats
# ════════════════════════════════════════════════════════
metrics_log = []
n_target = int(CONFIG["TARGET_RATIO"] * len(X_scaled))
max_iters = (n_target - len(idx_labeled)) // CONFIG["SAMPLES_PER_ITER"] + 1

for it in range(max_iters):
    print(f"\n🌀 Iter {it:02d} | labeled={len(idx_labeled)} / {n_target}")

    # ─── Train set 구성
    d_train = GCMCDataset(X_scaled[idx_labeled], Y_log[idx_labeled], LOW_log[idx_labeled])
    dl_train = DataLoader(d_train, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

    # ─── 학습 (tqdm으로 Epoch 모니터링)
    print(f"🧠 Training {len(idx_labeled)} samples ...")
    trainer.fit(dl_train, dl_train)

    # ─── Unlabeled set 예측 (MC Dropout)
    X_rem = torch.tensor(X_scaled[idx_unlabeled], dtype=torch.float32)
    y_mean, y_std = trainer.predict(X_rem, uncertainty=True, n_sim=20)

    # ─── 불확실도 통계 계산
    unc_mean = np.mean(y_std)
    unc_std = np.std(y_std)

    # 선택된 샘플 추출
    new_idx = sampler.sample_next(idx_unlabeled, LOW_log, y_std, CONFIG["SAMPLES_PER_ITER"])
    sel_unc_mean = np.mean(y_std[np.isin(idx_unlabeled, new_idx)])
    unc_zscore = (sel_unc_mean - unc_mean) / (unc_std + 1e-8)

    print(f"📈 Uncertainty stats → mean={unc_mean:.5f}, std={unc_std:.5f}, "
          f"selected_mean={sel_unc_mean:.5f}, zscore={unc_zscore:.2f}")

    # ─── 모델 성능 평가
    y_pred = np.exp(np.clip(y_mean, -20, 20))
    y_true = Y_all[idx_unlabeled]
    perf = trainer.evaluate(y_true, y_pred)
    print(f"   🔍 R2={perf['r2']:.4f} | MAE={perf['mae']:.4f} | RMSE={perf['rmse']:.4f}")

    # ─── index 업데이트
    idx_labeled = np.unique(np.concatenate([idx_labeled, new_idx]))
    idx_unlabeled = np.setdiff1d(idx_unlabeled, new_idx)

    # ─── iteration 로그 저장
    metrics_log.append({
        "iter": it,
        "n_labeled": len(idx_labeled),
        "n_unlabeled": len(idx_unlabeled),
        "unc_mean": unc_mean,
        "unc_std": unc_std,
        "sel_unc_mean": sel_unc_mean,
        "unc_zscore": unc_zscore,
        **perf
    })

    pd.DataFrame(metrics_log).to_csv(os.path.join(OUT_DIR, f"AL_metrics_iter_{it:02d}.csv"), index=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(idx_labeled) >= n_target:
        break


# ════════════════════════════════════════════════════════
# Save results
# ════════════════════════════════════════════════════════
df_result = pd.DataFrame(metrics_log)
df_result.to_csv(os.path.join(OUT_DIR, "AL_metrics_final.csv"), index=False)
trainer.save_model()
print(f"\n✅ Experiment completed → {OUT_DIR}")

