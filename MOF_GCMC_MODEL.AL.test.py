# -*- coding: utf-8 -*-
"""
Active Learning Test with Full Monitoring & CONFIG Summary
────────────────────────────────────────────
Dataset : 313K_merged_dataset.exclude.broken_cif.csv
Target  : 0.01bar → 15bar uptake mapping
Sampling: initial_ratio=0.01, target_ratio=0.8
Model   : Residual MLP (ActiveTrainer)
Output  : ./RUN_AL_313K_001to15/
"""
import json
import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from MOF_GCMC_DATALOADER import load_mof_dataset
from MOF_GCMC_SAMPLER_AL import ActiveSampler
from MOF_GCMC_MODEL_AL import ActiveTrainer



# ════════════════════════════════════════════════════════
# 💡 Global Config
# ════════════════════════════════════════════════════════
CONFIG = {
    "T": 313,
    "LOWP": "0.01",
    "HIGHP": "15",
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
    "VAL_SPLIT": 0.1,
    "LR": 1e-3,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

OUT_DIR = f"{CONFIG['OUT_BASE']}{CONFIG['T']}K_{CONFIG['LOWP']}to{CONFIG['HIGHP']}/"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(CONFIG["DATA_DIR"], f"{CONFIG['T']}K_merged_dataset.exclude.broken_cif.csv")

# 저장해두기
with open(os.path.join(OUT_DIR, "CONFIG_used.json"), "w") as f:
    json.dump(CONFIG, f, indent=4)

# ════════════════════════════════════════════════════════
# Utility: Config Summary Printer
# ════════════════════════════════════════════════════════
def print_config_summary(config: dict):
    print("\n" + "=" * 70)
    print("🚀 ACTIVE LEARNING EXPERIMENT CONFIGURATION SUMMARY")
    print("=" * 70)
    for k, v in config.items():
        print(f"{k:<20}: {v}")
    print("=" * 70)
    print(f"💾 Output directory: {OUT_DIR}")
    print(f"📁 Dataset path: {DATA_PATH}")
    print(f"🧠 Device: {CONFIG['DEVICE']}")
    print("=" * 70 + "\n")

print_config_summary(CONFIG)

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
low_col = meta["lowp_features"][0]

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
print(f"🎯 Initial labeled={len(idx_labeled)}, unlabeled={len(idx_unlabeled)}\n")

act = get_activation(CONFIG["ACTIVATION"])
model = GCMCModel(X_scaled.shape[1], CONFIG["HIDDENDIM1"], CONFIG["HIDDENDIM2"],
                  CONFIG["DROPOUT_RATE"], act)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
loss_fn = nn.MSELoss()
trainer = ActiveTrainer(model, optimizer, loss_fn,
                        epochs=CONFIG["EPOCHS"], patience=CONFIG["PATIENCE"],
                        device=CONFIG["DEVICE"], outdir=OUT_DIR)


# ════════════════════════════════════════════════════════
# Active Learning Loop (Full Monitoring)
# ════════════════════════════════════════════════════════
metrics_log = []
n_target = int(CONFIG["TARGET_RATIO"] * len(X_scaled))
max_iters = (n_target - len(idx_labeled)) // CONFIG["SAMPLES_PER_ITER"] + 1

import sys, time

for it in range(max_iters):
    iter_start = time.time()  # 🔹 시간 측정 시작
    tqdm.write(f"\n🌀 Iter {it:02d} | labeled={len(idx_labeled)} / {n_target}")

    # ── Split Train/Validation ────────────────────────────
    val_size = max(1, int(CONFIG["VAL_SPLIT"] * len(idx_labeled)))
    idx_val = np.random.choice(idx_labeled, val_size, replace=False)
    idx_train = np.setdiff1d(idx_labeled, idx_val)

    d_train = GCMCDataset(X_scaled[idx_train], Y_log[idx_train], LOW_log[idx_train])
    d_val = GCMCDataset(X_scaled[idx_val], Y_log[idx_val], LOW_log[idx_val])

    dl_train = DataLoader(d_train, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    dl_val = DataLoader(d_val, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    tqdm.write(f"📦 Train={len(idx_train)}, Val={len(idx_val)}, "
               f"Sampled={len(idx_labeled)}, Remain={len(idx_unlabeled)}")

    # ── Train with tqdm (epoch 로그 X) ─────────────────────
    pbar = tqdm(
        range(CONFIG["EPOCHS"]),
        desc=f"[Iter {it:02d}] Training",
        leave=True,
        dynamic_ncols=True,
        ascii=True,
        file=sys.stdout
    )

    best_val_loss = float("inf")
    patience_ctr = 0

    for epoch in pbar:
        model.train()
        losses = []
        for xb, yb, _ in dl_train:
            xb, yb = xb.to(CONFIG["DEVICE"]), yb.to(CONFIG["DEVICE"])
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = [
                loss_fn(model(xb.to(CONFIG["DEVICE"])), yb.to(CONFIG["DEVICE"])).item()
                for xb, yb, _ in dl_val
            ]
        val_loss = np.mean(val_losses)

        # tqdm 내부 갱신만 (화면 덮어쓰기)
        pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= CONFIG["PATIENCE"]:
            tqdm.write(f"⏸ Early stop at epoch {epoch}")
            break

    # ── Iteration Summary ─────────────────────────────
    tqdm.write(f"✅ Iter {it:02d} finished | best_val_loss={best_val_loss:.5f}")

    # ── MC Dropout Uncertainty ─────────────────────────────
    X_rem = torch.tensor(X_scaled[idx_unlabeled], dtype=torch.float32)
    y_mean, y_std = trainer.predict(X_rem, uncertainty=True, n_sim=20)
    unc_mean, unc_std = np.mean(y_std), np.std(y_std)

    # ── 샘플 선택 ──────────────────────────────────────────
    new_idx = sampler.sample_next(idx_unlabeled, LOW_log, y_std, CONFIG["SAMPLES_PER_ITER"])
    sel_unc_mean = np.mean(y_std[np.isin(idx_unlabeled, new_idx)])
    unc_zscore = (sel_unc_mean - unc_mean) / (unc_std + 1e-8)

    tqdm.write(f"📈 Uncertainty mean={unc_mean:.5f} ± {unc_std:.5f} | "
               f"sel_mean={sel_unc_mean:.5f} (z={unc_zscore:.2f})")

    # ── 성능 평가 ─────────────────────────────────────────
    y_pred = np.exp(np.clip(y_mean, -20, 20))
    y_true = Y_all[idx_unlabeled]
    perf = trainer.evaluate(y_true, y_pred)

    # ── 시간 측정 종료 ─────────────────────────────────────
    iter_end = time.time()
    elapsed_sec = iter_end - iter_start
    elapsed_min = elapsed_sec / 60

    tqdm.write(
        f"   🔍 R2={perf['r2']:.4f} | MAE={perf['mae']:.4f} | RMSE={perf['rmse']:.4f} | "
        f"⏱ Time={elapsed_sec:.2f}s ({elapsed_min:.2f} min)"
    )

    # ── 업데이트 및 로그 저장 ─────────────────────────────
    idx_labeled = np.unique(np.concatenate([idx_labeled, new_idx]))
    idx_unlabeled = np.setdiff1d(idx_unlabeled, new_idx)

    metrics_log.append({
        "iter": it,
        "train": len(idx_train),
        "val": len(idx_val),
        "sampled": len(new_idx),
        "remain": len(idx_unlabeled),
        "unc_mean": unc_mean,
        "unc_std": unc_std,
        "sel_unc_mean": sel_unc_mean,
        "unc_zscore": unc_zscore,
        "time_sec": elapsed_sec,
        "time_min": elapsed_min,
        **perf
    })
    pd.DataFrame(metrics_log).to_csv(os.path.join(OUT_DIR, f"AL_metrics_iter_{it:02d}.csv"), index=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(idx_labeled) >= n_target:
        break

# ════════════════════════════════════════════════════════
# Save Final Summary
# ════════════════════════════════════════════════════════
df_result = pd.DataFrame(metrics_log)
df_result.to_csv(os.path.join(OUT_DIR, "AL_metrics_final.csv"), index=False)
trainer.save_model()

print("\n" + "=" * 70)
print(f"✅ ACTIVE LEARNING EXPERIMENT COMPLETED")
print(f"📊 Total iterations: {len(metrics_log)}")
print(f"💾 Results saved in → {OUT_DIR}")
print("=" * 70 + "\n")




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

