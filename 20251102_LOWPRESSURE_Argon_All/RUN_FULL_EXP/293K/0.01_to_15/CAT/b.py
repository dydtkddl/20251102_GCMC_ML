import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ───────────────────────────────────────────────
# CSV 읽기 (첫 2줄 스킵)
df = pd.read_csv("summary_stats.csv", skiprows=2)

# 수동 컬럼명 복구
df.columns = [
    "train_ratio", "qt_frac",
    "R2_mean", "R2_std",
    "MAE_mean", "MAE_std",
    "RMSE_mean", "RMSE_std",
    "MAPE(%)_mean", "MAPE(%)_std"
]

df["train_ratio"] = df["train_ratio"].astype(float)
df["qt_frac"] = df["qt_frac"].astype(float)

# ───────────────────────────────────────────────
metrics = ["R2_mean", "MAE_mean", "RMSE_mean", "MAPE(%)_mean"]
metric_labels = ["R²", "MAE", "RMSE", "MAPE (%)"]
colors = ["#4C72B0", "#55A868", "#C44E52"]

plt.style.use("seaborn-v0_8-whitegrid")

# ───────────────────────────────────────────────
for metric, label in zip(metrics, metric_labels):
    plt.figure(figsize=(8,5))

    train_ratios = sorted(df["train_ratio"].unique())
    x = np.arange(len(train_ratios))
    width = 0.25
    qt_groups = [0.0, 0.5, 1.0]

    for i, qt in enumerate(qt_groups):
        values = []
        for tr in train_ratios:
            sub = df[df["train_ratio"] == tr]
            closest = sub.iloc[(sub["qt_frac"] - qt).abs().argsort()[:1]]
            values.append(closest[metric].values[0] if not closest.empty else np.nan)
        plt.bar(x + i*width - width, values, width, label=f"qt_frac={qt:.2f}", color=colors[i])
    print(metric)
    plt.xticks(x, [f"{tr:.2f}" for tr in train_ratios])
    plt.xlabel("Train Ratio")
    plt.ylabel(label)
    plt.title(f"{label} vs Train Ratio (qt_frac = 0 / mid / full)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_{metric.replace('(', '').replace(')', '').replace('%', 'pct')}.png", dpi=250)
    plt.close()

print("✅ All plots saved successfully → plot_R2_mean.png ... plot_MAPEpct_mean.png")

