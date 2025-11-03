import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ───────────────────────────────────────────────
# CSV 읽기: 상단 두 줄은 header 정보이므로 skiprows=2
df = pd.read_csv("summary_stats.csv", skiprows=2)

# 기본 구조 확인
print("✅ Loaded DataFrame:", df.shape)
print(df.head(3))

# ───────────────────────────────────────────────
# 타입 변환
df["train_ratio"] = df["train_ratio"].astype(float)
df["qt_frac"] = df["qt_frac"].astype(float)

# ───────────────────────────────────────────────
# Metric 컬럼 이름 정규화
rename_map = {
    "MAPE(%)": "MAPE(%)_mean" if "MAPE(%)" in df.columns else None
}
df.rename(columns=rename_map, inplace=True)

metrics = ["R2_mean", "MAE_mean", "RMSE_mean", "MAPE(%)_mean"]
metric_labels = ["R²", "MAE", "RMSE", "MAPE (%)"]
colors = ["#4C72B0", "#55A868", "#C44E52"]

plt.style.use("seaborn-v0_8-whitegrid")

# ───────────────────────────────────────────────
# Plot 루프
for metric, label in zip(metrics, metric_labels):
    if metric not in df.columns:
        continue
    plt.figure(figsize=(8,5))

    train_ratios = sorted(df["train_ratio"].unique())
    x = np.arange(len(train_ratios))
    width = 0.25
    qt_groups = [0.0, 0.5, 1.0]

    for i, qt in enumerate(qt_groups):
        values = []
        for tr in train_ratios:
            sub = df[df["train_ratio"] == tr]
            if sub.empty:
                values.append(np.nan)
                continue
            closest = sub.iloc[(sub["qt_frac"] - qt).abs().argsort()[:1]]
            values.append(closest[metric].values[0] if not closest.empty else np.nan)
        plt.bar(x + i*width - width, values, width, label=f"qt_frac={qt:.2f}", color=colors[i])

    plt.xticks(x, [f"{tr:.2f}" for tr in train_ratios])
    plt.xlabel("Train Ratio")
    plt.ylabel(label)
    plt.title(f"{label} vs Train Ratio (qt_frac = 0 / mid / full)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_{metric.replace('(', '').replace(')', '').replace('%', 'pct')}.png", dpi=250)
    plt.close()

print("✅ All plots saved successfully.")

