import os
import json
import pandas as pd
from tqdm import tqdm

base_dir = "."  # 최상위 경로 지정
records = []

# trainratio_* → struct_with_input → qtfrac_* → seed_*
for train_dir in tqdm(sorted(os.listdir(base_dir))):
    if not train_dir.startswith("trainratio_"):
        continue
    train_ratio = float(train_dir.replace("trainratio_", "").replace("_", "."))

    struct_path = os.path.join(base_dir, train_dir, "struct_with_input")
    if not os.path.isdir(struct_path):
        continue

    for qt_dir in sorted(os.listdir(struct_path)):
        if not qt_dir.startswith("qtfrac_"):
            continue
        qt_frac = float(qt_dir.replace("qtfrac_", "").replace("_", "."))
        qt_path = os.path.join(struct_path, qt_dir)

        for seed_dir in sorted(os.listdir(qt_path)):
            if not seed_dir.startswith("seed_"):
                continue
            seed = int(seed_dir.replace("seed_", ""))
            metrics_path = os.path.join(qt_path, seed_dir, "metrics.json")

            if not os.path.exists(metrics_path):
                continue

            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                record = {
                    "train_ratio": train_ratio,
                    "qt_frac": qt_frac,
                    "seed": seed,
                }
                record.update(metrics)
                records.append(record)
            except Exception as e:
                print(f"⚠️ Error reading {metrics_path}: {e}")

# ── 통합 DataFrame 생성 ─────────────────────────────
if not records:
    print("❌ No metrics.json found.")
    exit()

df = pd.DataFrame(records)

# 숫자형 컬럼만 통계 계산
metric_cols = [
    c for c in df.columns
    if c not in ["train_ratio", "qt_frac", "seed"]
    and pd.api.types.is_numeric_dtype(df[c])
]

# ── raw 저장
raw_csv = os.path.join(base_dir, "summary_raw.csv")
df.to_csv(raw_csv, index=False, encoding="utf-8-sig")

# ── 평균/표준편차 집계
grouped = df.groupby(["train_ratio", "qt_frac"])[metric_cols].agg(["mean", "std"])
stats_csv = os.path.join(base_dir, "summary_stats.csv")
grouped.to_csv(stats_csv, encoding="utf-8-sig")

print(f"✅ Raw results saved → {raw_csv}")
print(f"✅ Summary stats saved → {stats_csv}")

print("\n📊 Example preview:")
print(grouped.head())

