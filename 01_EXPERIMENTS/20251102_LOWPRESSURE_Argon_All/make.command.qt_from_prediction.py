# -*- coding: utf-8 -*-
"""
make.command.qt_from_prediction.py
────────────────────────────────────────────
Generates command list for retraining 5bar predictions
using prior 1bar predictions as quantile-sampling input.
"""

import itertools

temps = [ "293K" ]          # 온도 리스트
lowp_list = ["0.01"]                      # 저압 feature
midp_list = ["1"]                         # 중간압 (기존 예측 모델)
outp_list = ["5"]                         # 최종 목표 압력
seeds = [2025, 2026, 2027, 2028, 2029]
modes = ["struct", "struct_with_input"]
train_ratios = [i / 20 for i in range(1, 17)]  # 0.05~0.80 step 0.05

# ───────────────────────────────────────────────
source_model = "CAT"   # or "NN"
target_model = "CAT"   # 재학습에 쓸 모델
script_name = "engine.single.case.qt_from_prediction.py"

commands = []

for temp, lowp, midp, outp, seed, mode, train_ratio in itertools.product(
    temps, lowp_list, midp_list, outp_list, seeds, modes, train_ratios
):
    # struct 모드 → qt_frac = 0 고정
    if mode == "struct":
        qt_fracs = [0.0]
    else:
        # struct_with_input 모드 → 0, 절반, full
        qt_fracs = [
            float(f"{0.0:.2f}"),
            float(f"{train_ratio / 2:.2f}"),
            float(f"{train_ratio:.2f}")
        ]

    for qt_frac in qt_fracs:
        cmd = (
            f"python {script_name} "
            f"--temp {temp} "
            f"--lowp {lowp} "
            f"--outp {outp} "
            f"--seed {seed} "
            f"--mode {mode} "
            f"--train_ratio {train_ratio:.2f} "
            f"--qt_frac {qt_frac:.2f} "
            f"--source_model {source_model} "
            f"--target_model {target_model}"
        )
        commands.append(cmd)

# ───────────────────────────────────────────────
# 저장
with open("commands.qt_from_pred.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(commands))

print(f"✅ Generated {len(commands)} retrain commands → commands.qt_from_pred.txt")
