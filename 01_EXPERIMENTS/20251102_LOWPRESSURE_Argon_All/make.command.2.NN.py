import itertools

temps = ["273K", "293K", "313K"]  # 원하는 온도 다 넣어도 됨
lowp_list = ["0.01"]
outp_list = ["1", "15"]  # 예: 다중 출력 실험도 고려
seeds = [2025, 2026, 2027, 2028, 2029]
modes = ["struct", "struct_with_input"]
train_ratios = [i / 20 for i in range(1, 17)]  # 0.05 ~ 0.80 step 0.05

commands = []

for temp, lowp, outp, seed, mode, train_ratio in itertools.product(
    temps, lowp_list, outp_list, seeds, modes, train_ratios
):
    # struct 모드 → qt_frac=0 고정
    if mode == "struct":
        qt_fracs = [0.0]
    else:
        # struct_with_input 모드 → qt_frac = 0, 절반, full
        qt_fracs = [
            float(f"{0.0:.2f}"),
            float(f"{train_ratio / 2:.2f}"),
            float(f"{train_ratio:.2f}")
        ]

    for qt_frac in qt_fracs:
        cmd = (
            f"python engine.single.case.NN.py "
            f"--temp {temp} --lowp {lowp} --outp {outp} "
            f"--seed {seed} --mode {mode} "
            f"--train_ratio {train_ratio:.2f} --qt_frac {qt_frac:.2f} "
            f"--model nn"
        )
        commands.append(cmd)

# 저장
with open("commands.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(commands))

print(f"✅ Generated {len(commands)} NN commands → commands.txt")
