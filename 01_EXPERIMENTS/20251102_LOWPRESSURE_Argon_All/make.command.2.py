import itertools

temps = ["293K"]
lowp_list = ["0.01"]
outp_list = ["5"]
seeds = [2025, 2026, 2027, 2028, 2029]
modes = ["struct", "struct_with_input"]
train_ratios = [i / 20 for i in range(1, 17)]  # 0.05 ~ 0.80 step 0.05

commands = []

for temp, lowp, outp, seed, mode, train_ratio in itertools.product(
    temps, lowp_list, outp_list, seeds, modes, train_ratios
):
    # struct인 경우 → qt_frac 고정 (0)
    if mode == "struct":
        qt_fracs = [0.0]
    else:
        # struct_with_input인 경우 → qt_frac 세 단계
        qt_fracs = [0.0, round(train_ratio / 2, 2), round(train_ratio, 2)]

    for qt_frac in qt_fracs:
        cmd = (
            f"python engine.single.case.py "
            f"--temp {temp} --lowp {lowp} --outp {outp} "
            f"--seed {seed} --mode {mode} "
            f"--train_ratio {train_ratio:.2f} --qt_frac {qt_frac:.2f}"
        )
        commands.append(cmd)

with open("commands.txt", "w") as f:
    f.write("\n".join(commands))

print(f"✅ Generated {len(commands)} commands → commands.txt")
