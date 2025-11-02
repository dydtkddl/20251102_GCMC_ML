import itertools

temps = ["273K", "293K", "313K"]
lowp_list = ["HENRY", "0.01", "0.05", "0.1", "0.5"]
outp_list = ["1", "5", "15"]
seeds = [2025, 2026, 2027, 2028, 2029]
modes = ["struct", "lowr", "lowq"]
qt_fracs = [i / 20 for i in range(1, 17)]  # 0.05 ~ 0.80 step 0.05
commands = []

# 👇 mode를 가장 빠르게 변화시키기 위해 순서 재배열
for temp, lowp, outp, seed, qt, mode in itertools.product(
    temps, lowp_list, outp_list, seeds, qt_fracs, modes
):
    cmd = (
        f"python engine_single_case.py "
        f"--temp {temp} --lowp {lowp} --outp {outp} "
        f"--seed {seed} --mode {mode} --qt_frac {qt:.2f}"
    )
    commands.append(cmd)

with open("commands.txt", "w") as f:
    f.write("\n".join(commands))

print(f"✅ Generated {len(commands)} commands → commands.txt")
