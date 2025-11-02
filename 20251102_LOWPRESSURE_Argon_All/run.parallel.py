# -*- coding: utf-8 -*-
"""
Parallel Command Executor
────────────────────────────────────────────
Reads commands.txt and executes each in parallel (multiprocessing)
"""

import argparse, subprocess
from multiprocessing import Pool
from tqdm import tqdm

def run_command(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {cmd}")

def main(args):
    with open(args.commands, "r") as f:
        commands = [c.strip() for c in f if c.strip()]

    print(f"🚀 Executing {len(commands)} commands with {args.n_cpus} CPUs")

    with Pool(processes=args.n_cpus) as pool:
        list(tqdm(pool.imap_unordered(run_command, commands), total=len(commands)))

    print("🏁 All tasks completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", default="commands.txt")
    parser.add_argument("--n_cpus", type=int, default=8)
    args = parser.parse_args()
    main(args)
