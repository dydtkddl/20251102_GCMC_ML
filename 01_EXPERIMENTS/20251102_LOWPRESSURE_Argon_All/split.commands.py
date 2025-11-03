# -*- coding: utf-8 -*-
"""
Command Splitter
────────────────────────────────────────────
Splits a commands.txt file into N smaller parts for parallel or distributed runs.
"""

import argparse
import os
from math import ceil

def split_commands(file_path: str, n_parts: int, out_dir: str):
    # Read commands
    with open(file_path, "r") as f:
        commands = [line.strip() for line in f if line.strip()]
    total = len(commands)
    if total == 0:
        raise ValueError("❌ commands.txt is empty.")

    # Make output directory
    os.makedirs(out_dir, exist_ok=True)

    # Split size
    chunk_size = ceil(total / n_parts)

    print(f"📦 Total {total} commands → splitting into {n_parts} parts (≈{chunk_size} per file)")
    for i in range(n_parts):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        part_cmds = commands[start:end]
        part_path = os.path.join(out_dir, f"commands_part_{i+1}.txt")
        with open(part_path, "w") as f:
            f.write("\n".join(part_cmds))
        print(f"✅ {part_path}: {len(part_cmds)} commands")

    print("🏁 Split completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split commands.txt into multiple parts")
    parser.add_argument("--commands", type=str, default="commands.txt", help="Path to commands.txt")
    parser.add_argument("--n_parts", type=int, required=True, help="Number of splits")
    parser.add_argument("--out_dir", type=str, default="./SPLIT_COMMANDS", help="Output directory for split files")
    args = parser.parse_args()

    split_commands(args.commands, args.n_parts, args.out_dir)

