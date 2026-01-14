#!/usr/bin/env python
"""Run a hyperparameter sweep.

Executes multiple training runs with different hyperparameter combinations.

Usage:
    python scripts/train_sweep.py --sweep configs/sweeps/weight_decay.yaml
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


def load_sweep_config(sweep_path: Path) -> dict:
    """Load a sweep configuration file.

    Args:
        sweep_path: Path to sweep YAML file.

    Returns:
        Sweep configuration dictionary.
    """
    with open(sweep_path) as f:
        return yaml.safe_load(f)


def generate_sweep_runs(sweep_config: dict) -> list[dict]:
    """Generate all run configurations from a sweep.

    Args:
        sweep_config: Sweep configuration with base_config, grid, and repeats.

    Returns:
        List of run configurations (each is a dict of overrides).
    """
    base_config = sweep_config["base_config"]
    grid = sweep_config.get("grid", {})
    repeats = sweep_config.get("repeats", 1)

    # Generate all combinations of grid parameters
    if grid:
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        combinations = list(itertools.product(*values))
    else:
        combinations = [()]
        keys = []

    runs = []
    for combo in combinations:
        for seed_offset in range(repeats):
            run_config = {
                "base_config": base_config,
                "overrides": {},
            }

            # Add grid parameter overrides
            for key, value in zip(keys, combo):
                run_config["overrides"][key] = value

            # Add seed offset for repeats
            if repeats > 1:
                run_config["seed_offset"] = seed_offset

            runs.append(run_config)

    return runs


def run_training(base_config: str, overrides: dict, seed_offset: int = 0) -> str:
    """Execute a single training run.

    Args:
        base_config: Path to base config file.
        overrides: Dictionary of parameter overrides.
        seed_offset: Offset to add to base seed for repeats.

    Returns:
        Path to run directory.
    """
    cmd = ["python", "scripts/train.py", "--config", base_config]

    for key, value in overrides.items():
        cmd.extend(["--set", f"{key}={value}"])

    if seed_offset > 0:
        # Add seed offset via override
        cmd.extend(["--set", f"run.seed_offset={seed_offset}"])

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Training failed")
        print(result.stderr)
        return None

    # Extract run directory from output (last line)
    lines = result.stdout.strip().split("\n")
    for line in reversed(lines):
        if line.startswith("RUN_DIR="):
            return line.split("=")[1]

    return None


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Path to sweep config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        print(f"ERROR: Sweep config not found: {sweep_path}")
        sys.exit(1)

    sweep_config = load_sweep_config(sweep_path)
    runs = generate_sweep_runs(sweep_config)

    print(f"Generated {len(runs)} runs from sweep config")

    if args.dry_run:
        for i, run in enumerate(runs):
            print(f"\nRun {i+1}:")
            print(f"  Base config: {run['base_config']}")
            print(f"  Overrides: {run['overrides']}")
            if "seed_offset" in run:
                print(f"  Seed offset: {run['seed_offset']}")
        return

    run_dirs = []
    for i, run in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"Starting run {i+1}/{len(runs)}")
        print(f"{'='*60}")

        seed_offset = run.get("seed_offset", 0)
        run_dir = run_training(run["base_config"], run["overrides"], seed_offset)

        if run_dir:
            run_dirs.append(run_dir)
            print(f"Completed: {run_dir}")
        else:
            print(f"Failed: run {i+1}")

    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(run_dirs)}/{len(runs)} runs successful")
    print(f"{'='*60}")

    for run_dir in run_dirs:
        print(f"  {run_dir}")


if __name__ == "__main__":
    main()
