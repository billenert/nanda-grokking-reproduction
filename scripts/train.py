#!/usr/bin/env python
"""Training script for grokking reproduction.

Usage:
    python scripts/train.py --config configs/canonical.yaml
    python scripts/train.py --config configs/debug.yaml --set run.seed=42
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grokking.utils.io import load_config, apply_overrides
from grokking.train.trainer import train


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train grokking model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="Override config values (e.g., --set optim.lr=0.001)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    # Run training
    run_dir = train(config)

    # Print run directory as final line (for scripts to capture)
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
