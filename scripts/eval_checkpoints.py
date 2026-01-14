#!/usr/bin/env python
"""Evaluate all checkpoints in a run directory.

Usage:
    python scripts/eval_checkpoints.py --run_dir results/runs/XXXX
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grokking.eval import evaluate_all_checkpoints, save_eval_summary
from grokking.viz import plot_grokking_curve


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for evaluation (cpu or cuda)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_dir = Path(args.run_dir)

    print(f"Evaluating checkpoints in {run_dir}")

    # Evaluate all checkpoints
    results = evaluate_all_checkpoints(run_dir, args.device)

    # Save summary
    summary_path = save_eval_summary(results, run_dir)
    print(f"\nSaved evaluation summary to {summary_path}")

    # Generate grokking curve plot from metrics.jsonl
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        plot_path = run_dir / "figures" / "grokking_curve.png"
        plot_grokking_curve(metrics_path, plot_path)
        print(f"Saved grokking curve plot to {plot_path}")


if __name__ == "__main__":
    main()
