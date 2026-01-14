#!/usr/bin/env python
"""Analyze progress measures over training checkpoints.

Computes progress measures (top-k mass, spectral alignment) and
plots them alongside test accuracy over training.

Usage:
    python scripts/analyze_progress.py --run_dir results/runs/<run_dir>
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from grokking.data import ModularAdditionDataset
from grokking.data.tokenization import encode_example
from grokking.eval.metrics import compute_accuracy
from grokking.interp.progress_measures import compute_all_progress_measures
from grokking.models import Transformer
from grokking.viz.plots import plot_progress


def load_checkpoint(ckpt_path: Path, config: dict, device: str = "cpu"):
    """Load a model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Tuple of (model, step).
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine vocab size from config
    p = config["data"]["p"]
    data_format = config["data"]["format"]
    if data_format == "BOS_A_B_SEP":
        vocab_size = p + 2
    else:
        vocab_size = p + 1

    # Create model
    model = Transformer(config["model"], vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["step"]


def compute_test_accuracy(model, dataset, data_format: str, p: int, device: str) -> float:
    """Compute test accuracy for a model.

    Args:
        model: Transformer model.
        dataset: ModularAdditionDataset instance.
        data_format: Data format string.
        p: Prime modulus.
        device: Device to run evaluation on.

    Returns:
        Test accuracy as a float.
    """
    test_pairs = dataset.test_pairs
    test_tokens = []
    test_targets = []

    for a, b in test_pairs:
        c = dataset.get_label(a, b)
        tokens, target = encode_example(a, b, p, data_format)
        test_tokens.append(tokens)
        test_targets.append(target)

    test_tokens = torch.stack(test_tokens).to(device)
    test_targets = torch.tensor(test_targets).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(test_tokens)
        acc = compute_accuracy(logits, test_targets, p)

    return acc


def main():
    parser = argparse.ArgumentParser(description="Analyze progress measures")
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
        help="Device to use for analysis",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="4,8",
        help="Comma-separated k values for top-k mass",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    k_values = [int(k) for k in args.k_values.split(",")]

    # Load config
    config_path = run_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    p = config["data"]["p"]
    data_format = config["data"]["format"]
    train_fraction = config["data"]["train_fraction"]
    seed = config["run"]["seed"]

    # Create dataset
    dataset = ModularAdditionDataset(
        p=p,
        train_fraction=train_fraction,
        seed=seed,
    )

    # Find checkpoints
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoints directory not found: {ckpt_dir}")
        return

    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpt_files:
        print(f"ERROR: No checkpoint files found in {ckpt_dir}")
        return

    analysis_config = config.get("analysis", {})
    checkpoints_spec = analysis_config.get("checkpoints", "all")

    # Filter checkpoints if needed
    if checkpoints_spec != "all" and isinstance(checkpoints_spec, list):
        ckpt_files = [
            f for f in ckpt_files
            if int(f.stem.split("_")[1]) in checkpoints_spec
        ]

    print(f"Analyzing {len(ckpt_files)} checkpoints...")

    # Analyze each checkpoint
    results = {"checkpoints": []}

    for ckpt_path in ckpt_files:
        model, step = load_checkpoint(ckpt_path, config, args.device)

        # Compute test accuracy
        acc_test = compute_test_accuracy(model, dataset, data_format, p, args.device)

        # Compute progress measures
        measures = compute_all_progress_measures(model, p, k_values=k_values)

        checkpoint_result = {
            "step": step,
            "acc_test": acc_test,
            **measures,
        }
        results["checkpoints"].append(checkpoint_result)

        print(f"  Step {step}: acc_test={acc_test:.4f}, spec_align={measures['spec_align']:.4f}")

    # Save results
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    output_path = analysis_dir / "progress_measures.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Generate plot
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_path = figures_dir / "progress_measures.png"
    plot_progress(output_path, plot_path)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
