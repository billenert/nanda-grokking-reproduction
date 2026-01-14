#!/usr/bin/env python
"""Run Fourier ablation experiments on model checkpoints.

Ablates specified frequency components from embeddings and measures
the impact on test accuracy.

Usage:
    python scripts/run_ablation.py --run_dir results/runs/<run_dir> --step 200000
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from grokking.data import ModularAdditionDataset
from grokking.data.tokenization import encode_example
from grokking.eval.metrics import compute_accuracy
from grokking.interp.ablations import run_ablation
from grokking.models import Transformer
from grokking.viz.plots import plot_ablation


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


def create_eval_fn(dataset, data_format: str, p: int, device: str):
    """Create evaluation function for ablation.

    Args:
        dataset: ModularAdditionDataset instance.
        data_format: Data format string.
        p: Prime modulus.
        device: Device to run evaluation on.

    Returns:
        Function that takes a model and returns test accuracy.
    """
    # Prepare test data
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

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            logits = model(test_tokens)
            acc = compute_accuracy(logits, test_targets, p)
        return acc

    return evaluate


def main():
    parser = argparse.ArgumentParser(description="Run Fourier ablation")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to ablate (default: latest)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for ablation",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # Load config
    config_path = run_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    p = config["data"]["p"]
    data_format = config["data"]["format"]
    train_fraction = config["data"]["train_fraction"]
    seed = config["run"]["seed"]

    # Find checkpoint
    ckpt_dir = run_dir / "checkpoints"
    if args.step is not None:
        ckpt_path = ckpt_dir / f"step_{args.step}.pt"
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            return
    else:
        # Use latest checkpoint
        ckpt_files = sorted(ckpt_dir.glob("step_*.pt"))
        if not ckpt_files:
            print(f"ERROR: No checkpoint files found in {ckpt_dir}")
            return
        ckpt_path = ckpt_files[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    model, step = load_checkpoint(ckpt_path, config, args.device)

    # Create dataset
    dataset = ModularAdditionDataset(
        p=p,
        train_fraction=train_fraction,
        seed=seed,
    )

    # Create evaluation function
    evaluate_fn = create_eval_fn(dataset, data_format, p, args.device)

    # Get ablation config
    analysis_config = config.get("analysis", {})
    ablation_config = analysis_config.get("ablation", {})
    freq_specs = ablation_config.get("freq_sets", [
        {"type": "topk", "k": 4},
        {"type": "single", "k": 1},
        {"type": "dc"},
    ])

    print(f"Running ablation with {len(freq_specs)} frequency sets...")

    # Run ablation
    acc_base, results = run_ablation(model, freq_specs, evaluate_fn, p)

    print(f"\nBaseline accuracy: {acc_base:.4f}")
    print("\nAblation results:")
    for r in results:
        drop = acc_base - r["acc"]
        print(f"  {r['set']}: {r['acc']:.4f} (drop: {drop:.4f})")

    # Save results
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    output = {
        "step": step,
        "acc_base": acc_base,
        "results": results,
    }

    output_path = analysis_dir / f"ablation_step_{step}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Generate plot
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_path = figures_dir / f"ablation_step_{step}.png"
    plot_ablation(output_path, plot_path)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
