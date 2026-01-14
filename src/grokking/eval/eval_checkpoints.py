"""Checkpoint evaluation utilities.

Evaluates all checkpoints in a run directory and generates summary.
"""

import json
from pathlib import Path
from typing import Any

import torch

from grokking.data import (
    ModularAdditionDataset,
    ModularAdditionTrainDataset,
    ModularAdditionTestDataset,
    get_vocab,
)
from grokking.models import Transformer
from grokking.train.checkpointing import get_checkpoint_steps, load_checkpoint_for_eval
from grokking.train.trainer import evaluate
from grokking.utils.io import load_config
from grokking.utils.device import get_device


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    train_dataset: ModularAdditionTrainDataset,
    test_dataset: ModularAdditionTestDataset,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        device: Device to run evaluation on.

    Returns:
        Dictionary with step, train_loss, train_acc, test_loss, test_acc.
    """
    p = config["data"]["p"]
    format_ = config["data"]["format"]
    vocab = get_vocab(p, format_)

    # Create model and load checkpoint
    model = Transformer(config["model"], vocab["vocab_size"])
    model.to(device)

    step, _ = load_checkpoint_for_eval(checkpoint_path, model)

    # Evaluate on train and test
    train_loss, train_acc = evaluate(model, train_dataset, config, device)
    test_loss, test_acc = evaluate(model, test_dataset, config, device)

    return {
        "step": step,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def evaluate_all_checkpoints(
    run_dir: str | Path,
    device_str: str = "cpu",
) -> list[dict[str, Any]]:
    """Evaluate all checkpoints in a run directory.

    Args:
        run_dir: Path to run directory.
        device_str: Device string ("cpu" or "cuda").

    Returns:
        List of evaluation results, sorted by step.
    """
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"

    # Load config
    config = load_config(run_dir / "config.yaml")
    device = get_device(device_str)

    # Create datasets
    p = config["data"]["p"]
    train_fraction = config["data"]["train_fraction"]
    seed = config["run"]["seed"]

    base_dataset = ModularAdditionDataset(p, train_fraction, seed)
    train_dataset = ModularAdditionTrainDataset(base_dataset)
    test_dataset = ModularAdditionTestDataset(base_dataset)

    # Get all checkpoint steps
    steps = get_checkpoint_steps(checkpoint_dir)

    results = []
    for step in steps:
        checkpoint_path = checkpoint_dir / f"step_{step}.pt"
        result = evaluate_checkpoint(
            checkpoint_path, config, train_dataset, test_dataset, device
        )
        results.append(result)
        print(f"Step {step}: train_acc={result['train_acc']:.4f}, test_acc={result['test_acc']:.4f}")

    return sorted(results, key=lambda x: x["step"])


def save_eval_summary(
    results: list[dict[str, Any]],
    run_dir: str | Path,
) -> Path:
    """Save evaluation summary to JSON file.

    Args:
        results: List of evaluation results.
        run_dir: Path to run directory.

    Returns:
        Path to saved summary file.
    """
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    summary_path = analysis_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"checkpoints": results}, f, indent=2)

    return summary_path
