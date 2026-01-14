"""Checkpointing utilities.

- Checkpoint format: step, model_state_dict, optim_state_dict, config, rng_state
- Also maintains a 'latest.pt' symlink/copy
"""

import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from grokking.utils.seed import get_rng_state, set_rng_state


def save_checkpoint(
    path: str | Path,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    config: dict[str, Any],
) -> None:
    """Save a training checkpoint.

    Args:
        path: Path to save checkpoint (e.g., 'checkpoints/step_1000.pt').
        step: Current training step.
        model: Model to save.
        optimizer: Optimizer to save.
        config: Full resolved configuration.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "config": config,
        "rng_state": get_rng_state(),
    }

    # Save to temporary file first, then rename for atomicity
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    shutil.move(str(tmp_path), str(path))


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    restore_rng: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        restore_rng: Whether to restore RNG state.

    Returns:
        Tuple of (step, config) from the checkpoint.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    if restore_rng and "rng_state" in checkpoint:
        set_rng_state(checkpoint["rng_state"])

    return checkpoint["step"], checkpoint["config"]


def load_checkpoint_for_eval(
    path: str | Path,
    model: nn.Module,
) -> tuple[int, dict[str, Any]]:
    """Load a checkpoint for evaluation only (no optimizer/RNG).

    Args:
        path: Path to checkpoint file.
        model: Model to load state into.

    Returns:
        Tuple of (step, config) from the checkpoint.
    """
    return load_checkpoint(path, model, optimizer=None, restore_rng=False)


def update_latest_checkpoint(
    checkpoint_dir: str | Path,
    checkpoint_name: str,
) -> None:
    """Update the 'latest.pt' symlink/copy to point to the most recent checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        checkpoint_name: Name of the checkpoint file (e.g., 'step_1000.pt').
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest_path = checkpoint_dir / "latest.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name

    # Remove existing latest.pt if it exists
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()

    # Try to create symlink, fall back to copy if not supported
    try:
        latest_path.symlink_to(checkpoint_name)
    except OSError:
        # Symlinks not supported (e.g., some Windows systems)
        shutil.copy2(checkpoint_path, latest_path)


def get_checkpoint_steps(checkpoint_dir: str | Path) -> list[int]:
    """Get list of checkpoint steps in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Sorted list of step numbers from checkpoint files.
    """
    checkpoint_dir = Path(checkpoint_dir)
    steps = []

    if not checkpoint_dir.exists():
        return steps

    for f in checkpoint_dir.iterdir():
        if f.name.startswith("step_") and f.name.endswith(".pt"):
            try:
                step = int(f.name[5:-3])  # Extract number from "step_XXX.pt"
                steps.append(step)
            except ValueError:
                continue

    return sorted(steps)


def get_latest_checkpoint_path(checkpoint_dir: str | Path) -> Path | None:
    """Get path to the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist.
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest_path = checkpoint_dir / "latest.pt"

    if latest_path.exists():
        # Resolve symlink if needed
        return latest_path.resolve()

    # Fall back to finding the highest step number
    steps = get_checkpoint_steps(checkpoint_dir)
    if steps:
        return checkpoint_dir / f"step_{max(steps)}.pt"

    return None
