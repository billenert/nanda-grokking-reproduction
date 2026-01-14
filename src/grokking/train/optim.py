"""Optimizer and learning rate scheduler utilities.

Implements AdamW with configurable scheduler (constant or cosine with warmup).
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> AdamW:
    """Create AdamW optimizer from config.

    Args:
        model: Model to optimize.
        config: Full configuration dictionary with 'optim' section.

    Returns:
        Configured AdamW optimizer.
    """
    optim_config = config["optim"]

    optimizer = AdamW(
        model.parameters(),
        lr=optim_config["lr"],
        betas=tuple(optim_config["betas"]),
        eps=optim_config["eps"],
        weight_decay=optim_config["weight_decay"],
    )

    return optimizer


def create_scheduler(
    optimizer: AdamW,
    config: dict[str, Any],
    total_steps: int,
) -> LambdaLR:
    """Create learning rate scheduler from config.

    Supports:
        - "constant": Constant learning rate after warmup
        - "cosine": Cosine decay after warmup

    Args:
        optimizer: Optimizer to schedule.
        config: Full configuration dictionary with 'optim.scheduler' section.
        total_steps: Total number of training steps.

    Returns:
        Configured LambdaLR scheduler.
    """
    scheduler_config = config["optim"]["scheduler"]
    scheduler_name = scheduler_config["name"]
    warmup_steps = scheduler_config["warmup_steps"]

    if scheduler_name == "constant":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

    elif scheduler_name == "cosine":
        import math

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            # Cosine decay from 1 to 0 over remaining steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    else:
        raise ValueError(
            f"Invalid scheduler: {scheduler_name}. Must be 'constant' or 'cosine'."
        )

    return LambdaLR(optimizer, lr_lambda)


def get_lr(optimizer: AdamW) -> float:
    """Get current learning rate from optimizer.

    Args:
        optimizer: The optimizer.

    Returns:
        Current learning rate.
    """
    return optimizer.param_groups[0]["lr"]


def get_weight_decay(optimizer: AdamW) -> float:
    """Get weight decay from optimizer.

    Args:
        optimizer: The optimizer.

    Returns:
        Weight decay value.
    """
    return optimizer.param_groups[0]["weight_decay"]
