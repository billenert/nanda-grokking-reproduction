"""Main training loop.

- Training loop with uniform sampling with replacement
- Evaluation on full train and test splits
- Checkpointing and logging
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from grokking.data import (
    ModularAdditionDataset,
    ModularAdditionTrainDataset,
    ModularAdditionTestDataset,
    InfiniteSampler,
    create_batch,
    get_vocab,
)
from grokking.models import create_model
from grokking.train.optim import create_optimizer, create_scheduler, get_lr, get_weight_decay
from grokking.train.checkpointing import save_checkpoint, update_latest_checkpoint
from grokking.train.logging import MetricsLogger
from grokking.utils.seed import set_seed
from grokking.utils.device import get_device, get_dtype
from grokking.utils.io import save_config


def compute_loss_and_acc(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    p: int,
) -> tuple[torch.Tensor, float]:
    """Compute cross-entropy loss and accuracy.

    Args:
        model: The transformer model.
        tokens: Input tokens [batch, seq_len].
        targets: Target labels [batch].
        p: Prime modulus.

    Returns:
        Tuple of (loss, accuracy).
    """
    logits = model(tokens)  # [batch, seq_len, vocab_size]

    # Get logits at final position, restricted to number tokens
    logits_final = logits[:, -1, :p]  # [batch, p]

    # Cross-entropy loss
    loss = F.cross_entropy(logits_final, targets)

    # Accuracy
    preds = logits_final.argmax(dim=-1)
    acc = (preds == targets).float().mean().item()

    return loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: ModularAdditionTrainDataset | ModularAdditionTestDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: The transformer model.
        dataset: Dataset to evaluate on.
        config: Configuration dictionary.
        device: Device to run on.

    Returns:
        Tuple of (loss, accuracy).
    """
    model.eval()
    p = config["data"]["p"]
    format_ = config["data"]["format"]
    batch_size = config["train"]["batch_size"]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Evaluate in batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in dataloader:
        # DataLoader returns ((a_tensor, b_tensor), labels_tensor)
        (a_vals, b_vals), labels = batch
        # Convert to list of ((a, b), label) tuples
        examples = [
            ((a_vals[i].item(), b_vals[i].item()), labels[i].item())
            for i in range(len(labels))
        ]

        tokens, targets = create_batch(examples, p, format_)
        tokens = tokens.to(device)
        targets = targets.to(device)

        logits = model(tokens)
        logits_final = logits[:, -1, :p]

        loss = F.cross_entropy(logits_final, targets, reduction="sum")
        preds = logits_final.argmax(dim=-1)

        total_loss += loss.item()
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    model.train()
    return avg_loss, acc


def create_run_dir(config: dict[str, Any]) -> Path:
    """Create a run directory with timestamp.

    Args:
        config: Configuration dictionary.

    Returns:
        Path to the run directory.
    """
    out_dir = Path(config["run"]["out_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config["run"]["name"]
    seed = config["run"]["seed"]

    run_dir = out_dir / f"{timestamp}__{name}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)

    return run_dir


def train(config: dict[str, Any]) -> Path:
    """Run the full training loop.

    Args:
        config: Full configuration dictionary.

    Returns:
        Path to the run directory.
    """
    # Set seed for reproducibility
    seed = config["run"]["seed"]
    set_seed(seed)

    # Setup device
    device = get_device(config["run"]["device"])

    # Create run directory
    run_dir = create_run_dir(config)

    # Save config
    save_config(config, run_dir / "config.yaml")

    # Create dataset
    p = config["data"]["p"]
    train_fraction = config["data"]["train_fraction"]
    format_ = config["data"]["format"]

    base_dataset = ModularAdditionDataset(p, train_fraction, seed)
    train_dataset = ModularAdditionTrainDataset(base_dataset)
    test_dataset = ModularAdditionTestDataset(base_dataset)

    # Create model
    vocab = get_vocab(p, format_)
    vocab_size = vocab["vocab_size"]
    model = create_model(config, vocab_size)
    model.to(device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    total_steps = config["train"]["steps"]
    scheduler = create_scheduler(optimizer, config, total_steps)

    # Create data sampler for training
    sampler = InfiniteSampler(train_dataset, seed=seed)
    train_iter = iter(sampler)

    # Training config
    batch_size = config["train"]["batch_size"]
    eval_every = config["train"]["eval_every"]
    ckpt_every = config["train"]["ckpt_every"]
    log_every = config["train"]["log_every"]

    # Setup logging
    logger = MetricsLogger(run_dir / "metrics.jsonl")

    # Training loop
    model.train()
    pbar = tqdm(range(total_steps), desc="Training")

    for step in pbar:
        # Sample batch
        indices = [next(train_iter) for _ in range(batch_size)]
        examples = [train_dataset[i] for i in indices]

        # Convert to proper format for create_batch
        examples = [((a, b), label) for (a, b), label in examples]

        tokens, targets = create_batch(examples, p, format_)
        tokens = tokens.to(device)
        targets = targets.to(device)

        # Forward pass
        loss, acc = compute_loss_and_acc(model, tokens, targets, p)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if step % log_every == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

        # Evaluation
        if step % eval_every == 0 or step == total_steps - 1:
            train_loss, train_acc = evaluate(model, train_dataset, config, device)
            test_loss, test_acc = evaluate(model, test_dataset, config, device)

            lr = get_lr(optimizer)
            wd = get_weight_decay(optimizer)

            logger.log(step, "train", train_loss, train_acc, lr, wd)
            logger.log(step, "test", test_loss, test_acc, lr, wd)

            tqdm.write(
                f"Step {step}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
            )

        # Checkpointing
        if step % ckpt_every == 0 or step == total_steps - 1:
            ckpt_name = f"step_{step}.pt"
            ckpt_path = run_dir / "checkpoints" / ckpt_name
            save_checkpoint(ckpt_path, step, model, optimizer, config)
            update_latest_checkpoint(run_dir / "checkpoints", ckpt_name)

    logger.close()

    print(f"\nRUN_DIR={run_dir}")
    return run_dir
