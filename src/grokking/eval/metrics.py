"""Evaluation metrics.

Implements accuracy and loss computation for modular addition task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    p: int,
) -> float:
    """Compute accuracy for modular addition predictions.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size].
        targets: Target labels [batch].
        p: Prime modulus.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    # Get logits at final position, restricted to number tokens
    logits_final = logits[:, -1, :p]  # [batch, p]

    # Compute predictions
    preds = logits_final.argmax(dim=-1)

    # Compute accuracy
    correct = (preds == targets).float().sum()
    total = targets.size(0)

    return (correct / total).item()


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    p: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute cross-entropy loss for modular addition.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size].
        targets: Target labels [batch].
        p: Prime modulus.
        reduction: Loss reduction method ("mean", "sum", or "none").

    Returns:
        Loss tensor.
    """
    # Get logits at final position, restricted to number tokens
    logits_final = logits[:, -1, :p]  # [batch, p]

    return F.cross_entropy(logits_final, targets, reduction=reduction)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    p: int,
) -> dict[str, float]:
    """Evaluate model on a batch of data.

    Args:
        model: The transformer model.
        tokens: Input tokens [batch, seq_len].
        targets: Target labels [batch].
        p: Prime modulus.

    Returns:
        Dictionary with 'loss' and 'accuracy' keys.
    """
    model.eval()
    logits = model(tokens)

    loss = compute_loss(logits, targets, p).item()
    acc = compute_accuracy(logits, targets, p)

    return {"loss": loss, "accuracy": acc}
