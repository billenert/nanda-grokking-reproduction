"""Transformer model components.

- Embeddings: normal(0, 0.02)
- Linear layers: normal(0, 0.02)
- Biases: zeros
- LayerNorm weights: ones, biases: zeros
"""

from typing import Any

import torch.nn as nn

from grokking.models.transformer import Transformer
from grokking.models.components import (
    MultiHeadAttention,
    MLP,
    TransformerBlock,
)


def init_weights(module: nn.Module) -> None:
    """Initialize weights according to the specification.

    Args:
        module: PyTorch module to initialize.
    """
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def create_model(config: dict[str, Any], vocab_size: int) -> Transformer:
    """Create a transformer model from config.

    Args:
        config: Full configuration dictionary.
        vocab_size: Vocabulary size.

    Returns:
        Initialized Transformer model.
    """
    return Transformer(config["model"], vocab_size)


__all__ = [
    "Transformer",
    "MultiHeadAttention",
    "MLP",
    "TransformerBlock",
    "init_weights",
    "create_model",
]
