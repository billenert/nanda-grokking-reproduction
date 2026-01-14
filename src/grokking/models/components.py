"""Transformer model components.

- Multi-head self-attention with causal masking
- MLP with configurable activation
- Pre-norm architecture using LayerNorm
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    Implements causal attention where each position can only attend to
    previous positions. The causal mask is applied by setting forbidden
    positions to -inf prior to softmax.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability for attention weights.

        Raises:
            ValueError: If d_model is not divisible by n_heads.
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Register buffer for causal mask
        # Will be created lazily in forward pass
        self._causal_mask: Optional[torch.Tensor] = None

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal attention mask.

        Args:
            seq_len: Sequence length.
            device: Device to create mask on.

        Returns:
            Causal mask of shape [seq_len, seq_len] with -inf for forbidden positions.
        """
        if self._causal_mask is None or self._causal_mask.size(0) < seq_len:
            # Create mask where True indicates forbidden (future) positions
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1
            ).bool()
            # Convert to attention mask (-inf for forbidden positions)
            self._causal_mask = torch.where(
                mask,
                torch.tensor(float("-inf"), device=device),
                torch.tensor(0.0, device=device),
            )
        return self._causal_mask[:seq_len, :seq_len]

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (output, attention_weights) where:
                - output: [batch, seq_len, d_model]
                - attention_weights: [batch, n_heads, seq_len, seq_len] or None
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.W_q(x)  # [batch, seq, d_model]
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape to [batch, n_heads, seq, d_head]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        # [batch, n_heads, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        scores = scores + causal_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch, n_heads, seq, d_head]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None


class MLP(nn.Module):
    """Feed-forward MLP block.

    Two-layer MLP with configurable activation function.
    """

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """Initialize MLP.

        Args:
            d_model: Model dimension.
            d_mlp: Hidden layer dimension.
            activation: Activation function ("relu" or "gelu").
            dropout: Dropout probability.

        Raises:
            ValueError: If activation is not "relu" or "gelu".
        """
        super().__init__()

        if activation not in ("relu", "gelu"):
            raise ValueError(f"Invalid activation: {activation}. Must be 'relu' or 'gelu'.")

        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLP.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].

        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        x = self.W_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.W_out(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Implements: x = x + attn(ln1(x)), x = x + mlp(ln2(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        """Initialize transformer block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_mlp: MLP hidden dimension.
            activation: MLP activation function.
            dropout: Dropout probability.
            layer_norm_eps: Epsilon for layer normalization.
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = MLP(d_model, d_mlp, activation, dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for transformer block.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (output, attention_weights).
        """
        # Pre-norm attention
        attn_out, attn_weights = self.attn(self.ln1(x), return_attention)
        x = x + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.ln2(x))

        return x, attn_weights
