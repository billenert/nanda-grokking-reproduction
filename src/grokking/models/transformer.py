"""Decoder-only transformer model.

- Token embeddings (optional positional embeddings)
- Stack of transformer blocks with pre-norm architecture
- Final layer norm and unembedding
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from grokking.models.components import TransformerBlock


class Transformer(nn.Module):
    """Decoder-only transformer for modular addition.

    Architecture:
        1. Token embedding (+ optional position embedding)
        2. N transformer blocks (pre-norm)
        3. Final layer norm
        4. Unembedding linear layer

    Forward contract:
        Input: tokens [batch, seq_len] (LongTensor)
        Output: logits [batch, seq_len, vocab_size] (FloatTensor)
    """

    def __init__(
        self,
        config: dict[str, Any],
        vocab_size: int,
    ):
        """Initialize transformer.

        Args:
            config: Model configuration dictionary with keys:
                - d_model: Model dimension
                - n_layers: Number of transformer blocks
                - n_heads: Number of attention heads
                - d_mlp: MLP hidden dimension
                - dropout: Dropout probability
                - layer_norm_eps: LayerNorm epsilon
                - tie_embeddings: Whether to tie embedding/unembedding weights
                - positional_encoding: "none" or "learned"
                - max_seq_len: Max sequence length (for learned positions)
                - mlp_activation: "relu" or "gelu"
            vocab_size: Size of vocabulary.
        """
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size
        self.d_model = config["d_model"]
        self.n_layers = config["n_layers"]

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, self.d_model)

        # Optional positional embedding
        self.positional_encoding = config.get("positional_encoding", "none")
        if self.positional_encoding == "learned":
            max_seq_len = config.get("max_seq_len", 4)
            self.position_embedding = nn.Embedding(max_seq_len, self.d_model)
        else:
            self.position_embedding = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=config["n_heads"],
                d_mlp=config["d_mlp"],
                activation=config.get("mlp_activation", "relu"),
                dropout=config.get("dropout", 0.0),
                layer_norm_eps=config.get("layer_norm_eps", 1e-5),
            )
            for _ in range(self.n_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(
            self.d_model,
            eps=config.get("layer_norm_eps", 1e-5)
        )

        # Unembedding (output projection)
        # W_out.weight has shape [vocab_size, d_model]
        self.unembed = nn.Linear(self.d_model, vocab_size, bias=False)

        # Optionally tie weights
        if config.get("tie_embeddings", False):
            self.unembed.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights according to spec.

        - Embeddings: normal(0, 0.02)
        - Linear layers: normal(0, 0.02)
        - Biases: zeros
        - LayerNorm weights: ones, biases: zeros
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            tokens: Input tokens of shape [batch, seq_len].
            return_activations: Whether to return intermediate activations.

        Returns:
            If return_activations is False:
                logits: Output logits of shape [batch, seq_len, vocab_size]
            If return_activations is True:
                Tuple of (logits, activations_dict)
        """
        batch_size, seq_len = tokens.shape
        activations = {} if return_activations else None

        # Token embedding
        x = self.token_embedding(tokens)  # [batch, seq_len, d_model]

        # Add position embedding if using learned positions
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=tokens.device)
            x = x + self.position_embedding(positions)

        # Store initial residual stream
        if return_activations:
            activations["resid_pre.l0"] = x.detach()

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x, attn_weights = block(x, return_attention=return_activations)

            if return_activations:
                activations[f"resid_post.l{i}"] = x.detach()
                if attn_weights is not None:
                    activations[f"attn_weights.l{i}"] = attn_weights.detach()
                if i < self.n_layers - 1:
                    activations[f"resid_pre.l{i+1}"] = x.detach()

        # Final layer norm
        x = self.ln_final(x)

        # Unembedding
        logits = self.unembed(x)  # [batch, seq_len, vocab_size]

        if return_activations:
            return logits, activations
        return logits

    def get_embedding_weights(self) -> torch.Tensor:
        """Get token embedding weights.

        Returns:
            Embedding weight matrix of shape [vocab_size, d_model].
        """
        return self.token_embedding.weight

    def get_unembedding_weights(self) -> torch.Tensor:
        """Get unembedding weights.

        Returns:
            Unembedding weight matrix of shape [vocab_size, d_model].
        """
        return self.unembed.weight
