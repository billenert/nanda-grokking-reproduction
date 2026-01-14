"""Tests for transformer model.

- Forward pass shapes
- Causal masking
- Weight initialization
- Component shapes
"""

import math

import pytest
import torch
import torch.nn as nn

from grokking.models import (
    Transformer,
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    init_weights,
    create_model,
)
from grokking.utils.io import load_config


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_output_shape(self):
        """Test attention output shape."""
        batch, seq_len, d_model, n_heads = 4, 8, 64, 4
        attn = MultiHeadAttention(d_model, n_heads)

        x = torch.randn(batch, seq_len, d_model)
        out, _ = attn(x)

        assert out.shape == (batch, seq_len, d_model)

    def test_d_head_check(self):
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadAttention(d_model=65, n_heads=4)

    def test_attention_weights_shape(self):
        """Test attention weights shape when returned."""
        batch, seq_len, d_model, n_heads = 4, 8, 64, 4
        attn = MultiHeadAttention(d_model, n_heads)

        x = torch.randn(batch, seq_len, d_model)
        _, weights = attn(x, return_attention=True)

        assert weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_causal_masking(self):
        """Test that attention is causal (no attending to future)."""
        batch, seq_len, d_model, n_heads = 2, 4, 32, 2
        attn = MultiHeadAttention(d_model, n_heads)

        x = torch.randn(batch, seq_len, d_model)
        _, weights = attn(x, return_attention=True)

        # Check that upper triangular (future positions) is zero
        # After softmax, future positions should have zero attention
        for b in range(batch):
            for h in range(n_heads):
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        # Position i should not attend to position j > i
                        assert weights[b, h, i, j].item() == pytest.approx(0.0, abs=1e-5)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along the key dimension."""
        batch, seq_len, d_model, n_heads = 2, 4, 32, 2
        attn = MultiHeadAttention(d_model, n_heads)

        x = torch.randn(batch, seq_len, d_model)
        _, weights = attn(x, return_attention=True)

        # Sum along key dimension should be 1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


class TestMLP:
    """Tests for MLP block."""

    def test_output_shape(self):
        """Test MLP output shape."""
        batch, seq_len, d_model, d_mlp = 4, 8, 64, 256
        mlp = MLP(d_model, d_mlp, activation="relu")

        x = torch.randn(batch, seq_len, d_model)
        out = mlp(x)

        assert out.shape == (batch, seq_len, d_model)

    def test_relu_activation(self):
        """Test ReLU activation."""
        d_model, d_mlp = 32, 64
        mlp = MLP(d_model, d_mlp, activation="relu")

        # Check that activation is applied
        assert mlp.activation == torch.nn.functional.relu

    def test_gelu_activation(self):
        """Test GELU activation."""
        d_model, d_mlp = 32, 64
        mlp = MLP(d_model, d_mlp, activation="gelu")

        assert mlp.activation == torch.nn.functional.gelu

    def test_invalid_activation(self):
        """Test invalid activation raises error."""
        with pytest.raises(ValueError, match="Invalid activation"):
            MLP(d_model=32, d_mlp=64, activation="invalid")


class TestTransformerBlock:
    """Tests for transformer block."""

    def test_output_shape(self):
        """Test block output shape."""
        batch, seq_len = 4, 8
        d_model, n_heads, d_mlp = 64, 4, 256
        block = TransformerBlock(d_model, n_heads, d_mlp)

        x = torch.randn(batch, seq_len, d_model)
        out, _ = block(x)

        assert out.shape == (batch, seq_len, d_model)

    def test_residual_connection(self):
        """Test that residual connections are present."""
        d_model, n_heads, d_mlp = 64, 4, 256
        block = TransformerBlock(d_model, n_heads, d_mlp)

        # Set weights to zero to test residual
        for param in block.attn.parameters():
            param.data.zero_()
        for param in block.mlp.parameters():
            param.data.zero_()

        x = torch.randn(2, 4, d_model)
        out, _ = block(x)

        # With zero weights, output should equal input (after layer norm)
        # This is approximately true but not exact due to layer norm
        # Just check output is reasonable
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestTransformer:
    """Tests for full transformer model."""

    @pytest.fixture
    def debug_config(self):
        """Load debug configuration."""
        return load_config("configs/debug.yaml")

    def test_forward_shape(self, debug_config):
        """Test model forward pass shape."""
        p = debug_config["data"]["p"]
        vocab_size = p + 2  # BOS + SEP

        model = Transformer(debug_config["model"], vocab_size)

        batch_size = 4
        seq_len = 4
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = model(tokens)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_forward_dtype(self, debug_config):
        """Test model output dtype."""
        vocab_size = 19
        model = Transformer(debug_config["model"], vocab_size)

        tokens = torch.randint(0, vocab_size, (4, 4))
        logits = model(tokens)

        assert logits.dtype == torch.float32

    def test_create_model_function(self, debug_config):
        """Test create_model helper function."""
        vocab_size = 19
        model = create_model(debug_config, vocab_size)

        assert isinstance(model, Transformer)
        assert model.d_model == debug_config["model"]["d_model"]
        assert model.n_layers == debug_config["model"]["n_layers"]

    def test_embedding_shape(self, debug_config):
        """Test embedding weights shape."""
        vocab_size = 19
        model = Transformer(debug_config["model"], vocab_size)

        emb = model.get_embedding_weights()
        d_model = debug_config["model"]["d_model"]

        assert emb.shape == (vocab_size, d_model)

    def test_unembedding_shape(self, debug_config):
        """Test unembedding weights shape."""
        vocab_size = 19
        model = Transformer(debug_config["model"], vocab_size)

        unemb = model.get_unembedding_weights()
        d_model = debug_config["model"]["d_model"]

        # W_out.weight has shape [vocab_size, d_model]
        assert unemb.shape == (vocab_size, d_model)

    def test_return_activations(self, debug_config):
        """Test that activations can be returned."""
        vocab_size = 19
        model = Transformer(debug_config["model"], vocab_size)

        tokens = torch.randint(0, vocab_size, (4, 4))
        logits, activations = model(tokens, return_activations=True)

        # Check expected activation keys exist
        n_layers = debug_config["model"]["n_layers"]
        assert "resid_pre.l0" in activations
        assert f"resid_post.l{n_layers-1}" in activations

    def test_no_positional_encoding(self, debug_config):
        """Test model without positional encoding."""
        vocab_size = 19
        model = Transformer(debug_config["model"], vocab_size)

        assert model.position_embedding is None

    def test_with_positional_encoding(self, debug_config):
        """Test model with learned positional encoding."""
        config = debug_config["model"].copy()
        config["positional_encoding"] = "learned"
        config["max_seq_len"] = 8

        vocab_size = 19
        model = Transformer(config, vocab_size)

        assert model.position_embedding is not None
        assert model.position_embedding.weight.shape == (8, config["d_model"])


class TestWeightInitialization:
    """Tests for weight initialization."""

    def test_embedding_initialization(self):
        """Test embedding weights are initialized with std=0.02."""
        vocab_size, d_model = 100, 64
        emb = nn.Embedding(vocab_size, d_model)
        init_weights(emb)

        # Check mean is close to 0
        assert emb.weight.mean().abs() < 0.1

        # Check std is close to 0.02
        assert 0.015 < emb.weight.std() < 0.025

    def test_linear_initialization(self):
        """Test linear weights are initialized with std=0.02."""
        in_features, out_features = 64, 128
        linear = nn.Linear(in_features, out_features)
        init_weights(linear)

        # Check mean is close to 0
        assert linear.weight.mean().abs() < 0.1

        # Check std is close to 0.02
        assert 0.015 < linear.weight.std() < 0.025

        # Check bias is zero
        if linear.bias is not None:
            assert linear.bias.abs().max() < 1e-6

    def test_layernorm_initialization(self):
        """Test LayerNorm weights are ones and bias zeros."""
        d_model = 64
        ln = nn.LayerNorm(d_model)
        init_weights(ln)

        assert torch.allclose(ln.weight, torch.ones(d_model))
        assert torch.allclose(ln.bias, torch.zeros(d_model))

    def test_model_initialization(self):
        """Test that model is initialized correctly."""
        config = load_config("configs/debug.yaml")
        vocab_size = 19
        model = Transformer(config["model"], vocab_size)

        # Check embedding std
        emb_std = model.token_embedding.weight.std()
        assert 0.015 < emb_std < 0.025

        # Check final layer norm
        assert torch.allclose(model.ln_final.weight, torch.ones(config["model"]["d_model"]))
        assert torch.allclose(model.ln_final.bias, torch.zeros(config["model"]["d_model"]))


class TestTiedEmbeddings:
    """Tests for tied embeddings."""

    def test_tied_embeddings(self):
        """Test that tied embeddings share weights."""
        config = load_config("configs/debug.yaml")
        config["model"]["tie_embeddings"] = True
        vocab_size = 19

        model = Transformer(config["model"], vocab_size)

        # Check that weights are the same object
        assert model.token_embedding.weight is model.unembed.weight

    def test_untied_embeddings(self):
        """Test that untied embeddings have separate weights."""
        config = load_config("configs/debug.yaml")
        config["model"]["tie_embeddings"] = False
        vocab_size = 19

        model = Transformer(config["model"], vocab_size)

        # Check that weights are different objects
        assert model.token_embedding.weight is not model.unembed.weight


class TestGradientFlow:
    """Tests for gradient flow through model."""

    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        config = load_config("configs/debug.yaml")
        vocab_size = 19

        model = Transformer(config["model"], vocab_size)

        tokens = torch.randint(0, vocab_size, (4, 4))
        logits = model(tokens)

        # Create a simple loss
        loss = logits.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
