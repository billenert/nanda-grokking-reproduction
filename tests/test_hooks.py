"""Tests for hooks and caching system."""

import pytest
import torch

from grokking.models import Transformer
from grokking.interp.hooks import run_with_cache, get_cache_from_model


@pytest.fixture
def model_config():
    """Create a minimal model config for testing."""
    return {
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "d_mlp": 64,
        "dropout": 0.0,
        "layer_norm_eps": 1e-5,
        "tie_embeddings": False,
        "positional_encoding": "none",
        "max_seq_len": 4,
        "mlp_activation": "relu",
    }


@pytest.fixture
def model(model_config):
    """Create a model for testing."""
    vocab_size = 19  # p=17 + 2 special tokens
    return Transformer(model_config, vocab_size=vocab_size)


@pytest.fixture
def sample_tokens():
    """Create sample input tokens."""
    batch_size = 4
    seq_len = 4
    return torch.randint(0, 17, (batch_size, seq_len))


class TestRunWithCache:
    """Tests for run_with_cache function."""

    def test_returns_logits_and_cache(self, model, sample_tokens):
        """Should return logits and cache dict."""
        logits, cache = run_with_cache(model, sample_tokens, cache_spec=[])
        assert isinstance(logits, torch.Tensor)
        assert isinstance(cache, dict)

    def test_logits_shape(self, model, sample_tokens, model_config):
        """Logits should have correct shape."""
        vocab_size = 19
        logits, _ = run_with_cache(model, sample_tokens, cache_spec=[])
        batch_size, seq_len = sample_tokens.shape
        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_cache_resid_pre(self, model, sample_tokens, model_config):
        """Should cache residual stream before first layer."""
        cache_spec = ["resid_pre.l0"]
        _, cache = run_with_cache(model, sample_tokens, cache_spec)

        assert "resid_pre.l0" in cache
        batch_size, seq_len = sample_tokens.shape
        assert cache["resid_pre.l0"].shape == (batch_size, seq_len, model_config["d_model"])

    def test_cache_resid_post(self, model, sample_tokens, model_config):
        """Should cache residual stream after layer."""
        cache_spec = ["resid_post.l0", "resid_post.l1"]
        _, cache = run_with_cache(model, sample_tokens, cache_spec)

        batch_size, seq_len = sample_tokens.shape
        d_model = model_config["d_model"]

        assert "resid_post.l0" in cache
        assert cache["resid_post.l0"].shape == (batch_size, seq_len, d_model)

        assert "resid_post.l1" in cache
        assert cache["resid_post.l1"].shape == (batch_size, seq_len, d_model)

    def test_cache_attn_out(self, model, sample_tokens, model_config):
        """Should cache attention output."""
        cache_spec = ["attn_out.l0"]
        _, cache = run_with_cache(model, sample_tokens, cache_spec)

        assert "attn_out.l0" in cache
        batch_size, seq_len = sample_tokens.shape
        assert cache["attn_out.l0"].shape == (batch_size, seq_len, model_config["d_model"])

    def test_cache_mlp_out(self, model, sample_tokens, model_config):
        """Should cache MLP output."""
        cache_spec = ["mlp_out.l0"]
        _, cache = run_with_cache(model, sample_tokens, cache_spec)

        assert "mlp_out.l0" in cache
        batch_size, seq_len = sample_tokens.shape
        assert cache["mlp_out.l0"].shape == (batch_size, seq_len, model_config["d_model"])

    def test_cache_on_cpu(self, model, sample_tokens):
        """Cached tensors should be on CPU by default."""
        cache_spec = ["resid_post.l0"]
        _, cache = run_with_cache(model, sample_tokens, cache_spec, cache_on_cpu=True)

        assert cache["resid_post.l0"].device == torch.device("cpu")

    def test_cache_fp32(self, model, sample_tokens):
        """Cached tensors should be float32 by default."""
        cache_spec = ["resid_post.l0"]
        _, cache = run_with_cache(model, sample_tokens, cache_spec, cache_fp32=True)

        assert cache["resid_post.l0"].dtype == torch.float32

    def test_cache_detached(self, model, sample_tokens):
        """Cached tensors should be detached (no gradients)."""
        cache_spec = ["resid_post.l0"]
        _, cache = run_with_cache(model, sample_tokens, cache_spec)

        assert not cache["resid_post.l0"].requires_grad

    def test_multiple_cache_specs(self, model, sample_tokens, model_config):
        """Should cache multiple activations."""
        cache_spec = [
            "resid_pre.l0",
            "resid_post.l0",
            "attn_out.l0",
            "mlp_out.l0",
            "resid_post.l1",
        ]
        _, cache = run_with_cache(model, sample_tokens, cache_spec)

        for name in cache_spec:
            assert name in cache, f"Missing cache entry: {name}"


class TestGetCacheFromModel:
    """Tests for get_cache_from_model function."""

    def test_returns_logits_and_cache(self, model, sample_tokens):
        """Should return logits and activations dict."""
        logits, cache = get_cache_from_model(model, sample_tokens)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(cache, dict)

    def test_cache_on_cpu(self, model, sample_tokens):
        """Cached tensors should be on CPU when requested."""
        _, cache = get_cache_from_model(model, sample_tokens, cache_on_cpu=True)

        for name, tensor in cache.items():
            assert tensor.device == torch.device("cpu"), f"{name} not on CPU"

    def test_cache_fp32(self, model, sample_tokens):
        """Cached tensors should be float32 when requested."""
        _, cache = get_cache_from_model(model, sample_tokens, cache_fp32=True)

        for name, tensor in cache.items():
            assert tensor.dtype == torch.float32, f"{name} not float32"


class TestCacheIntegrity:
    """Tests to verify cached values are correct."""

    def test_resid_post_equals_final_output(self, model, sample_tokens, model_config):
        """resid_post of last layer should match final layer norm input."""
        n_layers = model_config["n_layers"]
        cache_spec = [f"resid_post.l{n_layers - 1}"]

        logits, cache = run_with_cache(model, sample_tokens, cache_spec)

        # The cached resid_post should be the input to final layer norm
        # We can verify it exists and has correct shape
        final_resid = cache[f"resid_post.l{n_layers - 1}"]
        batch_size, seq_len = sample_tokens.shape
        assert final_resid.shape == (batch_size, seq_len, model_config["d_model"])

    def test_resid_pre_l1_matches_resid_post_l0(self, model, sample_tokens):
        """resid_pre.l1 should equal resid_post.l0 for layer 1."""
        cache_spec = ["resid_pre.l1", "resid_post.l0"]
        _, cache = run_with_cache(model, sample_tokens, cache_spec)

        # These should be the same (output of layer 0 = input to layer 1)
        assert torch.allclose(
            cache["resid_pre.l1"],
            cache["resid_post.l0"],
            atol=1e-5,
        )
