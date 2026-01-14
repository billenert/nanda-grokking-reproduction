"""Tests for ablation functionality."""

import copy

import pytest
import torch

from grokking.models import Transformer
from grokking.interp.fourier import real_fourier_basis, compute_spectral_energy
from grokking.interp.ablations import (
    parse_frequency_set,
    ablate_embedding_frequencies,
    run_ablation,
)


@pytest.fixture
def model_config():
    """Create a minimal model config for testing."""
    return {
        "d_model": 32,
        "n_layers": 1,
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
    p = 7
    vocab_size = p + 2  # BOS_A_B_SEP format
    return Transformer(model_config, vocab_size=vocab_size)


class TestParseFrequencySet:
    """Tests for parse_frequency_set function."""

    def test_single_frequency(self):
        """Should return single frequency."""
        spec = {"type": "single", "k": 3}
        result = parse_frequency_set(spec, p=11)
        assert result == [3]

    def test_dc_frequency(self):
        """Should return [0] for DC."""
        spec = {"type": "dc"}
        result = parse_frequency_set(spec, p=11)
        assert result == [0]

    def test_range_frequencies(self):
        """Should return range of frequencies."""
        spec = {"type": "range", "k": 2, "end": 4}
        result = parse_frequency_set(spec, p=11)
        assert result == [2, 3, 4]

    def test_range_to_end(self):
        """Range without end should go to max frequency."""
        spec = {"type": "range", "k": 1}
        p = 7
        n_freq = (p - 1) // 2  # 3
        result = parse_frequency_set(spec, p=p)
        assert result == [1, 2, 3]

    def test_topk_frequencies(self):
        """Should return top k frequencies by energy."""
        freq_energies = [0.1, 0.5, 0.3]  # freq 2 highest, then 3, then 1
        spec = {"type": "topk", "k": 2}
        result = parse_frequency_set(spec, p=7, freq_energies=freq_energies)
        # Should return frequencies 2 and 3 (indices of highest energy)
        assert len(result) == 2
        assert set(result) == {2, 3}

    def test_topk_requires_energies(self):
        """topk without energies should raise error."""
        spec = {"type": "topk", "k": 2}
        with pytest.raises(ValueError, match="freq_energies required"):
            parse_frequency_set(spec, p=7)

    def test_unknown_type_raises(self):
        """Unknown type should raise error."""
        spec = {"type": "unknown"}
        with pytest.raises(ValueError, match="Unknown frequency set type"):
            parse_frequency_set(spec, p=7)


class TestAblateEmbeddingFrequencies:
    """Tests for ablate_embedding_frequencies function."""

    def test_modifies_model_inplace(self, model):
        """Should modify model embedding in place."""
        p = 7
        original_embed = model.token_embedding.weight.data[:p, :].clone()

        ablate_embedding_frequencies(model, [1], p)

        new_embed = model.token_embedding.weight.data[:p, :]
        # Embeddings should have changed
        assert not torch.allclose(original_embed, new_embed)

    def test_ablate_dc_zeros_dc(self, model):
        """Ablating DC should zero out DC component."""
        p = 7

        ablate_embedding_frequencies(model, [0], p)

        # Check DC component is zero
        E = model.token_embedding.weight.data[:p, :]
        B = real_fourier_basis(p, device=E.device)
        C = B @ E

        # DC is row 0
        assert torch.allclose(C[0], torch.zeros_like(C[0]), atol=1e-5)

    def test_ablate_frequency_zeros_both_cos_and_sin(self, model):
        """Ablating frequency k should zero both cos and sin components."""
        p = 7
        n_freq = (p - 1) // 2
        k = 2

        ablate_embedding_frequencies(model, [k], p)

        E = model.token_embedding.weight.data[:p, :]
        B = real_fourier_basis(p, device=E.device)
        C = B @ E

        # Cos at row k, sin at row (n_freq + k)
        assert torch.allclose(C[k], torch.zeros_like(C[k]), atol=1e-5)
        assert torch.allclose(C[n_freq + k], torch.zeros_like(C[n_freq + k]), atol=1e-5)

    def test_ablate_preserves_other_frequencies(self, model):
        """Ablating one frequency should preserve others."""
        p = 7

        # Get original coefficients
        E_orig = model.token_embedding.weight.data[:p, :].clone()
        B = real_fourier_basis(p, device=E_orig.device)
        C_orig = B @ E_orig

        # Ablate frequency 1
        ablate_embedding_frequencies(model, [1], p)

        E_new = model.token_embedding.weight.data[:p, :]
        C_new = B @ E_new

        # DC should be preserved
        assert torch.allclose(C_new[0], C_orig[0], atol=1e-5)

        # Frequency 2 and 3 should be preserved
        n_freq = (p - 1) // 2
        for k in [2, 3]:
            assert torch.allclose(C_new[k], C_orig[k], atol=1e-5)
            assert torch.allclose(C_new[n_freq + k], C_orig[n_freq + k], atol=1e-5)

    def test_ablate_multiple_frequencies(self, model):
        """Should ablate multiple frequencies."""
        p = 7
        n_freq = (p - 1) // 2

        ablate_embedding_frequencies(model, [1, 2], p)

        E = model.token_embedding.weight.data[:p, :]
        B = real_fourier_basis(p, device=E.device)
        C = B @ E

        # Both frequencies should be zeroed
        for k in [1, 2]:
            assert torch.allclose(C[k], torch.zeros_like(C[k]), atol=1e-5)
            assert torch.allclose(C[n_freq + k], torch.zeros_like(C[n_freq + k]), atol=1e-5)


class TestRunAblation:
    """Tests for run_ablation function."""

    def test_returns_baseline_and_results(self, model):
        """Should return baseline accuracy and results list."""
        p = 7

        def dummy_eval(m):
            return 0.5

        freq_specs = [{"type": "single", "k": 1}]
        acc_base, results = run_ablation(model, freq_specs, dummy_eval, p)

        assert isinstance(acc_base, float)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_results_contain_required_keys(self, model):
        """Results should contain set, acc, and freq_indices."""
        p = 7

        def dummy_eval(m):
            return 0.5

        freq_specs = [{"type": "single", "k": 2}]
        _, results = run_ablation(model, freq_specs, dummy_eval, p)

        result = results[0]
        assert "set" in result
        assert "acc" in result
        assert "freq_indices" in result

    def test_set_names_formatted_correctly(self, model):
        """Set names should follow expected format."""
        p = 7

        def dummy_eval(m):
            return 0.5

        freq_specs = [
            {"type": "single", "k": 3},
            {"type": "topk", "k": 2},
            {"type": "dc"},
        ]
        _, results = run_ablation(model, freq_specs, dummy_eval, p)

        assert results[0]["set"] == "single(3)"
        assert results[1]["set"] == "topk(2)"
        assert results[2]["set"] == "dc"

    def test_does_not_modify_original_model(self, model):
        """Original model should not be modified."""
        p = 7

        original_embed = model.token_embedding.weight.data[:p, :].clone()

        def dummy_eval(m):
            return 0.5

        freq_specs = [{"type": "single", "k": 1}]
        run_ablation(model, freq_specs, dummy_eval, p)

        # Original model unchanged
        assert torch.allclose(
            model.token_embedding.weight.data[:p, :],
            original_embed,
        )


class TestAblationReconstruction:
    """Tests for reconstruction after ablation."""

    def test_embedding_reconstructs_after_full_project(self, model):
        """Projecting to Fourier and back should preserve embedding."""
        p = 7

        E_orig = model.token_embedding.weight.data[:p, :].clone()
        B = real_fourier_basis(p, device=E_orig.device)

        # Project to Fourier and back
        C = B @ E_orig
        E_reconstructed = B.T @ C

        assert torch.allclose(E_reconstructed, E_orig, atol=1e-5)
