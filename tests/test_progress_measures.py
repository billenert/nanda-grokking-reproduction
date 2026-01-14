"""Tests for progress measures."""

import pytest
import torch

from grokking.models import Transformer
from grokking.interp.progress_measures import (
    compute_topk_mass,
    compute_spectral_alignment,
    compute_progress_measures,
    compute_all_progress_measures,
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


class TestComputeTopkMass:
    """Tests for compute_topk_mass function."""

    def test_returns_float(self):
        """Should return a float value."""
        freq_total = [0.1, 0.2, 0.3, 0.4]
        result = compute_topk_mass(freq_total, k=2)
        assert isinstance(result, float)

    def test_range_zero_to_one(self):
        """Result should be between 0 and 1."""
        freq_total = [0.1, 0.2, 0.3, 0.4]
        result = compute_topk_mass(freq_total, k=2)
        assert 0.0 <= result <= 1.0

    def test_all_frequencies_returns_one(self):
        """Taking all frequencies should return 1.0."""
        freq_total = [0.1, 0.2, 0.3]
        result = compute_topk_mass(freq_total, k=3)
        assert abs(result - 1.0) < 1e-6

    def test_topk_selects_highest(self):
        """Should select the k highest energy frequencies."""
        freq_total = [0.1, 0.4, 0.3, 0.2]  # sum = 1.0
        # Top 2: 0.4 + 0.3 = 0.7
        result = compute_topk_mass(freq_total, k=2)
        assert abs(result - 0.7) < 1e-6

    def test_single_frequency(self):
        """k=1 should return highest frequency ratio."""
        freq_total = [0.1, 0.5, 0.2, 0.2]  # sum = 1.0, max = 0.5
        result = compute_topk_mass(freq_total, k=1)
        assert abs(result - 0.5) < 1e-6

    def test_empty_list_returns_zero(self):
        """Empty list should return 0."""
        result = compute_topk_mass([], k=1)
        assert result == 0.0

    def test_all_zeros_returns_zero(self):
        """All-zero energies should return 0."""
        freq_total = [0.0, 0.0, 0.0]
        result = compute_topk_mass(freq_total, k=2)
        assert result == 0.0


class TestComputeSpectralAlignment:
    """Tests for compute_spectral_alignment function."""

    def test_returns_float(self):
        """Should return a float value."""
        embed_freq = [0.1, 0.2, 0.3]
        unembed_freq = [0.2, 0.3, 0.1]
        result = compute_spectral_alignment(embed_freq, unembed_freq)
        assert isinstance(result, float)

    def test_identical_spectra(self):
        """Identical spectra should have alignment 1.0."""
        freq = [0.1, 0.2, 0.3]
        result = compute_spectral_alignment(freq, freq)
        assert abs(result - 1.0) < 1e-5

    def test_orthogonal_spectra(self):
        """Orthogonal spectra should have alignment 0.0."""
        embed_freq = [1.0, 0.0, 0.0]
        unembed_freq = [0.0, 1.0, 0.0]
        result = compute_spectral_alignment(embed_freq, unembed_freq)
        assert abs(result) < 1e-5

    def test_range(self):
        """Result should be between -1 and 1 (cosine similarity range)."""
        embed_freq = [0.1, 0.2, 0.3]
        unembed_freq = [0.3, 0.1, 0.2]
        result = compute_spectral_alignment(embed_freq, unembed_freq)
        assert -1.0 <= result <= 1.0

    def test_empty_lists_return_zero(self):
        """Empty lists should return 0."""
        assert compute_spectral_alignment([], [0.1, 0.2]) == 0.0
        assert compute_spectral_alignment([0.1, 0.2], []) == 0.0

    def test_zero_norm_returns_zero(self):
        """Zero-norm vectors should return 0."""
        result = compute_spectral_alignment([0.0, 0.0], [0.1, 0.2])
        assert result == 0.0


class TestComputeProgressMeasures:
    """Tests for compute_progress_measures function."""

    def test_returns_dict(self, model):
        """Should return a dictionary."""
        p = 7
        result = compute_progress_measures(model, p, k=4)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, model):
        """Should contain all required measure keys."""
        p = 7
        k = 4
        result = compute_progress_measures(model, p, k=k)

        assert f"embed_mass_topk_{k}" in result
        assert f"unembed_mass_topk_{k}" in result
        assert "spec_align" in result

    def test_values_are_floats(self, model):
        """All values should be floats."""
        p = 7
        result = compute_progress_measures(model, p, k=4)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float"

    def test_mass_in_valid_range(self, model):
        """Mass values should be between 0 and 1."""
        p = 7
        k = 4
        result = compute_progress_measures(model, p, k=k)

        assert 0.0 <= result[f"embed_mass_topk_{k}"] <= 1.0
        assert 0.0 <= result[f"unembed_mass_topk_{k}"] <= 1.0

    def test_alignment_in_valid_range(self, model):
        """Alignment should be between -1 and 1."""
        p = 7
        result = compute_progress_measures(model, p, k=4)
        assert -1.0 <= result["spec_align"] <= 1.0


class TestComputeAllProgressMeasures:
    """Tests for compute_all_progress_measures function."""

    def test_returns_dict(self, model):
        """Should return a dictionary."""
        p = 7
        result = compute_all_progress_measures(model, p)
        assert isinstance(result, dict)

    def test_default_k_values(self, model):
        """Should compute for default k values [4, 8]."""
        p = 7
        result = compute_all_progress_measures(model, p)

        # Default k_values are [4, 8], but p=7 only has 3 frequencies
        # so we just check that the keys exist
        assert "spec_align" in result

    def test_custom_k_values(self, model):
        """Should compute for custom k values."""
        p = 7
        k_values = [1, 2]
        result = compute_all_progress_measures(model, p, k_values=k_values)

        assert "embed_mass_topk_1" in result
        assert "unembed_mass_topk_1" in result
        assert "embed_mass_topk_2" in result
        assert "unembed_mass_topk_2" in result
        assert "spec_align" in result

    def test_values_are_valid(self, model):
        """All values should be valid floats."""
        p = 7
        result = compute_all_progress_measures(model, p, k_values=[2])

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float"
            assert not torch.isnan(torch.tensor(value)), f"{key} is NaN"


class TestProgressMeasuresDeterminism:
    """Tests for determinism of progress measures."""

    def test_same_model_same_results(self, model):
        """Same model should produce same results."""
        p = 7
        result1 = compute_progress_measures(model, p, k=4)
        result2 = compute_progress_measures(model, p, k=4)

        for key in result1:
            assert result1[key] == result2[key], f"{key} differs between runs"
