"""Tests for Fourier infrastructure."""

import pytest
import torch

from grokking.interp.fourier import (
    real_fourier_basis,
    project_rows_to_basis,
    energy_per_basis,
    energy_per_frequency_group,
    compute_spectral_energy,
    get_topk_frequencies,
)


class TestRealFourierBasis:
    """Tests for the real Fourier basis construction."""

    @pytest.mark.parametrize("p", [5, 7, 11, 17, 113])
    def test_basis_shape(self, p: int):
        """Basis should have shape [p, p]."""
        B = real_fourier_basis(p)
        assert B.shape == (p, p)

    @pytest.mark.parametrize("p", [5, 7, 11, 17])
    def test_orthonormality(self, p: int):
        """B @ B.T should be identity within tolerance."""
        B = real_fourier_basis(p)
        product = B @ B.T
        identity = torch.eye(p)
        assert torch.allclose(product, identity, atol=1e-4), (
            f"B @ B.T not close to identity for p={p}. "
            f"Max deviation: {(product - identity).abs().max().item()}"
        )

    @pytest.mark.parametrize("p", [5, 7, 11, 17])
    def test_orthonormality_columns(self, p: int):
        """B.T @ B should also be identity (columns orthonormal)."""
        B = real_fourier_basis(p)
        product = B.T @ B
        identity = torch.eye(p)
        assert torch.allclose(product, identity, atol=1e-4), (
            f"B.T @ B not close to identity for p={p}. "
            f"Max deviation: {(product - identity).abs().max().item()}"
        )

    @pytest.mark.parametrize("p", [5, 7, 11, 17])
    def test_reconstruction(self, p: int):
        """B.T @ (B @ W) should reconstruct W."""
        B = real_fourier_basis(p)
        torch.manual_seed(42)
        W = torch.randn(p, 8)  # Random weight matrix
        C = B @ W  # Project to Fourier
        W_reconstructed = B.T @ C  # Reconstruct
        assert torch.allclose(W_reconstructed, W, atol=1e-4), (
            f"Reconstruction failed for p={p}. "
            f"Max deviation: {(W_reconstructed - W).abs().max().item()}"
        )

    def test_dc_component_is_constant(self):
        """DC component (row 0) should be constant 1/sqrt(p)."""
        p = 11
        B = real_fourier_basis(p)
        expected = 1.0 / (p ** 0.5)
        assert torch.allclose(B[0], torch.full((p,), expected), atol=1e-6)

    def test_device_parameter(self):
        """Basis should be created on specified device."""
        p = 7
        B = real_fourier_basis(p, device=torch.device("cpu"))
        assert B.device == torch.device("cpu")


class TestProjectRowsToBasis:
    """Tests for Fourier projection."""

    def test_projection_shape(self):
        """Projection should preserve second dimension."""
        p = 11
        d = 32
        B = real_fourier_basis(p)
        W = torch.randn(p, d)
        C = project_rows_to_basis(W, B)
        assert C.shape == (p, d)

    def test_projection_of_constant(self):
        """Constant function should have energy only in DC."""
        p = 7
        B = real_fourier_basis(p)
        # Constant function (all same value in each column)
        W = torch.ones(p, 4) * 3.0
        C = project_rows_to_basis(W, B)
        # DC coefficient should be non-zero
        assert C[0].abs().sum() > 0
        # Other coefficients should be near zero
        assert torch.allclose(C[1:], torch.zeros(p - 1, 4), atol=1e-5)


class TestEnergyComputation:
    """Tests for energy computation functions."""

    def test_energy_per_basis_shape(self):
        """Energy should have shape [p]."""
        p = 11
        C = torch.randn(p, 32)
        e = energy_per_basis(C)
        assert e.shape == (p,)

    def test_energy_is_nonnegative(self):
        """Energy values should be non-negative."""
        p = 11
        C = torch.randn(p, 32)
        e = energy_per_basis(C)
        assert (e >= 0).all()

    def test_energy_matches_squared_norm(self):
        """Energy should equal sum of squared coefficients per row."""
        p = 7
        C = torch.randn(p, 8)
        e = energy_per_basis(C)
        expected = (C ** 2).sum(dim=-1)
        assert torch.allclose(e, expected)

    def test_energy_per_frequency_group_keys(self):
        """Should return all expected keys."""
        p = 11
        n_freq = (p - 1) // 2
        e = torch.randn(p).abs()  # Non-negative
        result = energy_per_frequency_group(e, p)

        # Check required keys
        assert "dc" in result
        assert "freq_total" in result
        assert len(result["freq_total"]) == n_freq

        for k in range(1, n_freq + 1):
            assert f"cos_{k}" in result
            assert f"sin_{k}" in result
            assert f"freq_{k}_total" in result

    def test_energy_per_frequency_group_values(self):
        """Freq total should equal cos + sin for each frequency."""
        p = 11
        n_freq = (p - 1) // 2
        e = torch.randn(p).abs()
        result = energy_per_frequency_group(e, p)

        for k in range(1, n_freq + 1):
            expected = result[f"cos_{k}"] + result[f"sin_{k}"]
            assert abs(result[f"freq_{k}_total"] - expected) < 1e-6


class TestComputeSpectralEnergy:
    """Tests for the full spectral energy computation."""

    def test_full_pipeline(self):
        """Test the full spectral energy pipeline."""
        p = 7
        W = torch.randn(p, 16)
        result = compute_spectral_energy(W, p)

        assert "dc" in result
        assert "freq_total" in result
        assert len(result["freq_total"]) == (p - 1) // 2

    def test_truncates_to_p_rows(self):
        """Should work with W having more than p rows."""
        p = 7
        vocab_size = 10
        W = torch.randn(vocab_size, 16)
        result = compute_spectral_energy(W, p)

        # Should still compute correctly
        assert "dc" in result
        assert len(result["freq_total"]) == (p - 1) // 2


class TestGetTopkFrequencies:
    """Tests for finding top-k frequencies by energy."""

    def test_returns_k_frequencies(self):
        """Should return exactly k frequencies."""
        freq_total = [0.1, 0.5, 0.3, 0.8, 0.2]
        k = 3
        result = get_topk_frequencies(freq_total, k)
        assert len(result) == k

    def test_returns_highest_energy(self):
        """Should return frequencies with highest energy."""
        freq_total = [0.1, 0.5, 0.3, 0.8, 0.2]
        k = 2
        result = get_topk_frequencies(freq_total, k)
        # freq_total[3] = 0.8 -> frequency 4
        # freq_total[1] = 0.5 -> frequency 2
        assert set(result) == {4, 2}

    def test_returns_one_indexed(self):
        """Result should be 1-indexed frequencies."""
        freq_total = [1.0, 0.1, 0.1]  # First frequency has highest energy
        k = 1
        result = get_topk_frequencies(freq_total, k)
        assert result == [1]  # 1-indexed

    def test_handles_ties(self):
        """Should handle tied energies deterministically."""
        freq_total = [0.5, 0.5, 0.5]
        k = 2
        result = get_topk_frequencies(freq_total, k)
        assert len(result) == k


class TestParseval:
    """Test Parseval's theorem: energy conservation."""

    @pytest.mark.parametrize("p", [5, 7, 11])
    def test_parseval_identity(self, p: int):
        """Total energy in spatial domain equals total energy in Fourier domain."""
        B = real_fourier_basis(p)
        torch.manual_seed(123)
        W = torch.randn(p, 8)

        # Spatial energy
        spatial_energy = (W ** 2).sum()

        # Fourier energy
        C = project_rows_to_basis(W, B)
        fourier_energy = (C ** 2).sum()

        assert torch.allclose(spatial_energy, fourier_energy, rtol=1e-4), (
            f"Parseval failed for p={p}. "
            f"Spatial: {spatial_energy.item()}, Fourier: {fourier_energy.item()}"
        )
