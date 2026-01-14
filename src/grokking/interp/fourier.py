"""Fourier infrastructure for mechanistic interpretability.

- Real Fourier basis over Z_p
- DC component: phi_0(x) = 1 / sqrt(p)
- Cosine components: phi_cos_k(x) = sqrt(2/p) * cos(2*pi*k*x/p)
- Sine components: phi_sin_k(x) = sqrt(2/p) * sin(2*pi*k*x/p)

The basis is orthonormal: B @ B.T ≈ I
"""

import math
from typing import Any

import torch


def real_fourier_basis(p: int, device: torch.device | None = None) -> torch.Tensor:
    """Construct the real Fourier basis for Z_p.

    Args:
        p: Prime modulus (should be odd prime).
        device: Device to create tensor on.

    Returns:
        FloatTensor of shape [p, p] where rows are orthonormal basis vectors.
        Row ordering:
            - Row 0: DC component
            - Rows 1 to (p-1)/2: cosine components for k=1..(p-1)/2
            - Rows (p+1)/2 to p-1: sine components for k=1..(p-1)/2
    """
    x = torch.arange(p, dtype=torch.float32, device=device)  # [p]
    n_freq = (p - 1) // 2

    # Initialize basis matrix
    B = torch.zeros(p, p, dtype=torch.float32, device=device)

    # DC component (row 0)
    B[0, :] = 1.0 / math.sqrt(p)

    # Cosine and sine components
    for k in range(1, n_freq + 1):
        # Cosine: row k
        B[k, :] = math.sqrt(2.0 / p) * torch.cos(2 * math.pi * k * x / p)
        # Sine: row (n_freq + k)
        B[n_freq + k, :] = math.sqrt(2.0 / p) * torch.sin(2 * math.pi * k * x / p)

    return B


def project_rows_to_basis(
    W: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Project rows of W onto the Fourier basis.

    Args:
        W: Weight matrix of shape [p, d] where rows are indexed by x in Z_p.
        B: Fourier basis of shape [p, p] from real_fourier_basis().

    Returns:
        Coefficients C of shape [p, d] where C = B @ W.
    """
    return B @ W


def energy_per_basis(C: torch.Tensor) -> torch.Tensor:
    """Compute energy (squared norm) for each basis vector.

    Args:
        C: Coefficient matrix of shape [p, d] from project_rows_to_basis().

    Returns:
        Energy tensor of shape [p] where e[i] = sum_j C[i,j]^2.
    """
    return (C ** 2).sum(dim=-1)


def energy_per_frequency_group(
    e: torch.Tensor,
    p: int,
) -> dict[str, Any]:
    """Group energies by frequency.

    Args:
        e: Energy tensor of shape [p] from energy_per_basis().
        p: Prime modulus.

    Returns:
        Dictionary with grouped energies:
            - "dc": DC energy (scalar)
            - "cos_k": cosine energy for k=1..(p-1)/2
            - "sin_k": sine energy for k=1..(p-1)/2
            - "freq_k_total": combined cos+sin energy for each k
            - "freq_total": list of freq_k_total values for k=1..(p-1)/2
    """
    n_freq = (p - 1) // 2

    result = {
        "dc": e[0].item(),
    }

    freq_total = []
    for k in range(1, n_freq + 1):
        cos_k = e[k].item()
        sin_k = e[n_freq + k].item()
        result[f"cos_{k}"] = cos_k
        result[f"sin_{k}"] = sin_k
        result[f"freq_{k}_total"] = cos_k + sin_k
        freq_total.append(cos_k + sin_k)

    result["freq_total"] = freq_total

    return result


def compute_spectral_energy(
    W: torch.Tensor,
    p: int,
) -> dict[str, Any]:
    """Compute full spectral energy analysis for a weight matrix.

    Args:
        W: Weight matrix of shape [vocab_size, d] or [p, d].
        p: Prime modulus.

    Returns:
        Dictionary with spectral energy metrics.
    """
    # Extract number token rows if needed
    if W.shape[0] > p:
        W = W[:p, :]

    B = real_fourier_basis(p, device=W.device)
    C = project_rows_to_basis(W, B)
    e = energy_per_basis(C)

    return energy_per_frequency_group(e, p)


def get_topk_frequencies(
    freq_total: list[float],
    k: int,
) -> list[int]:
    """Get the k frequencies with highest total energy.

    Args:
        freq_total: List of frequency energies (index i = frequency i+1).
        k: Number of top frequencies to return.

    Returns:
        List of frequency indices (1-indexed) with highest energy.
    """
    # Create (energy, freq_index) pairs
    indexed = [(energy, i + 1) for i, energy in enumerate(freq_total)]
    # Sort by energy descending
    indexed.sort(reverse=True)
    # Return top k frequency indices
    return [freq for _, freq in indexed[:k]]
