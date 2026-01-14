"""Progress measures for tracking grokking.

- Embedding top-k mass: fraction of energy in top k frequencies
- Unembedding top-k mass: same for unembedding
- Spectral alignment: cosine similarity between embedding and unembedding spectra
"""

from typing import Any

import torch
import torch.nn as nn

from grokking.interp.fourier import compute_spectral_energy


def compute_topk_mass(
    freq_total: list[float],
    k: int,
) -> float:
    """Compute the fraction of energy in the top-k frequencies.

    Args:
        freq_total: List of frequency energies.
        k: Number of top frequencies.

    Returns:
        Ratio of top-k energy to total energy.
    """
    if not freq_total:
        return 0.0

    total_energy = sum(freq_total)
    if total_energy == 0:
        return 0.0

    sorted_energies = sorted(freq_total, reverse=True)
    topk_energy = sum(sorted_energies[:k])

    return topk_energy / total_energy


def compute_spectral_alignment(
    embed_freq: list[float],
    unembed_freq: list[float],
) -> float:
    """Compute cosine similarity between embedding and unembedding spectra.

    Args:
        embed_freq: Embedding frequency energies.
        unembed_freq: Unembedding frequency energies.

    Returns:
        Cosine similarity between the two spectra.
    """
    if not embed_freq or not unembed_freq:
        return 0.0

    # Convert to tensors
    pE = torch.tensor(embed_freq, dtype=torch.float32)
    pU = torch.tensor(unembed_freq, dtype=torch.float32)

    # Compute cosine similarity
    dot = torch.dot(pE, pU)
    norm_E = torch.norm(pE)
    norm_U = torch.norm(pU)

    if norm_E == 0 or norm_U == 0:
        return 0.0

    return (dot / (norm_E * norm_U)).item()


def compute_progress_measures(
    model: nn.Module,
    p: int,
    k: int = 8,
) -> dict[str, float]:
    """Compute all progress measures for a model.

    Args:
        model: Transformer model.
        p: Prime modulus.
        k: Number of top frequencies for mass computation.

    Returns:
        Dictionary with progress measures.
    """
    # Get embedding weights
    E = model.get_embedding_weights()[:p, :].detach()
    U = model.get_unembedding_weights()[:p, :].detach()

    # Compute spectral energies
    embed_energy = compute_spectral_energy(E, p)
    unembed_energy = compute_spectral_energy(U, p)

    embed_freq = embed_energy["freq_total"]
    unembed_freq = unembed_energy["freq_total"]

    # Compute measures
    measures = {
        f"embed_mass_topk_{k}": compute_topk_mass(embed_freq, k),
        f"unembed_mass_topk_{k}": compute_topk_mass(unembed_freq, k),
        "spec_align": compute_spectral_alignment(embed_freq, unembed_freq),
    }

    return measures


def compute_all_progress_measures(
    model: nn.Module,
    p: int,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute progress measures for multiple k values.

    Args:
        model: Transformer model.
        p: Prime modulus.
        k_values: List of k values for top-k mass (default [4, 8]).

    Returns:
        Dictionary with all progress measures.
    """
    if k_values is None:
        k_values = [4, 8]

    # Get embedding weights
    E = model.get_embedding_weights()[:p, :].detach()
    U = model.get_unembedding_weights()[:p, :].detach()

    # Compute spectral energies
    embed_energy = compute_spectral_energy(E, p)
    unembed_energy = compute_spectral_energy(U, p)

    embed_freq = embed_energy["freq_total"]
    unembed_freq = unembed_energy["freq_total"]

    measures = {}

    # Top-k mass for various k
    for k in k_values:
        measures[f"embed_mass_topk_{k}"] = compute_topk_mass(embed_freq, k)
        measures[f"unembed_mass_topk_{k}"] = compute_topk_mass(unembed_freq, k)

    # Spectral alignment
    measures["spec_align"] = compute_spectral_alignment(embed_freq, unembed_freq)

    return measures
