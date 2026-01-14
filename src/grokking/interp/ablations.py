"""Fourier ablation utilities.

- Project embeddings to Fourier space
- Zero out specified frequency components
- Reconstruct and evaluate accuracy
"""

import copy
from typing import Any

import torch
import torch.nn as nn

from grokking.interp.fourier import (
    real_fourier_basis,
    compute_spectral_energy,
    get_topk_frequencies,
)


def parse_frequency_set(
    spec: dict[str, Any],
    p: int,
    freq_energies: list[float] | None = None,
) -> list[int]:
    """Parse a frequency set specification.

    Args:
        spec: Dictionary with 'type' and 'k' keys.
        p: Prime modulus.
        freq_energies: Optional list of frequency energies (for topk).

    Returns:
        List of frequency indices to ablate (1-indexed, or 0 for DC).
    """
    set_type = spec["type"]
    k = spec.get("k", 1)

    n_freq = (p - 1) // 2

    if set_type == "single":
        # Ablate single frequency k
        return [k]

    elif set_type == "topk":
        # Ablate top k frequencies by energy
        if freq_energies is None:
            raise ValueError("freq_energies required for topk ablation")
        return get_topk_frequencies(freq_energies, k)

    elif set_type == "range":
        # Ablate frequencies in range [k, end]
        end = spec.get("end", n_freq)
        return list(range(k, end + 1))

    elif set_type == "dc":
        # Ablate DC component
        return [0]

    else:
        raise ValueError(f"Unknown frequency set type: {set_type}")


def ablate_embedding_frequencies(
    model: nn.Module,
    freq_set: list[int],
    p: int,
) -> None:
    """Ablate specified frequencies from embedding weights (in-place).

    Args:
        model: Transformer model (modified in-place).
        freq_set: List of frequency indices to zero out (0=DC, 1..(p-1)/2 for others).
        p: Prime modulus.
    """
    n_freq = (p - 1) // 2

    # Get embedding weights
    E = model.token_embedding.weight.data  # [vocab_size, d_model]
    E_num = E[:p, :].clone()  # Number token embeddings [p, d_model]

    # Construct Fourier basis
    B = real_fourier_basis(p, device=E.device)

    # Project to Fourier space
    C = B @ E_num  # [p, d_model]

    # Zero out specified frequencies
    for freq in freq_set:
        if freq == 0:
            # DC component
            C[0, :] = 0
        elif 1 <= freq <= n_freq:
            # Cosine component at row freq
            C[freq, :] = 0
            # Sine component at row (n_freq + freq)
            C[n_freq + freq, :] = 0

    # Reconstruct embeddings
    E_num_abl = B.T @ C  # [p, d_model]

    # Write back to model
    E[:p, :] = E_num_abl


def run_ablation(
    model: nn.Module,
    freq_specs: list[dict[str, Any]],
    evaluate_fn: callable,
    p: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Run Fourier ablation experiment.

    Args:
        model: Transformer model.
        freq_specs: List of frequency set specifications.
        evaluate_fn: Function that takes model and returns accuracy.
        p: Prime modulus.

    Returns:
        Tuple of (baseline_accuracy, ablation_results).
    """
    # Compute baseline accuracy
    acc_base = evaluate_fn(model)

    # Compute frequency energies for topk
    E = model.token_embedding.weight.data[:p, :]
    energy = compute_spectral_energy(E, p)
    freq_energies = energy["freq_total"]

    results = []

    for spec in freq_specs:
        # Create a copy of the model
        model_copy = copy.deepcopy(model)

        # Parse frequency set
        freq_set = parse_frequency_set(spec, p, freq_energies)

        # Ablate
        ablate_embedding_frequencies(model_copy, freq_set, p)

        # Evaluate
        acc_abl = evaluate_fn(model_copy)

        # Format set name
        set_type = spec["type"]
        if set_type == "single":
            set_name = f"single({spec['k']})"
        elif set_type == "topk":
            set_name = f"topk({spec['k']})"
        elif set_type == "range":
            set_name = f"range({spec['k']},{spec.get('end', (p-1)//2)})"
        elif set_type == "dc":
            set_name = "dc"
        else:
            set_name = str(spec)

        results.append({
            "set": set_name,
            "acc": acc_abl,
            "freq_indices": freq_set,
        })

    return acc_base, results
