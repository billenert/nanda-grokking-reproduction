"""Mechanistic interpretability tools."""

from grokking.interp.fourier import (
    real_fourier_basis,
    project_rows_to_basis,
    energy_per_basis,
    energy_per_frequency_group,
    compute_spectral_energy,
    get_topk_frequencies,
)

from grokking.interp.hooks import (
    run_with_cache,
    get_cache_from_model,
)

from grokking.interp.ablations import (
    parse_frequency_set,
    ablate_embedding_frequencies,
    run_ablation,
)

from grokking.interp.progress_measures import (
    compute_topk_mass,
    compute_spectral_alignment,
    compute_progress_measures,
    compute_all_progress_measures,
)

__all__ = [
    # Fourier
    "real_fourier_basis",
    "project_rows_to_basis",
    "energy_per_basis",
    "energy_per_frequency_group",
    "compute_spectral_energy",
    "get_topk_frequencies",
    # Hooks
    "run_with_cache",
    "get_cache_from_model",
    # Ablations
    "parse_frequency_set",
    "ablate_embedding_frequencies",
    "run_ablation",
    # Progress measures
    "compute_topk_mass",
    "compute_spectral_alignment",
    "compute_progress_measures",
    "compute_all_progress_measures",
]
