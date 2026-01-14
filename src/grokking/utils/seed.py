"""Deterministic seeding utilities.

- random.seed(seed)
- numpy.random.seed(seed)
- torch.manual_seed(seed)
- torch.cuda.manual_seed_all(seed) (if CUDA available)
- torch.backends.cudnn.deterministic = True
- torch.backends.cudnn.benchmark = False
"""

import random
import warnings
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all random number generators.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Warn about potential non-determinism
    if torch.cuda.is_available():
        # Some CUDA operations are non-deterministic
        warnings.warn(
            "WARNING: full determinism not guaranteed on this platform. "
            "Some CUDA operations may be non-deterministic.",
            stacklevel=2
        )


def get_rng_state() -> dict[str, Any]:
    """Capture the current RNG state for all random number generators.

    Returns:
        Dictionary containing RNG states for python, numpy, torch, and cuda.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state from a captured state dictionary.

    Args:
        state: Dictionary containing RNG states (from get_rng_state).
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
