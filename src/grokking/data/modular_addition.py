"""Modular addition dataset.

- Universe U = {(a,b) | a in [0..p-1], b in [0..p-1]}
- Label c = (a + b) mod p
- Deterministic train/test split based on seed
"""

import random
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler


class ModularAdditionDataset:
    """Dataset for modular addition task.

    Given prime modulus p, generates all pairs (a, b) where a, b in [0, p-1]
    and label c = (a + b) mod p.

    The dataset is split into train and test sets deterministically based on
    the seed and train_fraction parameters.
    """

    def __init__(
        self,
        p: int,
        train_fraction: float,
        seed: int = 0,
    ):
        """Initialize the modular addition dataset.

        Args:
            p: Prime modulus.
            train_fraction: Fraction of data to use for training.
            seed: Random seed for deterministic split.
        """
        self.p = p
        self.train_fraction = train_fraction
        self.seed = seed

        # Generate all pairs
        all_pairs = [(a, b) for a in range(p) for b in range(p)]

        # Deterministic shuffle based on seed
        rng = random.Random(seed)
        rng.shuffle(all_pairs)

        # Split into train/test
        n_train = int(train_fraction * len(all_pairs))
        self.train_pairs = all_pairs[:n_train]
        self.test_pairs = all_pairs[n_train:]

    def get_label(self, a: int, b: int) -> int:
        """Compute the label for a given pair.

        Args:
            a: First operand.
            b: Second operand.

        Returns:
            (a + b) mod p
        """
        return (a + b) % self.p


class ModularAdditionTrainDataset(Dataset):
    """PyTorch Dataset for training on modular addition."""

    def __init__(self, base_dataset: ModularAdditionDataset):
        """Initialize training dataset.

        Args:
            base_dataset: Base ModularAdditionDataset with train/test split.
        """
        self.base = base_dataset
        self.pairs = base_dataset.train_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[tuple[int, int], int]:
        """Get a training example.

        Args:
            idx: Index into the training set.

        Returns:
            Tuple of ((a, b), label).
        """
        a, b = self.pairs[idx]
        label = self.base.get_label(a, b)
        return (a, b), label


class ModularAdditionTestDataset(Dataset):
    """PyTorch Dataset for testing on modular addition."""

    def __init__(self, base_dataset: ModularAdditionDataset):
        """Initialize test dataset.

        Args:
            base_dataset: Base ModularAdditionDataset with train/test split.
        """
        self.base = base_dataset
        self.pairs = base_dataset.test_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[tuple[int, int], int]:
        """Get a test example.

        Args:
            idx: Index into the test set.

        Returns:
            Tuple of ((a, b), label).
        """
        a, b = self.pairs[idx]
        label = self.base.get_label(a, b)
        return (a, b), label


class InfiniteSampler(Sampler[int]):
    """Sampler that samples uniformly with replacement from a dataset.

    Used for training to sample infinitely with replacement.
    """

    def __init__(self, data_source: Dataset, seed: int = 0):
        """Initialize the sampler.

        Args:
            data_source: Dataset to sample from.
            seed: Random seed for reproducibility.
        """
        self.data_source = data_source
        self.seed = seed
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[int]:
        """Yield random indices indefinitely."""
        n = len(self.data_source)
        while True:
            yield self._rng.randint(0, n - 1)

    def __len__(self) -> int:
        """Return a large number since this is infinite."""
        return 2**62  # Effectively infinite


def create_batch(
    examples: list[tuple[tuple[int, int], int]],
    p: int,
    format: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a batch of tokenized examples.

    Args:
        examples: List of ((a, b), label) tuples.
        p: Prime modulus.
        format: Token format ("BOS_A_B_SEP" or "A_B_EQ").

    Returns:
        Tuple of (tokens, targets) tensors.
    """
    from grokking.data.tokenization import encode_example

    tokens_list = []
    targets_list = []

    for (a, b), label in examples:
        tokens, target = encode_example(a, b, p, format)
        tokens_list.append(tokens)
        targets_list.append(target)

    tokens = torch.stack(tokens_list)
    targets = torch.tensor(targets_list, dtype=torch.long)

    return tokens, targets
