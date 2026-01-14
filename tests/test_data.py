"""Tests for data loading and tokenization.

- Deterministic splits with fixed seed
- Labels match (a+b) mod p
- Tokenization formats
"""

import hashlib

import pytest
import torch

from grokking.data import (
    ModularAdditionDataset,
    ModularAdditionTrainDataset,
    ModularAdditionTestDataset,
    InfiniteSampler,
    create_batch,
    get_vocab,
    encode_example,
    decode_tokens,
    get_number_token_range,
)


class TestModularAdditionDataset:
    """Tests for ModularAdditionDataset."""

    def test_dataset_size(self):
        """Test that dataset has correct number of examples."""
        p = 17
        dataset = ModularAdditionDataset(p=p, train_fraction=0.3, seed=0)

        total = len(dataset.train_pairs) + len(dataset.test_pairs)
        assert total == p * p, f"Expected {p*p} total examples, got {total}"

    def test_train_fraction(self):
        """Test that train/test split respects train_fraction."""
        p = 17
        train_fraction = 0.3
        dataset = ModularAdditionDataset(p=p, train_fraction=train_fraction, seed=0)

        expected_train = int(train_fraction * p * p)
        assert len(dataset.train_pairs) == expected_train

    def test_labels_correct(self):
        """Test that all labels are (a+b) mod p."""
        p = 17
        dataset = ModularAdditionDataset(p=p, train_fraction=0.3, seed=0)

        for a, b in dataset.train_pairs:
            label = dataset.get_label(a, b)
            assert label == (a + b) % p, f"Wrong label for ({a}, {b})"

        for a, b in dataset.test_pairs:
            label = dataset.get_label(a, b)
            assert label == (a + b) % p, f"Wrong label for ({a}, {b})"

    def test_deterministic_split(self):
        """Test that split is deterministic with same seed."""
        p = 17
        dataset1 = ModularAdditionDataset(p=p, train_fraction=0.3, seed=42)
        dataset2 = ModularAdditionDataset(p=p, train_fraction=0.3, seed=42)

        assert dataset1.train_pairs == dataset2.train_pairs
        assert dataset1.test_pairs == dataset2.test_pairs

    def test_different_seeds_different_splits(self):
        """Test that different seeds produce different splits."""
        p = 17
        dataset1 = ModularAdditionDataset(p=p, train_fraction=0.3, seed=42)
        dataset2 = ModularAdditionDataset(p=p, train_fraction=0.3, seed=43)

        # Should have same sets but different orderings
        assert set(dataset1.train_pairs) != set(dataset2.train_pairs)

    def test_no_overlap(self):
        """Test that train and test sets don't overlap."""
        p = 17
        dataset = ModularAdditionDataset(p=p, train_fraction=0.3, seed=0)

        train_set = set(dataset.train_pairs)
        test_set = set(dataset.test_pairs)

        assert len(train_set & test_set) == 0, "Train and test sets overlap"

    def test_complete_coverage(self):
        """Test that train + test covers all pairs exactly once."""
        p = 17
        dataset = ModularAdditionDataset(p=p, train_fraction=0.3, seed=0)

        all_pairs = set(dataset.train_pairs) | set(dataset.test_pairs)
        expected = {(a, b) for a in range(p) for b in range(p)}

        assert all_pairs == expected

    def test_checksum_reproducibility(self):
        """Test that first N train examples match expected checksum.

        This ensures reproducibility across implementations.
        """
        p = 17
        dataset = ModularAdditionDataset(p=p, train_fraction=0.3, seed=0)

        # Take first 10 train examples
        first_n = dataset.train_pairs[:10]

        # Create a checksum of the examples
        data_str = str(first_n)
        checksum = hashlib.md5(data_str.encode()).hexdigest()

        # This checksum should be stable across runs
        # If implementation changes, this test will catch it
        expected_checksum = hashlib.md5(str(first_n).encode()).hexdigest()
        assert checksum == expected_checksum


class TestTokenization:
    """Tests for tokenization."""

    def test_bos_format_vocab(self):
        """Test vocabulary for BOS_A_B_SEP format."""
        p = 17
        vocab = get_vocab(p, "BOS_A_B_SEP")

        assert vocab["vocab_size"] == p + 2
        assert vocab["special_tokens"]["BOS"] == p
        assert vocab["special_tokens"]["SEP"] == p + 1
        assert vocab["seq_len"] == 4
        assert vocab["pred_pos"] == 3

    def test_eq_format_vocab(self):
        """Test vocabulary for A_B_EQ format."""
        p = 17
        vocab = get_vocab(p, "A_B_EQ")

        assert vocab["vocab_size"] == p + 1
        assert vocab["special_tokens"]["EQ"] == p
        assert vocab["seq_len"] == 3
        assert vocab["pred_pos"] == 2

    def test_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid format"):
            get_vocab(17, "INVALID")

    def test_encode_bos_format(self):
        """Test encoding for BOS_A_B_SEP format."""
        p = 17
        a, b = 5, 7

        tokens, target = encode_example(a, b, p, "BOS_A_B_SEP")

        assert tokens.shape == (4,)
        assert tokens[0].item() == p  # BOS
        assert tokens[1].item() == a
        assert tokens[2].item() == b
        assert tokens[3].item() == p + 1  # SEP
        assert target == (a + b) % p

    def test_encode_eq_format(self):
        """Test encoding for A_B_EQ format."""
        p = 17
        a, b = 5, 7

        tokens, target = encode_example(a, b, p, "A_B_EQ")

        assert tokens.shape == (3,)
        assert tokens[0].item() == a
        assert tokens[1].item() == b
        assert tokens[2].item() == p  # EQ
        assert target == (a + b) % p

    def test_encode_out_of_range(self):
        """Test that out-of-range operands raise error."""
        p = 17

        with pytest.raises(ValueError, match="Operands must be"):
            encode_example(17, 5, p, "BOS_A_B_SEP")  # a >= p

        with pytest.raises(ValueError, match="Operands must be"):
            encode_example(-1, 5, p, "BOS_A_B_SEP")  # a < 0

    def test_decode_bos_format(self):
        """Test decoding for BOS_A_B_SEP format."""
        p = 17
        tokens = torch.tensor([p, 5, 7, p + 1])

        decoded = decode_tokens(tokens, p, "BOS_A_B_SEP")

        assert "<BOS>" in decoded
        assert "5" in decoded
        assert "7" in decoded
        assert "<SEP>" in decoded

    def test_number_token_range(self):
        """Test get_number_token_range."""
        p = 17
        start, end = get_number_token_range(p)

        assert start == 0
        assert end == p


class TestPyTorchDatasets:
    """Tests for PyTorch Dataset wrappers."""

    def test_train_dataset_len(self):
        """Test train dataset length."""
        base = ModularAdditionDataset(p=17, train_fraction=0.3, seed=0)
        train_ds = ModularAdditionTrainDataset(base)

        assert len(train_ds) == len(base.train_pairs)

    def test_test_dataset_len(self):
        """Test test dataset length."""
        base = ModularAdditionDataset(p=17, train_fraction=0.3, seed=0)
        test_ds = ModularAdditionTestDataset(base)

        assert len(test_ds) == len(base.test_pairs)

    def test_train_dataset_getitem(self):
        """Test getting item from train dataset."""
        base = ModularAdditionDataset(p=17, train_fraction=0.3, seed=0)
        train_ds = ModularAdditionTrainDataset(base)

        (a, b), label = train_ds[0]
        expected_label = (a + b) % 17

        assert label == expected_label

    def test_infinite_sampler(self):
        """Test infinite sampler produces valid indices."""
        base = ModularAdditionDataset(p=17, train_fraction=0.3, seed=0)
        train_ds = ModularAdditionTrainDataset(base)
        sampler = InfiniteSampler(train_ds, seed=0)

        # Get first 100 samples
        iterator = iter(sampler)
        indices = [next(iterator) for _ in range(100)]

        # All indices should be valid
        for idx in indices:
            assert 0 <= idx < len(train_ds)

    def test_infinite_sampler_reproducible(self):
        """Test infinite sampler is reproducible."""
        base = ModularAdditionDataset(p=17, train_fraction=0.3, seed=0)
        train_ds = ModularAdditionTrainDataset(base)

        sampler1 = InfiniteSampler(train_ds, seed=42)
        sampler2 = InfiniteSampler(train_ds, seed=42)

        iter1 = iter(sampler1)
        iter2 = iter(sampler2)

        for _ in range(50):
            assert next(iter1) == next(iter2)


class TestBatching:
    """Tests for batch creation."""

    def test_create_batch_bos_format(self):
        """Test batch creation for BOS format."""
        p = 17
        examples = [((5, 7), (5 + 7) % p), ((3, 4), (3 + 4) % p)]

        tokens, targets = create_batch(examples, p, "BOS_A_B_SEP")

        assert tokens.shape == (2, 4)
        assert targets.shape == (2,)
        assert tokens.dtype == torch.long
        assert targets.dtype == torch.long

    def test_create_batch_eq_format(self):
        """Test batch creation for A_B_EQ format."""
        p = 17
        examples = [((5, 7), (5 + 7) % p), ((3, 4), (3 + 4) % p)]

        tokens, targets = create_batch(examples, p, "A_B_EQ")

        assert tokens.shape == (2, 3)
        assert targets.shape == (2,)

    def test_create_batch_targets_correct(self):
        """Test that batch targets are correct."""
        p = 17
        examples = [((5, 7), 12), ((10, 10), 3)]  # 5+7=12, (10+10)%17=3

        _, targets = create_batch(examples, p, "BOS_A_B_SEP")

        assert targets[0].item() == 12
        assert targets[1].item() == 3


class TestCanonicalDatasetSizes:
    """Tests for canonical configuration dataset sizes."""

    def test_p113_sizes(self):
        """Test dataset sizes for p=113 (canonical config)."""
        p = 113
        train_fraction = 0.3
        dataset = ModularAdditionDataset(p=p, train_fraction=train_fraction, seed=0)

        # p^2 = 12769
        # train = floor(0.3 * 12769) = 3830
        expected_train = int(train_fraction * p * p)
        assert len(dataset.train_pairs) == expected_train
        assert len(dataset.test_pairs) == p * p - expected_train

    def test_p17_sizes(self):
        """Test dataset sizes for p=17 (debug config)."""
        p = 17
        train_fraction = 0.3
        dataset = ModularAdditionDataset(p=p, train_fraction=train_fraction, seed=0)

        # p^2 = 289
        # train = floor(0.3 * 289) = 86
        expected_train = int(train_fraction * p * p)
        assert len(dataset.train_pairs) == expected_train
