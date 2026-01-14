"""Tests for utility functions."""

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from grokking.utils.seed import get_rng_state, set_rng_state, set_seed
from grokking.utils.device import get_device, get_dtype
from grokking.utils.io import (
    apply_overrides,
    load_config,
    load_jsonl,
    save_config,
    save_jsonl,
)
from grokking.utils.math import is_prime, next_prime


class TestSeeding:
    """Tests for seeding utilities."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible random numbers."""
        set_seed(42)
        rand1 = random.random()
        np1 = np.random.rand()
        torch1 = torch.rand(1).item()

        set_seed(42)
        rand2 = random.random()
        np2 = np.random.rand()
        torch2 = torch.rand(1).item()

        assert rand1 == rand2
        assert np1 == np2
        assert torch1 == torch2

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different random numbers."""
        set_seed(42)
        rand1 = random.random()

        set_seed(43)
        rand2 = random.random()

        assert rand1 != rand2

    def test_rng_state_save_restore(self):
        """Test that RNG state can be saved and restored."""
        set_seed(42)

        # Generate some random numbers to advance state
        random.random()
        np.random.rand()
        torch.rand(1)

        # Save state
        state = get_rng_state()

        # Generate more random numbers
        rand1 = random.random()
        np1 = np.random.rand()
        torch1 = torch.rand(1).item()

        # Restore state
        set_rng_state(state)

        # Should get the same numbers
        rand2 = random.random()
        np2 = np.random.rand()
        torch2 = torch.rand(1).item()

        assert rand1 == rand2
        assert np1 == np2
        assert torch1 == torch2


class TestDevice:
    """Tests for device utilities."""

    def test_get_device_cpu(self):
        """Test getting CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_invalid(self):
        """Test that invalid device raises error."""
        with pytest.raises(ValueError, match="Invalid device"):
            get_device("invalid")

    def test_get_dtype_float32(self):
        """Test getting float32 dtype."""
        dtype = get_dtype("float32")
        assert dtype == torch.float32

    def test_get_dtype_float16(self):
        """Test getting float16 dtype (only valid on CUDA)."""
        cpu_device = torch.device("cpu")
        with pytest.raises(ValueError, match="float16 is only supported on CUDA"):
            get_dtype("float16", device=cpu_device)

    def test_get_dtype_invalid(self):
        """Test that invalid dtype raises error."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            get_dtype("invalid")


class TestConfigIO:
    """Tests for configuration I/O."""

    def test_load_config(self):
        """Test loading a YAML config file."""
        config = load_config("configs/debug.yaml")

        assert "run" in config
        assert "data" in config
        assert "model" in config
        assert config["data"]["p"] == 17
        assert config["run"]["name"] == "debug"

    def test_load_config_not_found(self):
        """Test that loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent.yaml")

    def test_save_and_load_config(self):
        """Test saving and loading a config."""
        config = {"test": {"nested": {"value": 42}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            save_config(config, path)

            loaded = load_config(path)
            assert loaded == config

    def test_apply_overrides_simple(self):
        """Test applying simple overrides."""
        config = {"model": {"d_model": 128}}
        overrides = ["model.d_model=64"]

        result = apply_overrides(config, overrides)

        assert result["model"]["d_model"] == 64
        # Original should be unchanged
        assert config["model"]["d_model"] == 128

    def test_apply_overrides_types(self):
        """Test that overrides parse different types correctly."""
        config = {
            "int_val": 0,
            "float_val": 0.0,
            "bool_val": False,
            "none_val": "something",
            "str_val": "",
        }

        overrides = [
            "int_val=42",
            "float_val=3.14",
            "bool_val=true",
            "none_val=null",
            "str_val=hello",
        ]

        result = apply_overrides(config, overrides)

        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["none_val"] is None
        assert result["str_val"] == "hello"

    def test_apply_overrides_scientific_notation(self):
        """Test that scientific notation is parsed correctly."""
        config = {"lr": 0.0}
        overrides = ["lr=1e-3"]

        result = apply_overrides(config, overrides)

        assert result["lr"] == 1e-3

    def test_apply_overrides_list(self):
        """Test that list values are parsed correctly."""
        config = {"betas": [0.0, 0.0]}
        overrides = ["betas=[0.9, 0.98]"]

        result = apply_overrides(config, overrides)

        assert result["betas"] == [0.9, 0.98]

    def test_apply_overrides_invalid_format(self):
        """Test that invalid override format raises error."""
        config = {"value": 0}

        with pytest.raises(ValueError, match="Invalid override format"):
            apply_overrides(config, ["invalid_no_equals"])

    def test_apply_overrides_nonexistent_key(self):
        """Test that non-existent key raises error."""
        config = {"model": {"d_model": 128}}

        with pytest.raises(KeyError):
            apply_overrides(config, ["model.nonexistent=42"])


class TestJsonlIO:
    """Tests for JSONL I/O."""

    def test_save_and_load_jsonl(self):
        """Test saving and loading JSONL data."""
        data = [
            {"step": 0, "loss": 1.0},
            {"step": 1, "loss": 0.5},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            save_jsonl(data, path)

            loaded = load_jsonl(path)
            assert loaded == data

    def test_save_jsonl_appends(self):
        """Test that save_jsonl appends to existing file."""
        data1 = [{"step": 0}]
        data2 = [{"step": 1}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            save_jsonl(data1, path)
            save_jsonl(data2, path)

            loaded = load_jsonl(path)
            assert len(loaded) == 2
            assert loaded[0]["step"] == 0
            assert loaded[1]["step"] == 1


class TestMath:
    """Tests for math utilities."""

    def test_is_prime(self):
        """Test prime number checking."""
        primes = [2, 3, 5, 7, 11, 13, 17, 113]
        non_primes = [0, 1, 4, 6, 8, 9, 10, 100]

        for p in primes:
            assert is_prime(p), f"{p} should be prime"

        for n in non_primes:
            assert not is_prime(n), f"{n} should not be prime"

    def test_next_prime(self):
        """Test finding next prime."""
        assert next_prime(10) == 11
        assert next_prime(11) == 11
        assert next_prime(12) == 13
        assert next_prime(100) == 101


class TestConfigSchema:
    """Tests to verify config files have correct schema."""

    def test_debug_config_has_all_keys(self):
        """Test that debug config has all required keys."""
        config = load_config("configs/debug.yaml")

        # Check top-level keys
        required_keys = ["run", "data", "model", "train", "optim", "analysis", "interp"]
        for key in required_keys:
            assert key in config, f"Missing top-level key: {key}"

        # Check run keys
        run_keys = ["name", "seed", "out_dir", "device", "dtype"]
        for key in run_keys:
            assert key in config["run"], f"Missing run key: {key}"

        # Check data keys
        data_keys = ["p", "train_fraction", "format"]
        for key in data_keys:
            assert key in config["data"], f"Missing data key: {key}"

        # Check model keys
        model_keys = [
            "d_model", "n_layers", "n_heads", "d_mlp", "dropout",
            "layer_norm_eps", "tie_embeddings", "positional_encoding",
            "max_seq_len", "mlp_activation"
        ]
        for key in model_keys:
            assert key in config["model"], f"Missing model key: {key}"

    def test_canonical_config_has_all_keys(self):
        """Test that canonical config has all required keys."""
        config = load_config("configs/canonical.yaml")

        # Same checks as debug
        required_keys = ["run", "data", "model", "train", "optim", "analysis", "interp"]
        for key in required_keys:
            assert key in config, f"Missing top-level key: {key}"

    def test_debug_config_values(self):
        """Test that debug config has expected values for fast iteration."""
        config = load_config("configs/debug.yaml")

        assert config["data"]["p"] == 17, "Debug should use small prime"
        assert config["train"]["steps"] == 1000, "Debug should have short run"
        assert config["run"]["device"] == "cpu", "Debug should use CPU"
