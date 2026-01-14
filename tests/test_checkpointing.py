"""Tests for checkpointing.

- Save/load checkpoints with correct format
- RNG state restoration
"""

import tempfile
from pathlib import Path

import pytest
import torch

from grokking.models import Transformer
from grokking.train.optim import create_optimizer
from grokking.train.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_for_eval,
    update_latest_checkpoint,
    get_checkpoint_steps,
    get_latest_checkpoint_path,
)
from grokking.utils.io import load_config
from grokking.utils.seed import set_seed


class TestCheckpointSaveLoad:
    """Tests for checkpoint save/load."""

    @pytest.fixture
    def model_and_optimizer(self):
        """Create a model and optimizer for testing."""
        config = load_config("configs/debug.yaml")
        vocab_size = 19  # p=17 + 2 special tokens
        model = Transformer(config["model"], vocab_size)
        optimizer = create_optimizer(model, config)
        return model, optimizer, config

    def test_save_and_load_model(self, model_and_optimizer):
        """Test saving and loading model state."""
        model, optimizer, config = model_and_optimizer

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"

            # Save checkpoint
            save_checkpoint(ckpt_path, step=100, model=model, optimizer=optimizer, config=config)

            # Create new model with same architecture
            vocab_size = 19
            new_model = Transformer(config["model"], vocab_size)
            new_optimizer = create_optimizer(new_model, config)

            # Verify weights are different before loading
            assert not torch.allclose(
                model.token_embedding.weight,
                new_model.token_embedding.weight
            )

            # Load checkpoint
            step, loaded_config = load_checkpoint(
                ckpt_path, new_model, new_optimizer, restore_rng=False
            )

            # Verify step and config
            assert step == 100
            assert loaded_config == config

            # Verify model weights are now identical
            for (name1, p1), (name2, p2) in zip(
                model.named_parameters(),
                new_model.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(p1, p2), f"Mismatch in {name1}"

    def test_checkpoint_format(self, model_and_optimizer):
        """Test that checkpoint contains required keys."""
        model, optimizer, config = model_and_optimizer

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            save_checkpoint(ckpt_path, step=100, model=model, optimizer=optimizer, config=config)

            checkpoint = torch.load(ckpt_path, weights_only=False)

            # Check required keys
            assert "step" in checkpoint
            assert "model_state_dict" in checkpoint
            assert "optim_state_dict" in checkpoint
            assert "config" in checkpoint
            assert "rng_state" in checkpoint

            # Check RNG state keys
            rng_state = checkpoint["rng_state"]
            assert "python" in rng_state
            assert "numpy" in rng_state
            assert "torch" in rng_state


class TestRNGRestoration:
    """Tests for RNG state restoration."""

    def test_rng_state_restored(self):
        """Test that RNG state is correctly restored from checkpoint."""
        config = load_config("configs/debug.yaml")
        vocab_size = 19
        model = Transformer(config["model"], vocab_size)
        optimizer = create_optimizer(model, config)

        # Set seed and generate some random numbers
        set_seed(42)
        _ = torch.randn(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            save_checkpoint(ckpt_path, step=100, model=model, optimizer=optimizer, config=config)

            # Generate more random numbers (advancing RNG state)
            rand1 = torch.randn(5)

            # Restore RNG state from checkpoint
            new_model = Transformer(config["model"], vocab_size)
            load_checkpoint(ckpt_path, new_model, restore_rng=True)

            # Generate same random numbers
            rand2 = torch.randn(5)

            assert torch.allclose(rand1, rand2), "RNG state not restored correctly"

    def test_rng_restoration_produces_identical_tensor(self):
        """Test that restoring RNG produces identical next random tensor."""
        config = load_config("configs/debug.yaml")
        vocab_size = 19
        model = Transformer(config["model"], vocab_size)
        optimizer = create_optimizer(model, config)

        set_seed(123)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            save_checkpoint(ckpt_path, step=0, model=model, optimizer=optimizer, config=config)

            # Generate next random tensor
            expected = torch.randn(10, 10)

            # Restore and generate again
            load_checkpoint(ckpt_path, model, restore_rng=True)
            actual = torch.randn(10, 10)

            assert torch.allclose(expected, actual)


class TestLatestCheckpoint:
    """Tests for latest checkpoint management."""

    def test_update_latest_checkpoint(self):
        """Test updating latest.pt symlink/copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)

            # Create dummy checkpoint files
            (ckpt_dir / "step_100.pt").write_text("dummy")
            (ckpt_dir / "step_200.pt").write_text("dummy")

            # Update latest
            update_latest_checkpoint(ckpt_dir, "step_200.pt")

            latest_path = ckpt_dir / "latest.pt"
            assert latest_path.exists()

    def test_get_checkpoint_steps(self):
        """Test getting list of checkpoint steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)

            # Create checkpoint files
            (ckpt_dir / "step_0.pt").write_text("dummy")
            (ckpt_dir / "step_100.pt").write_text("dummy")
            (ckpt_dir / "step_500.pt").write_text("dummy")
            (ckpt_dir / "latest.pt").write_text("dummy")  # Should be ignored

            steps = get_checkpoint_steps(ckpt_dir)

            assert steps == [0, 100, 500]

    def test_get_latest_checkpoint_path(self):
        """Test getting path to latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)

            # Create checkpoint files
            (ckpt_dir / "step_100.pt").write_text("dummy")
            (ckpt_dir / "step_500.pt").write_text("dummy")

            # Without latest.pt symlink, should return highest step
            path = get_latest_checkpoint_path(ckpt_dir)
            assert path == ckpt_dir / "step_500.pt"


class TestLoadForEval:
    """Tests for evaluation-only loading."""

    def test_load_for_eval_no_rng_restore(self):
        """Test that load_for_eval doesn't restore RNG."""
        config = load_config("configs/debug.yaml")
        vocab_size = 19
        model = Transformer(config["model"], vocab_size)
        optimizer = create_optimizer(model, config)

        # Save checkpoint with seed=42 RNG state
        set_seed(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            save_checkpoint(ckpt_path, step=100, model=model, optimizer=optimizer, config=config)

            # Set up a known RNG state by creating model (which advances RNG during init)
            set_seed(999)
            new_model = Transformer(config["model"], vocab_size)

            # Record current state - RNG has been advanced by model init
            expected_next = torch.randn(5).clone()

            # Reset to same state and recreate model
            set_seed(999)
            new_model = Transformer(config["model"], vocab_size)

            # Load for eval (should NOT restore RNG to seed 42 state)
            load_checkpoint_for_eval(ckpt_path, new_model)

            # Generate next random - should match expected_next if RNG wasn't restored
            actual_next = torch.randn(5)

            assert torch.allclose(actual_next, expected_next), \
                "RNG was unexpectedly restored by load_checkpoint_for_eval"
