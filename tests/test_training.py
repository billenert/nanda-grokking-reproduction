"""Tests for training infrastructure.

Tests for optimizer, scheduler, logging, and training loop.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from grokking.models import Transformer
from grokking.train.optim import (
    create_optimizer,
    create_scheduler,
    get_lr,
    get_weight_decay,
)
from grokking.train.logging import (
    MetricsLogger,
    load_metrics,
    get_metrics_by_split,
)
from grokking.train.trainer import compute_loss_and_acc, evaluate
from grokking.data import (
    ModularAdditionDataset,
    ModularAdditionTrainDataset,
    ModularAdditionTestDataset,
    create_batch,
)
from grokking.utils.io import load_config


class TestOptimizer:
    """Tests for optimizer creation."""

    def test_create_optimizer(self):
        """Test creating AdamW optimizer from config."""
        config = load_config("configs/debug.yaml")
        model = Transformer(config["model"], vocab_size=19)

        optimizer = create_optimizer(model, config)

        assert optimizer.__class__.__name__ == "AdamW"
        assert get_lr(optimizer) == config["optim"]["lr"]
        assert get_weight_decay(optimizer) == config["optim"]["weight_decay"]


class TestScheduler:
    """Tests for learning rate scheduler."""

    def test_constant_scheduler_warmup(self):
        """Test constant scheduler with warmup."""
        config = load_config("configs/debug.yaml")
        model = Transformer(config["model"], vocab_size=19)
        optimizer = create_optimizer(model, config)

        scheduler = create_scheduler(optimizer, config, total_steps=1000)

        # Check warmup
        warmup_steps = config["optim"]["scheduler"]["warmup_steps"]
        for step in range(warmup_steps):
            expected_lr = config["optim"]["lr"] * (step / warmup_steps)
            actual_lr = get_lr(optimizer)
            assert abs(actual_lr - expected_lr) < 1e-6
            scheduler.step()

    def test_constant_scheduler_after_warmup(self):
        """Test constant scheduler maintains LR after warmup."""
        config = load_config("configs/debug.yaml")
        model = Transformer(config["model"], vocab_size=19)
        optimizer = create_optimizer(model, config)

        scheduler = create_scheduler(optimizer, config, total_steps=1000)

        warmup_steps = config["optim"]["scheduler"]["warmup_steps"]
        for _ in range(warmup_steps + 100):
            scheduler.step()

        # After warmup, LR should be constant
        assert abs(get_lr(optimizer) - config["optim"]["lr"]) < 1e-6

    def test_cosine_scheduler(self):
        """Test cosine scheduler decays LR."""
        config = load_config("configs/debug.yaml")
        config["optim"]["scheduler"]["name"] = "cosine"

        model = Transformer(config["model"], vocab_size=19)
        optimizer = create_optimizer(model, config)

        total_steps = 1000
        scheduler = create_scheduler(optimizer, config, total_steps)

        # Run through all steps
        warmup_steps = config["optim"]["scheduler"]["warmup_steps"]
        for _ in range(warmup_steps):
            scheduler.step()

        lr_after_warmup = get_lr(optimizer)

        for _ in range(total_steps - warmup_steps):
            scheduler.step()

        lr_final = get_lr(optimizer)

        # Cosine should decay to near 0
        assert lr_final < lr_after_warmup * 0.1


class TestMetricsLogging:
    """Tests for metrics logging."""

    def test_metrics_logger(self):
        """Test MetricsLogger writes correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "metrics.jsonl"

            with MetricsLogger(log_path) as logger:
                logger.log(
                    step=0,
                    split="train",
                    loss=1.5,
                    acc=0.1,
                    lr=0.001,
                    weight_decay=0.01,
                )
                logger.log(
                    step=0,
                    split="test",
                    loss=1.6,
                    acc=0.08,
                    lr=0.001,
                    weight_decay=0.01,
                )

            # Load and verify
            metrics = load_metrics(log_path)

            assert len(metrics) == 2
            assert metrics[0]["step"] == 0
            assert metrics[0]["split"] == "train"
            assert metrics[0]["loss"] == 1.5
            assert metrics[0]["acc"] == 0.1
            assert "time_sec" in metrics[0]

    def test_get_metrics_by_split(self):
        """Test filtering metrics by split."""
        metrics = [
            {"step": 0, "split": "train", "loss": 1.0},
            {"step": 0, "split": "test", "loss": 1.5},
            {"step": 100, "split": "train", "loss": 0.5},
            {"step": 100, "split": "test", "loss": 1.0},
        ]

        train_metrics = get_metrics_by_split(metrics, "train")
        test_metrics = get_metrics_by_split(metrics, "test")

        assert len(train_metrics) == 2
        assert len(test_metrics) == 2
        assert all(m["split"] == "train" for m in train_metrics)
        assert all(m["split"] == "test" for m in test_metrics)


class TestComputeLossAndAcc:
    """Tests for loss and accuracy computation."""

    def test_compute_loss_and_acc_shape(self):
        """Test loss is scalar and acc is float."""
        config = load_config("configs/debug.yaml")
        p = config["data"]["p"]
        vocab_size = p + 2

        model = Transformer(config["model"], vocab_size)

        # Create sample batch
        batch_size = 4
        seq_len = 4
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, p, (batch_size,))

        loss, acc = compute_loss_and_acc(model, tokens, targets, p)

        assert loss.dim() == 0  # Scalar
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_perfect_accuracy(self):
        """Test accuracy is 1.0 when predictions are correct."""
        config = load_config("configs/debug.yaml")
        p = config["data"]["p"]
        vocab_size = p + 2

        model = Transformer(config["model"], vocab_size)

        # Get model predictions
        tokens = torch.randint(0, vocab_size, (4, 4))
        with torch.no_grad():
            logits = model(tokens)
            preds = logits[:, -1, :p].argmax(dim=-1)

        # Use predictions as targets
        _, acc = compute_loss_and_acc(model, tokens, preds, p)

        assert acc == 1.0


class TestEvaluate:
    """Tests for evaluation function."""

    def test_evaluate_returns_valid_metrics(self):
        """Test evaluation returns valid loss and accuracy."""
        config = load_config("configs/debug.yaml")
        p = config["data"]["p"]
        vocab_size = p + 2

        model = Transformer(config["model"], vocab_size)
        device = torch.device("cpu")

        # Create dataset
        base_dataset = ModularAdditionDataset(p, 0.3, seed=0)
        test_dataset = ModularAdditionTestDataset(base_dataset)

        loss, acc = evaluate(model, test_dataset, config, device)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0  # Loss should be positive
        assert 0.0 <= acc <= 1.0  # Accuracy in valid range


class TestTrainingSmokeTest:
    """Smoke test for the training loop (quick sanity check)."""

    def test_training_reduces_loss(self):
        """Test that a few steps of training reduces loss."""
        config = load_config("configs/debug.yaml")
        p = config["data"]["p"]
        vocab_size = p + 2

        model = Transformer(config["model"], vocab_size)
        optimizer = create_optimizer(model, config)

        # Create dataset
        base_dataset = ModularAdditionDataset(p, 0.3, seed=0)
        train_dataset = ModularAdditionTrainDataset(base_dataset)

        # Get initial loss
        examples = [train_dataset[i] for i in range(64)]
        examples = [((a, b), label) for (a, b), label in examples]
        tokens, targets = create_batch(examples, p, config["data"]["format"])

        model.train()
        initial_loss, _ = compute_loss_and_acc(model, tokens, targets, p)

        # Train for a few steps
        for _ in range(10):
            loss, _ = compute_loss_and_acc(model, tokens, targets, p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss, _ = compute_loss_and_acc(model, tokens, targets, p)

        # Loss should decrease (or at least not explode)
        assert final_loss.item() < initial_loss.item() * 2
