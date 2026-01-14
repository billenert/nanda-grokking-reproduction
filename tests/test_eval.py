"""Tests for evaluation metrics and checkpoint evaluation."""

import tempfile
from pathlib import Path

import pytest
import torch

from grokking.eval.metrics import compute_accuracy, compute_loss, evaluate_model
from grokking.models import Transformer
from grokking.utils.io import load_config


class TestMetrics:
    """Tests for evaluation metrics."""

    @pytest.fixture
    def model_and_data(self):
        """Create model and sample data for testing."""
        config = load_config("configs/debug.yaml")
        p = config["data"]["p"]
        vocab_size = p + 2

        model = Transformer(config["model"], vocab_size)

        # Create sample batch
        batch_size = 8
        seq_len = 4
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, p, (batch_size,))

        return model, tokens, targets, p

    def test_compute_accuracy_range(self, model_and_data):
        """Test accuracy is in valid range."""
        model, tokens, targets, p = model_and_data

        with torch.no_grad():
            logits = model(tokens)

        acc = compute_accuracy(logits, targets, p)

        assert 0.0 <= acc <= 1.0

    def test_compute_accuracy_perfect(self, model_and_data):
        """Test accuracy is 1.0 for perfect predictions."""
        model, tokens, targets, p = model_and_data

        with torch.no_grad():
            logits = model(tokens)
            # Get model's predictions
            preds = logits[:, -1, :p].argmax(dim=-1)

        # Compute accuracy with predictions as targets
        acc = compute_accuracy(logits, preds, p)

        assert acc == 1.0

    def test_compute_loss_positive(self, model_and_data):
        """Test loss is positive."""
        model, tokens, targets, p = model_and_data

        with torch.no_grad():
            logits = model(tokens)

        loss = compute_loss(logits, targets, p)

        assert loss.item() > 0

    def test_compute_loss_reduction(self, model_and_data):
        """Test loss reduction options."""
        model, tokens, targets, p = model_and_data

        with torch.no_grad():
            logits = model(tokens)

        loss_mean = compute_loss(logits, targets, p, reduction="mean")
        loss_sum = compute_loss(logits, targets, p, reduction="sum")
        loss_none = compute_loss(logits, targets, p, reduction="none")

        assert loss_mean.dim() == 0  # Scalar
        assert loss_sum.dim() == 0  # Scalar
        assert loss_none.dim() == 1  # Per-sample
        assert loss_none.shape[0] == targets.shape[0]

    def test_evaluate_model(self, model_and_data):
        """Test evaluate_model returns correct format."""
        model, tokens, targets, p = model_and_data

        result = evaluate_model(model, tokens, targets, p)

        assert "loss" in result
        assert "accuracy" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["accuracy"], float)
        assert result["loss"] > 0
        assert 0.0 <= result["accuracy"] <= 1.0


class TestPlotFunctions:
    """Tests for plotting functions."""

    def test_plot_grokking_curve_creates_file(self):
        """Test that plot_grokking_curve creates a PNG file."""
        from grokking.viz import plot_grokking_curve

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy metrics
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            with open(metrics_path, "w") as f:
                for step in [0, 100, 200]:
                    f.write(f'{{"step": {step}, "split": "train", "acc": 0.5, "loss": 1.0}}\n')
                    f.write(f'{{"step": {step}, "split": "test", "acc": 0.3, "loss": 1.5}}\n')

            out_path = Path(tmpdir) / "grokking_curve.png"
            plot_grokking_curve(metrics_path, out_path)

            assert out_path.exists()
            assert out_path.stat().st_size > 0  # File has content
