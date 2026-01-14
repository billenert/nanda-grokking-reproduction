"""Training logging utilities.

Implements metrics logging to JSONL format.
"""

import json
import time
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Logger for training metrics to JSONL format.

    Each line in the output file is a JSON object with:
        - step: int
        - split: "train" | "test"
        - loss: float
        - acc: float
        - lr: float
        - weight_decay: float
        - time_sec: float
    """

    def __init__(self, log_path: str | Path):
        """Initialize the metrics logger.

        Args:
            log_path: Path to the JSONL log file.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self._file = open(self.log_path, "a")
        self._start_time = time.time()

    def log(
        self,
        step: int,
        split: str,
        loss: float,
        acc: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        """Log a metrics entry.

        Args:
            step: Training step.
            split: Data split ("train" or "test").
            loss: Loss value.
            acc: Accuracy value.
            lr: Current learning rate.
            weight_decay: Current weight decay.
        """
        entry = {
            "step": step,
            "split": split,
            "loss": loss,
            "acc": acc,
            "lr": lr,
            "weight_decay": weight_decay,
            "time_sec": time.time() - self._start_time,
        }

        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def load_metrics(log_path: str | Path) -> list[dict[str, Any]]:
    """Load metrics from a JSONL log file.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        List of metric dictionaries.
    """
    log_path = Path(log_path)
    metrics = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    return metrics


def get_metrics_by_split(
    metrics: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    """Filter metrics by split.

    Args:
        metrics: List of metric dictionaries.
        split: Split to filter by ("train" or "test").

    Returns:
        Filtered list of metrics.
    """
    return [m for m in metrics if m["split"] == split]


def get_metric_at_step(
    metrics: list[dict[str, Any]],
    step: int,
    split: str | None = None,
) -> dict[str, Any] | None:
    """Get metrics at a specific step.

    Args:
        metrics: List of metric dictionaries.
        step: Step to find.
        split: Optional split to filter by.

    Returns:
        Metrics dictionary at that step, or None if not found.
    """
    for m in metrics:
        if m["step"] == step:
            if split is None or m["split"] == split:
                return m
    return None
