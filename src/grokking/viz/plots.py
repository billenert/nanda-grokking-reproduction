"""Visualization functions.

- Grokking curve (train/test accuracy vs step)
- Fourier spectra
- Ablation results
- Progress measures

All plots: PNG format, 150 DPI minimum, include title/labels/legend.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_grokking_curve(
    metrics_path: str | Path,
    out_path: str | Path,
    dpi: int = 150,
) -> None:
    """Plot the grokking curve (train/test accuracy and loss vs step).

    Args:
        metrics_path: Path to metrics.jsonl file.
        out_path: Path to save the plot.
        dpi: DPI for the saved image.
    """
    metrics_path = Path(metrics_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metrics
    train_data = {"step": [], "acc": [], "loss": []}
    test_data = {"step": [], "acc": [], "loss": []}

    with open(metrics_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry["split"] == "train":
                train_data["step"].append(entry["step"])
                train_data["acc"].append(entry["acc"])
                train_data["loss"].append(entry["loss"])
            else:
                test_data["step"].append(entry["step"])
                test_data["acc"].append(entry["acc"])
                test_data["loss"].append(entry["loss"])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    ax1.plot(train_data["step"], train_data["acc"], label="Train", color="blue")
    ax1.plot(test_data["step"], test_data["acc"], label="Test", color="orange")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Grokking Curve - Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Loss plot
    ax2.plot(train_data["step"], train_data["loss"], label="Train", color="blue")
    ax2.plot(test_data["step"], test_data["loss"], label="Test", color="orange")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Grokking Curve - Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_fourier_spectra(
    fourier_metrics_path: str | Path,
    out_path: str | Path,
    dpi: int = 150,
) -> None:
    """Plot Fourier spectral energy over training.

    Args:
        fourier_metrics_path: Path to fourier_metrics.json file.
        out_path: Path to save the plot.
        dpi: DPI for the saved image.
    """
    fourier_metrics_path = Path(fourier_metrics_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fourier_metrics_path, "r") as f:
        data = json.load(f)

    p = data["p"]
    checkpoints = data["checkpoints"]

    # Extract steps and frequency energies
    steps = [cp["step"] for cp in checkpoints]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot embedding DC energy over time
    embed_dc = [cp["embedding"]["dc"] for cp in checkpoints]
    axes[0, 0].plot(steps, embed_dc)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("DC Energy")
    axes[0, 0].set_title("Embedding DC Energy")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot unembedding DC energy over time
    unembed_dc = [cp["unembedding"]["dc"] for cp in checkpoints]
    axes[0, 1].plot(steps, unembed_dc)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("DC Energy")
    axes[0, 1].set_title("Unembedding DC Energy")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot frequency spectrum at last checkpoint
    if checkpoints:
        last_cp = checkpoints[-1]
        n_freqs = len(last_cp["embedding"]["freq_total"])
        freq_indices = np.arange(1, n_freqs + 1)

        axes[1, 0].bar(freq_indices, last_cp["embedding"]["freq_total"])
        axes[1, 0].set_xlabel("Frequency k")
        axes[1, 0].set_ylabel("Energy")
        axes[1, 0].set_title(f"Embedding Freq Energy (Step {last_cp['step']})")

        axes[1, 1].bar(freq_indices, last_cp["unembedding"]["freq_total"])
        axes[1, 1].set_xlabel("Frequency k")
        axes[1, 1].set_ylabel("Energy")
        axes[1, 1].set_title(f"Unembedding Freq Energy (Step {last_cp['step']})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_ablation(
    ablation_path: str | Path,
    out_path: str | Path,
    dpi: int = 150,
) -> None:
    """Plot ablation results.

    Args:
        ablation_path: Path to ablation JSON file.
        out_path: Path to save the plot.
        dpi: DPI for the saved image.
    """
    ablation_path = Path(ablation_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ablation_path, "r") as f:
        data = json.load(f)

    step = data["step"]
    acc_base = data["acc_base"]
    results = data["results"]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sets = [r["set"] for r in results]
    accs = [r["acc"] for r in results]

    x = np.arange(len(sets))
    bars = ax.bar(x, accs, color="steelblue")

    # Add baseline line
    ax.axhline(y=acc_base, color="red", linestyle="--", label=f"Baseline: {acc_base:.3f}")

    ax.set_xlabel("Ablated Frequency Set")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Fourier Ablation Results (Step {step})")
    ax.set_xticks(x)
    ax.set_xticklabels(sets, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_progress(
    progress_path: str | Path,
    out_path: str | Path,
    dpi: int = 150,
) -> None:
    """Plot progress measures alongside accuracy.

    Args:
        progress_path: Path to progress_measures.json file.
        out_path: Path to save the plot.
        dpi: DPI for the saved image.
    """
    progress_path = Path(progress_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(progress_path, "r") as f:
        data = json.load(f)

    checkpoints = data["checkpoints"]
    steps = [cp["step"] for cp in checkpoints]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Test accuracy
    test_acc = [cp["acc_test"] for cp in checkpoints]
    axes[0, 0].plot(steps, test_acc, color="blue")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Test Accuracy")
    axes[0, 0].set_title("Test Accuracy")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.05, 1.05)

    # Embedding top-k mass
    if "embed_mass_topk_8" in checkpoints[0]:
        embed_mass = [cp["embed_mass_topk_8"] for cp in checkpoints]
        axes[0, 1].plot(steps, embed_mass, color="green")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Top-8 Mass")
        axes[0, 1].set_title("Embedding Top-8 Frequency Mass")
        axes[0, 1].grid(True, alpha=0.3)

    # Unembedding top-k mass
    if "unembed_mass_topk_8" in checkpoints[0]:
        unembed_mass = [cp["unembed_mass_topk_8"] for cp in checkpoints]
        axes[1, 0].plot(steps, unembed_mass, color="orange")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Top-8 Mass")
        axes[1, 0].set_title("Unembedding Top-8 Frequency Mass")
        axes[1, 0].grid(True, alpha=0.3)

    # Spectral alignment
    if "spec_align" in checkpoints[0]:
        spec_align = [cp["spec_align"] for cp in checkpoints]
        axes[1, 1].plot(steps, spec_align, color="purple")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Cosine Similarity")
        axes[1, 1].set_title("Embedding-Unembedding Spectral Alignment")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
