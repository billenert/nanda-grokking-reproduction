#!/usr/bin/env python
"""Analyze Fourier spectra of model checkpoints.

Computes spectral energy of embedding and unembedding matrices in the
real Fourier basis, tracking how frequency concentration evolves.

Usage:
    python scripts/analyze_fourier.py --run_dir results/runs/<run_dir>
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from grokking.interp.fourier import compute_spectral_energy
from grokking.models import Transformer
from grokking.viz.plots import plot_fourier_spectra


def load_checkpoint(ckpt_path: Path, config: dict, device: str = "cpu") -> tuple:
    """Load a model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Tuple of (model, step).
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine vocab size from config
    p = config["data"]["p"]
    data_format = config["data"]["format"]
    if data_format == "BOS_A_B_SEP":
        vocab_size = p + 2
    else:
        vocab_size = p + 1

    # Create model
    model = Transformer(config["model"], vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["step"]


def analyze_checkpoint(model, p: int) -> dict:
    """Compute Fourier spectral energy for a model.

    Args:
        model: Transformer model.
        p: Prime modulus.

    Returns:
        Dictionary with embedding and unembedding spectral energies.
    """
    # Get embedding weights (number tokens only)
    E = model.get_embedding_weights()[:p, :].detach()
    embed_energy = compute_spectral_energy(E, p)

    # Get unembedding weights (number tokens only)
    U = model.get_unembedding_weights()[:p, :].detach()
    unembed_energy = compute_spectral_energy(U, p)

    return {
        "embedding": {
            "freq_total": embed_energy["freq_total"],
            "dc": embed_energy["dc"],
        },
        "unembedding": {
            "freq_total": unembed_energy["freq_total"],
            "dc": unembed_energy["dc"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Fourier spectra")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for analysis",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # Load config
    config_path = run_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    p = config["data"]["p"]
    analysis_config = config.get("analysis", {})
    checkpoints_spec = analysis_config.get("checkpoints", "all")

    # Find checkpoints
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoints directory not found: {ckpt_dir}")
        return

    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpt_files:
        print(f"ERROR: No checkpoint files found in {ckpt_dir}")
        return

    # Filter checkpoints if needed
    if checkpoints_spec != "all" and isinstance(checkpoints_spec, list):
        ckpt_files = [
            f for f in ckpt_files
            if int(f.stem.split("_")[1]) in checkpoints_spec
        ]

    print(f"Analyzing {len(ckpt_files)} checkpoints...")

    # Analyze each checkpoint
    results = {
        "p": p,
        "basis": "real_fourier_v1",
        "checkpoints": [],
    }

    for ckpt_path in ckpt_files:
        model, step = load_checkpoint(ckpt_path, config, args.device)
        analysis = analyze_checkpoint(model, p)

        results["checkpoints"].append({
            "step": step,
            **analysis,
        })

        print(f"  Step {step}: analyzed")

    # Save results
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    output_path = analysis_dir / "fourier_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_path}")

    # Generate plot
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_path = figures_dir / "fourier_spectra.png"
    plot_fourier_spectra(output_path, plot_path)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
