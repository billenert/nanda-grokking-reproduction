# Grokking Modular Addition — Reproduction

A reproduction of the **grokking** phenomenon on modular addition using a small decoder-only transformer trained from scratch, along with **Fourier-based mechanistic interpretability** analyses.

## What is Grokking?

Grokking is a phenomenon where neural networks suddenly generalize long after achieving near-perfect training accuracy. In this reproduction:

1. A transformer is trained on modular addition: `(a + b) mod p`
2. The model quickly memorizes the training data (high train accuracy, low test accuracy)
3. After continued training with weight decay, the model suddenly generalizes (test accuracy jumps)

The key ingredients for grokking are:
- **Limited training data**: Only ~30% of all possible pairs
- **Weight decay**: Regularization that eventually forces generalization
- **Extended training**: Many more steps than needed to fit the training data

## Quickstart

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd grokking

# Install dependencies
pip install -e .
```

### Run a Quick Debug Training

```bash
# Run with debug config (fast, ~1 minute on CPU)
python scripts/train.py --config configs/debug.yaml
```

### Run Canonical Reproduction

```bash
# Full reproduction with all analyses (takes longer)
./scripts/reproduce_canonical.sh
```

## Project Structure

```
grokking/
├── configs/
│   ├── base.yaml           # Base configuration
│   ├── canonical.yaml      # Canonical hyperparameters
│   ├── debug.yaml          # Fast debug config
│   └── sweeps/             # Sweep configurations
├── scripts/
│   ├── train.py            # Training script
│   ├── eval_checkpoints.py # Evaluation script
│   ├── analyze_fourier.py  # Fourier analysis
│   ├── analyze_progress.py # Progress measures
│   ├── run_ablation.py     # Fourier ablation
│   ├── train_sweep.py      # Hyperparameter sweeps
│   └── reproduce_canonical.sh
├── src/grokking/
│   ├── data/               # Dataset and tokenization
│   ├── models/             # Transformer implementation
│   ├── train/              # Training loop and utilities
│   ├── eval/               # Evaluation metrics
│   ├── interp/             # Interpretability tools
│   │   ├── fourier.py      # Fourier basis
│   │   ├── hooks.py        # Activation caching
│   │   ├── ablations.py    # Fourier ablations
│   │   └── progress_measures.py
│   ├── viz/                # Plotting functions
│   └── utils/              # Utilities
├── tests/                  # Unit tests
└── results/                # Output directory (gitignored)
```

## Scripts Usage

### Training

```bash
# Basic training
python scripts/train.py --config configs/canonical.yaml

# Override parameters via CLI
python scripts/train.py --config configs/canonical.yaml \
    --set optim.weight_decay=0.003 \
    --set run.seed=42
```

### Analysis Scripts

After training, run analysis scripts on the run directory:

```bash
# Evaluate checkpoints and generate grokking curve
python scripts/eval_checkpoints.py --run_dir results/runs/<run_dir>

# Fourier spectral analysis
python scripts/analyze_fourier.py --run_dir results/runs/<run_dir>

# Progress measures
python scripts/analyze_progress.py --run_dir results/runs/<run_dir>

# Fourier ablation
python scripts/run_ablation.py --run_dir results/runs/<run_dir> --step 200000
```

### Hyperparameter Sweeps

```bash
# Run a weight decay sweep
python scripts/train_sweep.py --sweep configs/sweeps/weight_decay.yaml

# Dry run (show commands without running)
python scripts/train_sweep.py --sweep configs/sweeps/weight_decay.yaml --dry-run
```

## Output Files

After a successful run, the following files are generated:

```
results/runs/<timestamp>__<name>__seed<seed>/
├── config.yaml                    # Resolved configuration
├── metrics.jsonl                  # Training metrics
├── checkpoints/
│   ├── step_*.pt                  # Model checkpoints
│   └── latest.pt                  # Latest checkpoint
├── figures/
│   ├── grokking_curve.png         # Train/test accuracy curves
│   ├── fourier_spectra.png        # Spectral energy plots
│   ├── progress_measures.png      # Progress measures over time
│   └── ablation_step_*.png        # Ablation results
└── analysis/
    ├── eval_summary.json          # Evaluation summary
    ├── fourier_metrics.json       # Fourier spectral data
    ├── progress_measures.json     # Progress measures data
    └── ablation_step_*.json       # Ablation results
```

## Configuration

### Canonical Hyperparameters

| Parameter | Value |
|-----------|-------|
| Prime modulus (p) | 113 |
| Train fraction | 0.3 |
| Model dimension | 128 |
| Layers | 2 |
| Attention heads | 4 |
| MLP dimension | 512 |
| Learning rate | 1e-3 |
| Weight decay | 1e-3 |
| Batch size | 4096 |
| Training steps | 200,000 |

### Debug Configuration

For fast iteration and testing, use `configs/debug.yaml`:
- Small prime (p=17, only 289 examples)
- Single layer, smaller dimensions
- 1000 steps (completes in ~1 minute on CPU)

## Success Criteria

A successful reproduction produces:

1. **Grokking curve**: Train accuracy reaches 99%+ early, test accuracy stays low, then suddenly jumps to 95%+

2. **Fourier structure**: Embedding spectral energy concentrates into a small subset of frequencies

3. **Ablation impact**: Ablating top frequencies causes ≥30 percentage point drop in test accuracy

4. **Progress measures**: Track embedding/unembedding frequency concentration over training

## If Grokking Doesn't Occur

If the canonical configuration doesn't produce grokking:

1. **Try different weight decay values**:
   ```bash
   python scripts/train_sweep.py --sweep configs/sweeps/weight_decay.yaml
   ```

2. **Try different train fractions**:
   ```bash
   python scripts/train_sweep.py --sweep configs/sweeps/train_fraction.yaml
   ```

3. **Run for more steps**: Increase `train.steps` in the config

4. **Check the loss curves**: Sometimes grokking occurs very late in training

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fourier.py -v

# Run with coverage
pytest tests/ --cov=grokking
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- numpy
- pyyaml
- matplotlib
- tqdm

## License

MIT License
