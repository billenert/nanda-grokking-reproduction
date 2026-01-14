#!/usr/bin/env bash
#
# Reproduce canonical grokking experiment end-to-end.
#
# This script runs the full reproduction pipeline:
# 1. Training with canonical hyperparameters
# 2. Evaluation and grokking curve generation
# 3. Fourier spectral analysis
# 4. Progress measures analysis
# 5. Fourier ablation experiment
#
# Usage:
#   ./scripts/reproduce_canonical.sh
#
# The script will exit with non-zero status if any step fails.

set -e  # Exit on any error

echo "=================================================="
echo "Grokking Modular Addition - Canonical Reproduction"
echo "=================================================="
echo ""

# Step 1: Train the model
echo "Step 1/5: Training model with canonical config..."
echo "----------------------------------------"

# Run training and capture run directory
TRAIN_OUTPUT=$(python scripts/train.py --config configs/canonical.yaml 2>&1)
echo "$TRAIN_OUTPUT"

# Extract run directory from output
RUN_DIR=$(echo "$TRAIN_OUTPUT" | grep "^RUN_DIR=" | cut -d'=' -f2)

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: Could not extract run directory from training output"
    exit 1
fi

echo ""
echo "Run directory: $RUN_DIR"
echo ""

# Step 2: Evaluate checkpoints and generate grokking curve
echo "Step 2/5: Evaluating checkpoints..."
echo "----------------------------------------"
python scripts/eval_checkpoints.py --run_dir "$RUN_DIR"
echo ""

# Step 3: Fourier spectral analysis
echo "Step 3/5: Analyzing Fourier spectra..."
echo "----------------------------------------"
python scripts/analyze_fourier.py --run_dir "$RUN_DIR"
echo ""

# Step 4: Progress measures analysis
echo "Step 4/5: Computing progress measures..."
echo "----------------------------------------"
python scripts/analyze_progress.py --run_dir "$RUN_DIR"
echo ""

# Step 5: Run ablation on final checkpoint
echo "Step 5/5: Running Fourier ablation..."
echo "----------------------------------------"
# Find the latest checkpoint step
LATEST_STEP=$(ls "$RUN_DIR/checkpoints/" | grep 'step_' | sed 's/step_//' | sed 's/.pt//' | sort -n | tail -1)
python scripts/run_ablation.py --run_dir "$RUN_DIR" --step "$LATEST_STEP"
echo ""

# Verify outputs
echo "=================================================="
echo "Verifying outputs..."
echo "=================================================="

ERRORS=0

# Check required files
FILES_TO_CHECK=(
    "config.yaml"
    "metrics.jsonl"
    "checkpoints/latest.pt"
    "figures/grokking_curve.png"
    "analysis/eval_summary.json"
    "analysis/fourier_metrics.json"
    "figures/fourier_spectra.png"
    "analysis/progress_measures.json"
    "figures/progress_measures.png"
)

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$RUN_DIR/$file" ]; then
        echo "  [OK] $file"
    else
        echo "  [MISSING] $file"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check for ablation files (with dynamic step number)
if ls "$RUN_DIR/analysis/ablation_step_"*.json 1> /dev/null 2>&1; then
    echo "  [OK] analysis/ablation_step_*.json"
else
    echo "  [MISSING] analysis/ablation_step_*.json"
    ERRORS=$((ERRORS + 1))
fi

if ls "$RUN_DIR/figures/ablation_step_"*.png 1> /dev/null 2>&1; then
    echo "  [OK] figures/ablation_step_*.png"
else
    echo "  [MISSING] figures/ablation_step_*.png"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=================================================="
if [ $ERRORS -eq 0 ]; then
    echo "SUCCESS: All outputs generated correctly!"
    echo "Run directory: $RUN_DIR"
else
    echo "ERROR: $ERRORS required files are missing"
    exit 1
fi
echo "=================================================="
