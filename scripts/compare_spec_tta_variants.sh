#!/bin/bash
# scripts/compare_spec_tta_variants.sh
# Compare original SPEC-TTA with improved versions

GPU=${1:-0}

echo "================================================"
echo "SPEC-TTA Comparison Experiments"
echo "================================================"
echo ""

# 1. Baseline: Original settings (no updates expected)
echo "[1/5] Running baseline with original settings..."
bash scripts/iTransformer/ETTh1_96/run_spec_tta.sh $GPU 16 0.05 0.01

# 2. Lower drift threshold only
echo ""
echo "[2/5] Running with lower drift threshold..."
bash scripts/iTransformer/ETTh1_96/run_spec_tta.sh $GPU 16 0.05 0.001

# 3. Lower threshold + higher learning rate
echo ""
echo "[3/5] Running with lower threshold + higher LR..."
CUDA_VISIBLE_DEVICES=$GPU python main.py \
  --config config.py \
  MODEL.NAME "iTransformer" \
  DATASET.DATA "ETTh1" \
  DATASET.LOOKBACK 96 \
  DATASET.HORIZON 96 \
  TTA.METHOD "SPEC_TTA" \
  TTA.SPEC_TTA.K_BINS 16 \
  TTA.SPEC_TTA.BETA_FREQ 0.05 \
  TTA.SPEC_TTA.DRIFT_THRESHOLD 0.001 \
  TTA.SPEC_TTA.LR 0.005 \
  CHECKPOINT.LOAD_DIR "./checkpoints/iTransformer/ETTh1_96" \
  RESULT_DIR "SPEC_TTA_LOWTHRESH_HIGHLR"

# 4. More frequency bins
echo ""
echo "[4/5] Running with more frequency bins (K=32)..."
bash scripts/iTransformer/ETTh1_96/run_spec_tta.sh $GPU 32 0.05 0.001

# 5. Optimized combination
echo ""
echo "[5/5] Running with optimized hyperparameters..."
CUDA_VISIBLE_DEVICES=$GPU python main.py \
  --config config.py \
  MODEL.NAME "iTransformer" \
  DATASET.DATA "ETTh1" \
  DATASET.LOOKBACK 96 \
  DATASET.HORIZON 96 \
  TTA.METHOD "SPEC_TTA" \
  TTA.SPEC_TTA.K_BINS 32 \
  TTA.SPEC_TTA.BETA_FREQ 0.1 \
  TTA.SPEC_TTA.DRIFT_THRESHOLD 0.0005 \
  TTA.SPEC_TTA.LR 0.005 \
  TTA.SPEC_TTA.LAMBDA_PW 0.5 \
  TTA.SPEC_TTA.LAMBDA_PROX 1e-5 \
  CHECKPOINT.LOAD_DIR "./checkpoints/iTransformer/ETTh1_96" \
  RESULT_DIR "SPEC_TTA_OPTIMIZED"

echo ""
echo "================================================"
echo "All experiments completed!"
echo "================================================"
echo ""
echo "Results locations:"
echo "  1. results/SPEC_TTA_KBINS_16_BETA_0.05_DRIFT_0.01/"
echo "  2. results/SPEC_TTA_KBINS_16_BETA_0.05_DRIFT_0.001/"
echo "  3. results/SPEC_TTA_LOWTHRESH_HIGHLR/"
echo "  4. results/SPEC_TTA_KBINS_32_BETA_0.05_DRIFT_0.001/"
echo "  5. results/SPEC_TTA_OPTIMIZED/"
echo ""

# Generate comparison table
echo "Generating comparison table..."
python3 <<'EOFPY'
import os
import numpy as np

results_base = "results"
variants = [
    ("Baseline (original)", "SPEC_TTA_KBINS_16_BETA_0.05_DRIFT_0.01"),
    ("Lower threshold", "SPEC_TTA_KBINS_16_BETA_0.05_DRIFT_0.001"),
    ("Lower thresh + high LR", "SPEC_TTA_LOWTHRESH_HIGHLR"),
    ("More freq bins (K=32)", "SPEC_TTA_KBINS_32_BETA_0.05_DRIFT_0.001"),
    ("Optimized", "SPEC_TTA_OPTIMIZED")
]

print("\n" + "="*80)
print("SPEC-TTA Performance Comparison")
print("="*80)
print(f"{'Variant':<30} {'Test MSE':>12} {'Test MAE':>12} {'Improvement':>12}")
print("-"*80)

baseline_mse = None
for name, dirname in variants:
    test_file = os.path.join(results_base, dirname, "iTransformer/ETTh1_96/test.txt")
    
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            content = f.read()
            # Parse: test_mse: 0.5409, test_mae: 0.5948, ...
            parts = content.split(',')
            mse = float(parts[0].split(':')[1].strip())
            mae = float(parts[1].split(':')[1].strip())
            
            if baseline_mse is None:
                baseline_mse = mse
                improvement = "baseline"
            else:
                improvement = f"{(baseline_mse - mse) / baseline_mse * 100:+.2f}%"
            
            print(f"{name:<30} {mse:>12.4f} {mae:>12.4f} {improvement:>12}")
    else:
        print(f"{name:<30} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

print("="*80)
EOFPY

echo ""
echo "Done! Check the table above for performance comparison."
