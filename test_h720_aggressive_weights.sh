#!/bin/bash
# Test H=720 with LESS conservative weights
# Hypothesis: Current weights (beta_freq=0.001, lr=0.0002) are TOO conservative
# Try more aggressive settings to allow better adaptation

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=720

OUTPUT_DIR="results/SPEC_TTA_H720_AGGRESSIVE_WEIGHTS"
LOG_FILE="test_h720_aggressive_weights.log"

echo "=== Testing H=720 with More Aggressive Weights ===" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
mkdir -p "$OUTPUT_DIR/$MODEL"

# Baseline E+G (current conservative)
echo "========================================" | tee -a "$LOG_FILE"
echo "Test 1: Current E+G (Conservative)" | tee -a "$LOG_FILE"
echo "beta_freq=0.001, lambda_prox=0.005, lr=0.0002" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

python -u main.py \
    DATA.NAME "$DATASET" \
    DATA.PRED_LEN "$HORIZON" \
    MODEL.NAME "$MODEL" \
    MODEL.pred_len "$HORIZON" \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TRAIN.CHECKPOINT_DIR "$CHECKPOINT_DIR" \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE True \
    RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
    2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_conservative.txt"

echo "" | tee -a "$LOG_FILE"

# Test 2: Moderately aggressive (H=336 weights)
echo "========================================" | tee -a "$LOG_FILE"
echo "Test 2: Moderate (Use H=336 weights)" | tee -a "$LOG_FILE"
echo "beta_freq=0.005, lambda_prox=0.001, lr=0.0005" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

python -u main.py \
    DATA.NAME "$DATASET" \
    DATA.PRED_LEN "$HORIZON" \
    MODEL.NAME "$MODEL" \
    MODEL.pred_len "$HORIZON" \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TRAIN.CHECKPOINT_DIR "$CHECKPOINT_DIR" \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE False \
    TTA.SPEC_TTA.BETA_FREQ 0.005 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.001 \
    TTA.SPEC_TTA.LR 0.0005 \
    RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
    2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_moderate.txt"

echo "" | tee -a "$LOG_FILE"

# Test 3: Aggressive (H=192 weights)
echo "========================================" | tee -a "$LOG_FILE"
echo "Test 3: Aggressive (Use H=192 weights)" | tee -a "$LOG_FILE"
echo "beta_freq=0.025, lambda_prox=0.0005, lr=0.00075" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

python -u main.py \
    DATA.NAME "$DATASET" \
    DATA.PRED_LEN "$HORIZON" \
    MODEL.NAME "$MODEL" \
    MODEL.pred_len "$HORIZON" \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TRAIN.CHECKPOINT_DIR "$CHECKPOINT_DIR" \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE False \
    TTA.SPEC_TTA.BETA_FREQ 0.025 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0005 \
    TTA.SPEC_TTA.LR 0.00075 \
    RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
    2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_aggressive.txt"

echo "" | tee -a "$LOG_FILE"

# Test 4: Very aggressive (closer to H=96 weights)
echo "========================================" | tee -a "$LOG_FILE"
echo "Test 4: Very Aggressive (near H=96 weights)" | tee -a "$LOG_FILE"
echo "beta_freq=0.01, lambda_prox=0.0001, lr=0.0005" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

python -u main.py \
    DATA.NAME "$DATASET" \
    DATA.PRED_LEN "$HORIZON" \
    MODEL.NAME "$MODEL" \
    MODEL.pred_len "$HORIZON" \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TRAIN.CHECKPOINT_DIR "$CHECKPOINT_DIR" \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE False \
    TTA.SPEC_TTA.BETA_FREQ 0.01 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
    TTA.SPEC_TTA.LR 0.0005 \
    RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
    2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_very_aggressive.txt"

echo "" | tee -a "$LOG_FILE"

# Summary
echo "=== Summary ===" | tee -a "$LOG_FILE"
python3 << 'PYEOF' | tee -a "$LOG_FILE"
import re
import glob

baseline_mse = 0.4302

configs = [
    ("Conservative (current)", "conservative"),
    ("Moderate (H=336)", "moderate"),
    ("Aggressive (H=192)", "aggressive"),
    ("Very Aggressive", "very_aggressive")
]

print("\nWeight Configuration Comparison (H=720):")
print(f"Baseline (no TTA): MSE={baseline_mse:.4f}\n")
print("| Config | MSE | Updates | vs Baseline |")
print("|--------|-----|---------|-------------|")

best = (None, float('inf'), 0, 0)

for name, suffix in configs:
    try:
        with open(f"results/SPEC_TTA_H720_AGGRESSIVE_WEIGHTS/iTransformer/ETTh2_720_{suffix}.txt") as f:
            content = f.read()
            mse_match = re.search(r'Final MSE: ([\d.]+)', content)
            updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
            if mse_match:
                mse = float(mse_match.group(1))
                updates = int(updates_match.group(1)) if updates_match else 0
                change = ((mse - baseline_mse) / baseline_mse * 100)
                print(f"| {name:20s} | {mse:.4f} | {updates:7d} | {change:+6.1f}% |")
                if mse < best[1]:
                    best = (name, mse, updates, change)
    except:
        print(f"| {name:20s} | N/A | N/A | N/A |")

if best[0]:
    print(f"\n✓ Best: {best[0]}, MSE={best[1]:.4f} ({best[3]:+.1f}%), Updates={best[2]}")
    if best[1] < baseline_mse:
        print(f"  ✓✓ SUCCESS: Better than baseline!")
    elif best[1] < 0.5:
        print(f"  ✓ Good: Close to baseline")
    else:
        print(f"  ⚠ Still needs improvement")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
