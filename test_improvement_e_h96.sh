#!/bin/bash
# Test Improvement E (Adaptive Loss Schedule) individually
# Phase 1: Test on H=96 to ensure no regression

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=96

OUTPUT_DIR="results/SPEC_TTA_IMPROVEMENT_E_ADAPTIVE_LOSS"
LOG_FILE="test_improvement_e.log"

echo "=== Testing Improvement E: Adaptive Loss Schedule ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET, Model: $MODEL, Horizon: $HORIZON" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"

if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
    echo "ERROR: Checkpoint not found" | tee -a "$LOG_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/$MODEL"

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Configuration for H=96 with Adaptive Schedule:" | tee -a "$LOG_FILE"
echo "  beta_freq: 0.05 (standard for short horizon)" | tee -a "$LOG_FILE"
echo "  lambda_pw: 1.00" | tee -a "$LOG_FILE"
echo "  lambda_prox: 0.0001" | tee -a "$LOG_FILE"
echo "  lr: 0.001" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# NOTE: The adaptive schedule for H=96 should use same weights as baseline
# So this test verifies the adaptive schedule mechanism doesn't break anything

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
    TTA.SPEC_TTA.PATCH_LEN 24 \
    TTA.SPEC_TTA.HUBER_DELTA 0.5 \
    TTA.SPEC_TTA.BETA_FREQ 0.05 \
    TTA.SPEC_TTA.LAMBDA_PW 1.0 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
    TTA.SPEC_TTA.LAMBDA_HC 0.1 \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.01 \
    TTA.SPEC_TTA.LR 0.001 \
    TTA.SPEC_TTA.GRAD_CLIP 1.0 \
    TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE True \
    RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
    2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt"

echo "" | tee -a "$LOG_FILE"
echo "=== Results ===" | tee -a "$LOG_FILE"
grep -E "(Final MSE|Final MAE|Total Adaptation)" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Comparison ===" | tee -a "$LOG_FILE"

python3 << 'PYEOF' | tee -a "$LOG_FILE"
import re

# Baseline (sanity check)
baseline_mse = 0.2656

# Current SPEC-TTA with temporal heads (no improvements)
try:
    with open("results/SPEC_TTA_SANITY_CHECK/iTransformer/ETTh2_96.txt") as f:
        content = f.read()
        match = re.search(r'Final MSE: ([\d.]+)', content)
        if match:
            current_mse = float(match.group(1))
        else:
            current_mse = None
except:
    current_mse = None

# With improvement E
try:
    with open("results/SPEC_TTA_IMPROVEMENT_E_ADAPTIVE_LOSS/iTransformer/ETTh2_96.txt") as f:
        content = f.read()
        match = re.search(r'Final MSE: ([\d.]+)', content)
        updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
        if match:
            improvement_e_mse = float(match.group(1))
            improvement_e_updates = int(updates_match.group(1)) if updates_match else 0
        else:
            improvement_e_mse = None
            improvement_e_updates = 0
except:
    improvement_e_mse = None
    improvement_e_updates = 0

print("\nH=96 Results:")
print(f"  Baseline (no TTA):              MSE = {baseline_mse:.4f}")

if current_mse:
    change_current = ((baseline_mse - current_mse) / baseline_mse * 100)
    print(f"  Current SPEC-TTA (temporal):    MSE = {current_mse:.4f} ({change_current:+.1f}%)")

if improvement_e_mse:
    change_e = ((baseline_mse - improvement_e_mse) / baseline_mse * 100)
    print(f"  With Improvement E (adaptive):  MSE = {improvement_e_mse:.4f} ({change_e:+.1f}%), Updates = {improvement_e_updates}")
    
    if current_mse and improvement_e_mse:
        diff = current_mse - improvement_e_mse
        if abs(diff) < 0.01:
            print(f"\n✓ No regression: Difference {diff:+.4f} (negligible)")
        elif diff > 0:
            print(f"\n✓ Improvement: {diff:.4f} better than baseline")
        else:
            print(f"\n⚠ Slight regression: {-diff:.4f} worse (acceptable if < 5%)")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
