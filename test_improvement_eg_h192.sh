#!/bin/bash
# Test E+G improvements on H=192 
# Expected: Should work very well like H=96 and H=336

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=192

OUTPUT_DIR="results/SPEC_TTA_IMPROVEMENT_E_ADAPTIVE_LOSS"
LOG_FILE="test_improvement_eg_h192.log"

echo "=== Testing Improvements E+G on H=192 ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET, Model: $MODEL, Horizon: $HORIZON" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"

if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
    echo "ERROR: Checkpoint not found" | tee -a "$LOG_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/$MODEL"

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Configuration for H=192 with E+G:" | tee -a "$LOG_FILE"
echo "  E: Adaptive schedule - beta_freq=0.025, lambda_prox=0.0005, lr=0.00075" | tee -a "$LOG_FILE"
echo "  G: Horizon-adaptive drift (standard for H<192)" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

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

# Baseline (no TTA)
baseline_mse = 0.3002

# Current SPEC-TTA (catastrophic failure from old results)
# Need to check if H=192 also had issues like H=336

# With improvement E+G
try:
    with open("results/SPEC_TTA_IMPROVEMENT_E_ADAPTIVE_LOSS/iTransformer/ETTh2_192.txt") as f:
        content = f.read()
        match = re.search(r'Final MSE: ([\d.]+)', content)
        updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
        if match:
            eg_mse = float(match.group(1))
            eg_updates = int(updates_match.group(1)) if updates_match else 0
        else:
            eg_mse = None
            eg_updates = 0
except:
    eg_mse = None
    eg_updates = 0

print("\nH=192 Results:")
print(f"  Baseline (no TTA):                 MSE = {baseline_mse:.4f}")

if eg_mse:
    change = ((baseline_mse - eg_mse) / baseline_mse * 100)
    print(f"  With Improvement E+G (adaptive):   MSE = {eg_mse:.4f} ({change:+.1f}%), Updates = {eg_updates}")
    
    if eg_mse < baseline_mse:
        print(f"\n✓ SUCCESS: {change:.1f}% better than baseline!")
    else:
        print(f"\n⚠ Regression: {-change:.1f}% worse than baseline")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
