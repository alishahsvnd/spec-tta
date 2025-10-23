#!/bin/bash
# Test Improvement E (Adaptive Loss Schedule) + G (Horizon-Adaptive Drift) on H=720
# Expected: Even better performance with very long horizon adaptive weights

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=720

OUTPUT_DIR="results/SPEC_TTA_IMPROVEMENT_E_ADAPTIVE_LOSS"
LOG_FILE="test_improvement_e_h720.log"

echo "=== Testing Improvement E on H=720 (Very Long Horizon) ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET, Model: $MODEL, Horizon: $HORIZON" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"

if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
    echo "ERROR: Checkpoint not found" | tee -a "$LOG_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/$MODEL"

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Configuration for H=720 with Adaptive Schedule:" | tee -a "$LOG_FILE"
echo "  beta_freq: 0.001 (highly reduced from 0.05)" | tee -a "$LOG_FILE"
echo "  lambda_pw: 0.3" | tee -a "$LOG_FILE"
echo "  lambda_prox: 0.005 (highly increased from 0.0001)" | tee -a "$LOG_FILE"
echo "  lr: 0.0002 (highly reduced from 0.001)" | tee -a "$LOG_FILE"
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
baseline_mse = 0.4302

# Current SPEC-TTA (catastrophic failure)
current_mse = 1.332  # Known catastrophic result

# With improvement E+G
try:
    with open("results/SPEC_TTA_IMPROVEMENT_E_ADAPTIVE_LOSS/iTransformer/ETTh2_720.txt") as f:
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

print("\nH=720 Results:")
print(f"  Baseline (no TTA):                 MSE = {baseline_mse:.4f}")
print(f"  Current SPEC-TTA (catastrophic):   MSE = {current_mse:.4f} (+209.5%)")

if improvement_e_mse:
    change_e = ((improvement_e_mse - baseline_mse) / baseline_mse * 100)
    print(f"  With Improvement E+G (adaptive):   MSE = {improvement_e_mse:.4f} ({change_e:+.1f}%), Updates = {improvement_e_updates}")
    
    if improvement_e_mse < baseline_mse:
        improvement_pct = ((baseline_mse - improvement_e_mse) / baseline_mse * 100)
        print(f"\n✓ SUCCESS: {improvement_pct:.1f}% better than baseline!")
    elif improvement_e_mse < current_mse:
        improvement_pct = ((current_mse - improvement_e_mse) / current_mse * 100)
        print(f"\n⚠ Partial improvement: {improvement_pct:.1f}% better than catastrophic SPEC-TTA")
        print(f"  But still {change_e:+.1f}% worse than no-TTA baseline")
    else:
        print(f"\n✗ No improvement: Still catastrophic")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
