#!/bin/bash
# Test ALL Improvements D+E+F+G on H=720
# D: Output-only mode (freeze input adapter)
# E: Adaptive loss schedule
# F: Safe update manager with rollback
# G: Horizon-adaptive drift detection (already in drift.py)

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=720

OUTPUT_DIR="results/SPEC_TTA_IMPROVEMENT_DEFG_ALL"
LOG_FILE="test_improvement_defg_h720.log"

echo "=== Testing ALL Improvements D+E+F+G on H=720 ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET, Model: $MODEL, Horizon: $HORIZON" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"

if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
    echo "ERROR: Checkpoint not found" | tee -a "$LOG_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/$MODEL"

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Configuration for H=720 with D+E+F+G:" | tee -a "$LOG_FILE"
echo "  D: Output-only mode (freeze input adapter)" | tee -a "$LOG_FILE"
echo "  E: Adaptive schedule - beta_freq=0.001, lambda_prox=0.005, lr=0.0002" | tee -a "$LOG_FILE"
echo "  F: Safe updates (max_param_norm=5.0, patience=5)" | tee -a "$LOG_FILE"
echo "  G: Horizon-adaptive drift (sliding window)" | tee -a "$LOG_FILE"
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
    TTA.SPEC_TTA.USE_OUTPUT_ONLY True \
    TTA.SPEC_TTA.USE_SAFE_UPDATES True \
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

# Previous results
current_mse = 1.332  # Old SPEC-TTA (catastrophic)
eg_mse = 0.698  # E+G only
deg_mse = 0.698  # D+E+G

# With D+E+F+G
try:
    with open("results/SPEC_TTA_IMPROVEMENT_DEFG_ALL/iTransformer/ETTh2_720.txt") as f:
        content = f.read()
        match = re.search(r'Final MSE: ([\d.]+)', content)
        updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
        if match:
            defg_mse = float(match.group(1))
            defg_updates = int(updates_match.group(1)) if updates_match else 0
        else:
            defg_mse = None
            defg_updates = 0
except:
    defg_mse = None
    defg_updates = 0

print("\nH=720 Results Summary:")
print(f"  Baseline (no TTA):                 MSE = {baseline_mse:.4f}")
print(f"  Old SPEC-TTA (catastrophic):       MSE = {current_mse:.4f} (+209.5%)")
print(f"  E+G only:                          MSE = {eg_mse:.4f} (+62.2%)")
print(f"  D+E+G:                             MSE = {deg_mse:.4f} (+62.2%)")

if defg_mse:
    change_vs_baseline = ((defg_mse - baseline_mse) / baseline_mse * 100)
    print(f"  D+E+F+G (ALL improvements):        MSE = {defg_mse:.4f} ({change_vs_baseline:+.1f}%), Updates = {defg_updates}")
    
    if defg_mse < baseline_mse:
        improvement_pct = ((baseline_mse - defg_mse) / baseline_mse * 100)
        print(f"\n✓✓✓ SUCCESS: {improvement_pct:.1f}% better than baseline!")
    elif defg_mse < eg_mse * 0.95:
        improvement_pct = ((eg_mse - defg_mse) / eg_mse * 100)
        print(f"\n✓ Improvement: {improvement_pct:.1f}% better than E+G")
    elif defg_mse < baseline_mse * 1.2:
        print(f"\n✓ Acceptable: Within 20% of baseline")
    else:
        print(f"\n⚠ Still needs work: {change_vs_baseline:+.1f}% worse than baseline")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
