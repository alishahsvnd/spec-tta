#!/bin/bash
# Test H=720 with HIGHER drift threshold (more selective updates)
# Hypothesis: Current 0.01 causes too many updates that make things worse
# Try 0.02 to only update when drift is really significant

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=720

OUTPUT_DIR="results/SPEC_TTA_H720_DRIFT_THRESHOLD_EXPERIMENTS"
LOG_FILE="test_h720_drift_threshold.log"

echo "=== Testing H=720 with Different Drift Thresholds ===" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
mkdir -p "$OUTPUT_DIR/$MODEL"

# Test multiple thresholds
for THRESHOLD in 0.005 0.01 0.02 0.05; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Testing drift_threshold = $THRESHOLD" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
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
        TTA.SPEC_TTA.DRIFT_THRESHOLD "$THRESHOLD" \
        TTA.SPEC_TTA.LR 0.001 \
        TTA.SPEC_TTA.GRAD_CLIP 1.0 \
        TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE True \
        RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
        2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_threshold_${THRESHOLD}.txt"
    
    # Extract results
    MSE=$(grep "Final MSE:" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_threshold_${THRESHOLD}.txt" | awk '{print $3}')
    UPDATES=$(grep "Total Adaptation Updates:" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_threshold_${THRESHOLD}.txt" | awk '{print $4}')
    
    echo "Threshold=$THRESHOLD: MSE=$MSE, Updates=$UPDATES" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "=== Summary ===" | tee -a "$LOG_FILE"
python3 << 'PYEOF' | tee -a "$LOG_FILE"
import re
import glob

baseline_mse = 0.4302
results = []

for file in sorted(glob.glob("results/SPEC_TTA_H720_DRIFT_THRESHOLD_EXPERIMENTS/iTransformer/ETTh2_720_threshold_*.txt")):
    threshold = re.search(r'threshold_([\d.]+)\.txt', file).group(1)
    with open(file) as f:
        content = f.read()
        mse_match = re.search(r'Final MSE: ([\d.]+)', content)
        updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
        if mse_match:
            mse = float(mse_match.group(1))
            updates = int(updates_match.group(1)) if updates_match else 0
            change = ((mse - baseline_mse) / baseline_mse * 100)
            results.append((threshold, mse, updates, change))

print("\nDrift Threshold Comparison (H=720):")
print(f"Baseline: MSE={baseline_mse:.4f}")
print("\n| Threshold | MSE | Updates | vs Baseline |")
print("|-----------|-----|---------|-------------|")
for threshold, mse, updates, change in results:
    print(f"| {threshold:8s} | {mse:.4f} | {updates:7d} | {change:+6.1f}% |")

best_threshold, best_mse, best_updates, best_change = min(results, key=lambda x: x[1])
print(f"\n✓ Best: threshold={best_threshold}, MSE={best_mse:.4f} ({best_change:+.1f}%), Updates={best_updates}")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
