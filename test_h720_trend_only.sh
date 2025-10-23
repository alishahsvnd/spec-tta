#!/bin/bash
# Test H=720 with trend-only adaptation (no spectral updates)
# Hypothesis: Spectral adaptation is harmful for H=720, but trend fitting might help

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=720

OUTPUT_DIR="results/SPEC_TTA_H720_TREND_ONLY"
LOG_FILE="test_h720_trend_only.log"

echo "=== Testing H=720: Trend-Only vs Full Adaptation ===" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
mkdir -p "$OUTPUT_DIR/$MODEL"

# Baseline - no TTA
echo "========================================" | tee -a "$LOG_FILE"
echo "Baseline: No TTA" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

python -u main.py \
    DATA.NAME "$DATASET" \
    DATA.PRED_LEN "$HORIZON" \
    MODEL.NAME "$MODEL" \
    MODEL.pred_len "$HORIZON" \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    TRAIN.CHECKPOINT_DIR "$CHECKPOINT_DIR" \
    TTA.ENABLE False \
    RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
    2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_baseline.txt"

echo "" | tee -a "$LOG_FILE"

# Test with very high drift threshold (almost no updates)
echo "========================================" | tee -a "$LOG_FILE"
echo "Test: Almost No Updates (threshold=0.1)" | tee -a "$LOG_FILE"
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
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.1 \
    TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE True \
    RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
    2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_minimal_updates.txt"

echo "" | tee -a "$LOG_FILE"

# Summary
echo "=== Summary ===" | tee -a "$LOG_FILE"
python3 << 'PYEOF' | tee -a "$LOG_FILE"
import re

configs = [
    ("Baseline (no TTA)", "baseline"),
    ("Minimal Updates (threshold=0.1)", "minimal_updates"),
]

print("\nH=720 Adaptation Strategy Comparison:")
print("\n| Config | MSE | Updates | Note |")
print("|--------|-----|---------|------|")

results = []
for name, suffix in configs:
    try:
        with open(f"results/SPEC_TTA_H720_TREND_ONLY/iTransformer/ETTh2_720_{suffix}.txt") as f:
            content = f.read()
            mse_match = re.search(r'(test_mse|Final MSE).*?([\d.]+)', content)
            updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
            if mse_match:
                mse = float(mse_match.group(2))
                updates = int(updates_match.group(1)) if updates_match else 0
                print(f"| {name:30s} | {mse:.4f} | {updates:7d} | |")
                results.append((name, mse, updates))
    except Exception as e:
        print(f"| {name:30s} | ERROR | N/A | {str(e)[:20]} |")

if len(results) >= 2:
    baseline_mse = results[0][1]
    minimal_mse = results[1][1]
    if minimal_mse < baseline_mse:
        improvement = ((baseline_mse - minimal_mse) / baseline_mse * 100)
        print(f"\n✓ Minimal updates better by {improvement:.1f}%!")
    elif minimal_mse > baseline_mse * 1.1:
        degradation = ((minimal_mse - baseline_mse) / baseline_mse * 100)
        print(f"\n⚠ Minimal updates {degradation:.1f}% worse")
    else:
        print(f"\n≈ Similar performance")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
