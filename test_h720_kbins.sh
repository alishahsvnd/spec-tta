#!/bin/bash
# Test H=720 with different K_BINS values
# Hypothesis: Longer horizons need more frequency bins to capture complexity
# Current K=32 might be insufficient

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=720

OUTPUT_DIR="results/SPEC_TTA_H720_KBINS_EXPERIMENTS"
LOG_FILE="test_h720_kbins.log"

echo "=== Testing H=720 with Different K_BINS ===" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
mkdir -p "$OUTPUT_DIR/$MODEL"

# Test multiple K values
for K in 16 32 48 64; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Testing K_BINS = $K" | tee -a "$LOG_FILE"
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
        TTA.SPEC_TTA.K_BINS "$K" \
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
        2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_kbins_${K}.txt"
    
    # Extract results
    MSE=$(grep "Final MSE:" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_kbins_${K}.txt" | awk '{print $3}')
    UPDATES=$(grep "Total Adaptation Updates:" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_kbins_${K}.txt" | awk '{print $4}')
    PARAMS=$(grep "Total Trainable Parameters:" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}_kbins_${K}.txt" | awk '{print $4}')
    
    echo "K=$K: MSE=$MSE, Updates=$UPDATES, Params=$PARAMS" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "=== Summary ===" | tee -a "$LOG_FILE"
python3 << 'PYEOF' | tee -a "$LOG_FILE"
import re
import glob

baseline_mse = 0.4302
results = []

for file in sorted(glob.glob("results/SPEC_TTA_H720_KBINS_EXPERIMENTS/iTransformer/ETTh2_720_kbins_*.txt")):
    k = re.search(r'kbins_(\d+)\.txt', file).group(1)
    with open(file) as f:
        content = f.read()
        mse_match = re.search(r'Final MSE: ([\d.]+)', content)
        updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
        params_match = re.search(r'Total Trainable Parameters:\s+(\d+)', content)
        if mse_match:
            mse = float(mse_match.group(1))
            updates = int(updates_match.group(1)) if updates_match else 0
            params = int(params_match.group(1)) if params_match else 0
            change = ((mse - baseline_mse) / baseline_mse * 100)
            results.append((k, mse, updates, params, change))

print("\nK_BINS Comparison (H=720):")
print(f"Baseline: MSE={baseline_mse:.4f}")
print("\n| K_BINS | MSE | Updates | Params | vs Baseline |")
print("|--------|-----|---------|--------|-------------|")
for k, mse, updates, params, change in results:
    print(f"| {k:6s} | {mse:.4f} | {updates:7d} | {params:6d} | {change:+6.1f}% |")

best_k, best_mse, best_updates, best_params, best_change = min(results, key=lambda x: x[1])
print(f"\n✓ Best: K_BINS={best_k}, MSE={best_mse:.4f} ({best_change:+.1f}%), Updates={best_updates}, Params={best_params}")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
