#!/bin/bash
# Test SPEC-TTA with PT mode on SHORT horizons (H=96, H=192)
# Verify if the 10% PT mode change breaks previously good results

set -e

DATASET="ETTh2"
MODEL="iTransformer"
OUTPUT_DIR="results/SPEC_TTA_TRUE_PT_SHORT_HORIZONS_${DATASET}"
LOG_FILE="spec_tta_pt_short_horizons.log"

echo "=== Testing SPEC-TTA PT Mode on SHORT Horizons ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET, Model: $MODEL" | tee -a "$LOG_FILE"
echo "Testing on H=96 and H=192 (previously successful)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create output directory
mkdir -p "$OUTPUT_DIR/$MODEL"

for HORIZON in 96 192; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running: $MODEL on $DATASET H=$HORIZON" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
    
    if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        echo "ERROR: Checkpoint not found at $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
        continue
    fi
    
    PT_SIZE=$(echo "scale=0; $HORIZON * 0.1 / 1" | bc)
    echo "PT Mode: First $PT_SIZE timesteps observed (10% of horizon)" | tee -a "$LOG_FILE"
    
    python -u main.py \
        DATA.NAME "$DATASET" \
        DATA.PRED_LEN "$HORIZON" \
        MODEL.NAME "$MODEL" \
        MODEL.pred_len "$HORIZON" \
        TRAIN.ENABLE False \
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
        RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
        2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt"
    
    echo "" | tee -a "$LOG_FILE"
    echo "Results:" | tee -a "$LOG_FILE"
    grep -E "(Final MSE|Final MAE|Total Adaptation)" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "=== Detailed Comparison ===" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for HORIZON in 96 192; do
    echo "" | tee -a "$LOG_FILE"
    echo "--- H=$HORIZON (SHORT HORIZON) ---" | tee -a "$LOG_FILE"
    
    echo "Baseline (no TTA):" | tee -a "$LOG_FILE"
    grep "test_mse" "results/SPEC_TTA_BENCHMARK_ETTh2/iTransformer/iTransformer/ETTh2_${HORIZON}/test.txt" 2>/dev/null | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "Previous SPEC-TTA (full mask - should have been GOOD):" | tee -a "$LOG_FILE"
    find results/SPEC_TTA_BENCHMARK_ETTh2 -name "*${HORIZON}*.txt" -exec grep -l "iTransformer" {} \; -exec grep -h "Final MSE\|Final MAE\|Total Adaptation" {} \; 2>/dev/null | head -3 | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "New SPEC-TTA (10% PT mode + temporal heads):" | tee -a "$LOG_FILE"
    grep -E "(Final MSE|Final MAE|Total Adaptation)" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt" 2>/dev/null | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "=== Summary Table ===" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

python3 << 'PYEOF' | tee -a "$LOG_FILE"
import re

results = {}
for h in [96, 192]:
    results[h] = {}
    
    # Baseline
    try:
        with open(f"results/SPEC_TTA_BENCHMARK_ETTh2/iTransformer/iTransformer/ETTh2_{h}/test.txt") as f:
            content = f.read()
            match = re.search(r'test_mse: ([\d.]+), test_mae: ([\d.]+)', content)
            if match:
                results[h]['baseline_mse'] = float(match.group(1))
                results[h]['baseline_mae'] = float(match.group(2))
    except: pass
    
    # Previous SPEC-TTA
    try:
        import glob
        files = glob.glob(f"results/SPEC_TTA_BENCHMARK_ETTh2/*{h}*.txt")
        for f in files:
            with open(f) as file:
                content = file.read()
                if 'iTransformer' in content:
                    mse = re.search(r'Final MSE: ([\d.]+)', content)
                    mae = re.search(r'Final MAE: ([\d.]+)', content)
                    if mse and mae:
                        results[h]['prev_mse'] = float(mse.group(1))
                        results[h]['prev_mae'] = float(mae.group(1))
                        break
    except: pass
    
    # New PT mode
    try:
        with open(f"results/SPEC_TTA_TRUE_PT_SHORT_HORIZONS_ETTh2/iTransformer/ETTh2_{h}.txt") as f:
            content = f.read()
            mse = re.search(r'Final MSE: ([\d.]+)', content)
            mae = re.search(r'Final MAE: ([\d.]+)', content)
            if mse and mae:
                results[h]['pt_mse'] = float(mse.group(1))
                results[h]['pt_mae'] = float(mae.group(1))
    except: pass

print("\nHorizon | Baseline MSE | Previous SPEC-TTA | New PT Mode | Change")
print("--------|--------------|-------------------|-------------|--------")
for h in [96, 192]:
    r = results[h]
    base_mse = r.get('baseline_mse', 0)
    prev_mse = r.get('prev_mse', 0)
    pt_mse = r.get('pt_mse', 0)
    
    if base_mse and prev_mse and pt_mse:
        prev_gain = ((base_mse - prev_mse) / base_mse * 100)
        pt_gain = ((base_mse - pt_mse) / base_mse * 100)
        status = "✓" if pt_gain > 0 else "✗"
        print(f"H={h:3d}  | {base_mse:12.4f} | {prev_mse:10.4f} ({prev_gain:+5.1f}%) | {pt_mse:7.4f} ({pt_gain:+5.1f}%) | {status}")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
