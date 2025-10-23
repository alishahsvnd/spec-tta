#!/bin/bash
# Test SPEC-TTA with TimeShift+PolyTrend in TRUE PT MODE (10% prefix)
# This matches the paper's original design with partial targets

set -e

DATASET="ETTh2"
MODEL="iTransformer"
OUTPUT_DIR="results/SPEC_TTA_TRUE_PT_MODE_${DATASET}"
LOG_FILE="spec_tta_true_pt_mode.log"

echo "=== SPEC-TTA with True PT Mode (10% Prefix) + Temporal Heads ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET, Model: $MODEL" | tee -a "$LOG_FILE"
echo "Testing on H=336 and H=720 (long horizons that previously failed)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create output directory
mkdir -p "$OUTPUT_DIR/$MODEL"

for HORIZON in 336 720; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running: $MODEL on $DATASET H=$HORIZON" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
    
    if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        echo "ERROR: Checkpoint not found at $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "PT Mode: First $(echo "scale=0; $HORIZON * 0.1 / 1" | bc) timesteps observed (10% of horizon)" | tee -a "$LOG_FILE"
    
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
echo "=== Comparison Summary ===" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for HORIZON in 336 720; do
    echo "" | tee -a "$LOG_FILE"
    echo "--- H=$HORIZON ---" | tee -a "$LOG_FILE"
    
    echo "Baseline (no TTA):" | tee -a "$LOG_FILE"
    grep "test_mse" "results/SPEC_TTA_BENCHMARK_ETTh2/iTransformer/iTransformer/ETTh2_${HORIZON}/test.txt" 2>/dev/null | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"
    
    echo "Previous SPEC-TTA (full horizon mask - broken):" | tee -a "$LOG_FILE"
    find results/SPEC_TTA_BENCHMARK_ETTh2 -name "*${HORIZON}*.txt" -exec grep -l "iTransformer" {} \; -exec grep -h "Final MSE\|Final MAE\|Total Adaptation" {} \; 2>/dev/null | head -3 | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"
    
    echo "New SPEC-TTA (true PT mode 10% + temporal heads):" | tee -a "$LOG_FILE"
    grep -E "(Final MSE|Final MAE|Total Adaptation)" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt" 2>/dev/null | tee -a "$LOG_FILE" || echo "Not found" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
