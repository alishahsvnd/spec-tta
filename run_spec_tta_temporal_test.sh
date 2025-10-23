#!/bin/bash
# Test SPEC-TTA with TimeShift+PolyTrend temporal heads on ETTh2 H=336
# Compare against catastrophic baseline results

set -e

DATASET="ETTh2"
HORIZON=336
MODEL="iTransformer"
OUTPUT_DIR="results/SPEC_TTA_TEMPORAL_HEADS_${DATASET}"
LOG_FILE="spec_tta_temporal_heads_test.log"

echo "=== Testing SPEC-TTA with TimeShift+PolyTrend Temporal Heads ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET, Horizon: $HORIZON, Model: $MODEL" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create output directory
mkdir -p "$OUTPUT_DIR/$MODEL"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"

if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Running SPEC-TTA experiment..." | tee -a "$LOG_FILE"
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
echo "=== Experiment Complete ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Extract and display results
if [ -f "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt" ]; then
    echo "Results for $MODEL on $DATASET H=$HORIZON:" | tee -a "$LOG_FILE"
    tail -20 "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Compare with previous catastrophic results
    echo "=== Comparison with Previous Results ===" | tee -a "$LOG_FILE"
    BASELINE_FILE="results/SPEC_TTA_BENCHMARK_${DATASET}/${MODEL}/${DATASET}_${HORIZON}.txt"
    if [ -f "$BASELINE_FILE" ]; then
        echo "Previous SPEC-TTA (catastrophic):" | tee -a "$LOG_FILE"
        grep -E "(Final MSE|Final MAE)" "$BASELINE_FILE" | tee -a "$LOG_FILE"
    fi
    
    PETSA_FILE="results/PETSA_COMPARISON_${DATASET}/${MODEL}/${DATASET}_${HORIZON}.txt"
    if [ -f "$PETSA_FILE" ]; then
        echo "PETSA baseline:" | tee -a "$LOG_FILE"
        grep -E "(Test MSE|Test MAE)" "$PETSA_FILE" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "New SPEC-TTA with temporal heads:" | tee -a "$LOG_FILE"
    grep -E "(Final MSE|Final MAE)" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
echo "Results saved to: $OUTPUT_DIR"
