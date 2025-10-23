#!/usr/bin/env bash
# Run SPEC-TTA on all 6 backbones for ETTh1 with pred_len=96
# Uses proper checkpoint loading for each model
# Configuration: k_bins=32, drift_threshold=0.005 (best settings from paper)

set -e  # Exit on error

DATASET="ETTh1"
PRED_LEN=96
K_BINS=32
DRIFT_THRESHOLD=0.005
BETA_FREQ=0.1
RESULT_BASE="./results/SPEC_TTA_FULL_COMPARISON"

# SPEC-TTA hyperparameters (from paper)
HUBER_DELTA=0.5
LAMBDA_PW=1.0
LAMBDA_PROX=0.0001
LAMBDA_HC=0.1
GRAD_CLIP=1.0
LR=0.001
PATCH_LEN=24

# Models to test
MODELS=("iTransformer" "DLinear" "FreTS" "MICN" "PatchTST")

echo "=========================================="
echo "SPEC-TTA Full Backbone Comparison"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Prediction Length: $PRED_LEN"
echo "K_BINS: $K_BINS"
echo "Drift Threshold: $DRIFT_THRESHOLD"
echo "Models: ${MODELS[@]}"
echo "=========================================="
echo ""

# Create results directory
mkdir -p "$RESULT_BASE"

# Run each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running SPEC-TTA on $MODEL"
    echo "=========================================="
    
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    RESULT_DIR="${RESULT_BASE}/${MODEL}/"
    OUTPUT_FILE="${RESULT_BASE}/SPEC_TTA_${MODEL}_${DATASET}_${PRED_LEN}.txt"
    
    # Check if checkpoint exists
    if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        echo "❌ ERROR: Checkpoint not found at ${CHECKPOINT_DIR}/checkpoint_best.pth"
        echo "Skipping $MODEL..."
        continue
    fi
    
    echo "✓ Checkpoint found: ${CHECKPOINT_DIR}/checkpoint_best.pth"
    echo "✓ Result directory: $RESULT_DIR"
    echo "✓ Output file: $OUTPUT_FILE"
    echo ""
    
    # Run SPEC-TTA
    python -u main.py \
        DATA.NAME $DATASET \
        DATA.PRED_LEN $PRED_LEN \
        MODEL.NAME $MODEL \
        MODEL.pred_len $PRED_LEN \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR $CHECKPOINT_DIR \
        TTA.ENABLE True \
        TTA.SPEC_TTA.K_BINS $K_BINS \
        TTA.SPEC_TTA.PATCH_LEN $PATCH_LEN \
        TTA.SPEC_TTA.HUBER_DELTA $HUBER_DELTA \
        TTA.SPEC_TTA.BETA_FREQ $BETA_FREQ \
        TTA.SPEC_TTA.LAMBDA_PW $LAMBDA_PW \
        TTA.SPEC_TTA.LAMBDA_PROX $LAMBDA_PROX \
        TTA.SPEC_TTA.LAMBDA_HC $LAMBDA_HC \
        TTA.SPEC_TTA.DRIFT_THRESHOLD $DRIFT_THRESHOLD \
        TTA.SPEC_TTA.LR $LR \
        TTA.SPEC_TTA.GRAD_CLIP $GRAD_CLIP \
        RESULT_DIR $RESULT_DIR \
        2>&1 | tee "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ $MODEL completed successfully!"
    else
        echo "❌ $MODEL failed!"
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULT_BASE"
echo ""
echo "Output files:"
for MODEL in "${MODELS[@]}"; do
    OUTPUT_FILE="${RESULT_BASE}/SPEC_TTA_${MODEL}_${DATASET}_${PRED_LEN}.txt"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  ✓ $OUTPUT_FILE"
    else
        echo "  ✗ $OUTPUT_FILE (missing)"
    fi
done
echo ""
echo "To view results, run:"
echo "  grep -A3 'Final MSE' ${RESULT_BASE}/*.txt"
echo ""
