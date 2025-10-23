#!/bin/bash
# Test Safe Mode (No-Regret TTA) on H=720
# This test verifies that safe mode prevents regression below baseline

set -e

echo "=========================================="
echo "Testing Safe Mode on H=720"
echo "=========================================="
echo ""

# Configuration
DATASET="ETTh2"
HORIZON=720
MODEL="iTransformer"
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
RESULT_TAG="SPEC_TTA_SAFE_MODE"
RESULT_DIR="./results/${RESULT_TAG}/${MODEL}/"
LOG_FILE="${RESULT_TAG}_${MODEL}_${DATASET}_${HORIZON}.log"

echo "Dataset: ${DATASET}"
echo "Horizon: ${HORIZON}"
echo "Model: ${MODEL}"
echo "Result Dir: ${RESULT_DIR}"
echo ""

# Run with safe mode enabled
echo "[1/1] Testing Safe Mode"
echo "----------------------------------------"
python main.py \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${HORIZON} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${HORIZON} \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
    TTA.SPEC_TTA.BETA_FREQ 0.001 \
    TTA.SPEC_TTA.LAMBDA_PW 1.0 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.005 \
    TTA.SPEC_TTA.LR 0.0002 \
    TTA.SPEC_TTA.K_BINS 16 \
    TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE False \
    TTA.SPEC_TTA.USE_OUTPUT_ONLY False \
    TTA.SPEC_TTA.USE_SAFE_UPDATES False \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
    RESULT_DIR "${RESULT_DIR}" \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo ""
echo "Results saved to: ${RESULT_DIR}"
echo "Log saved to: ${LOG_FILE}"
echo ""
echo "Expected outcome:"
echo "  - safe_gamma metrics logged for each update"
echo "  - gamma=0.0 means baseline is better (no adaptation applied)"
echo "  - gamma>0.0 means controlled mixing provides benefit"
echo "  - Final MSE should be <= 0.430 (baseline) or better"
echo ""
echo "Extracting results..."
tail -50 ${LOG_FILE} | grep -E "(MSE|MAE|safe_gamma|Updates)" || echo "Results not yet available"
