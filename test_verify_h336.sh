#!/bin/bash
# Quick verification that H=336 results are not affected by safe mode changes

set -e

echo "=========================================="
echo "Verification Test: H=336 (Should be unchanged)"
echo "=========================================="
echo ""

DATASET="ETTh2"
HORIZON=336
MODEL="iTransformer"
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
RESULT_TAG="SPEC_TTA_VERIFY_H336"
RESULT_DIR="./results/${RESULT_TAG}/${MODEL}/"
LOG_FILE="${RESULT_TAG}_${MODEL}_${DATASET}_${HORIZON}.log"

echo "Testing H=336 with Improvement E+G"
echo "Expected MSE: ~0.263 (from previous runs with E+G improvements)"
echo ""

python main.py \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${HORIZON} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${HORIZON} \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
    TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE True \
    TTA.SPEC_TTA.K_BINS 16 \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
    RESULT_DIR "${RESULT_DIR}" \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "Results:"
tail -50 ${LOG_FILE} | grep -E "(Final MSE|Final MAE|use_safe_mode)" || echo "Check log for details"
echo ""
echo "Expected: MSE ≈ 0.263 (no change from before)"
