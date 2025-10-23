#!/bin/bash
# Quick verification that H=96 results are not affected by safe mode changes

set -e

echo "=========================================="
echo "Verification Test: H=96 (Should be unchanged)"
echo "=========================================="
echo ""

DATASET="ETTh2"
HORIZON=96
MODEL="iTransformer"
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
RESULT_TAG="SPEC_TTA_VERIFY_H96"
RESULT_DIR="./results/${RESULT_TAG}/${MODEL}/"
LOG_FILE="${RESULT_TAG}_${MODEL}_${DATASET}_${HORIZON}.log"

echo "Testing H=96 to verify no regression from safe mode changes"
echo "Expected MSE: ~0.264 (from previous runs)"
echo ""

python main.py \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${HORIZON} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${HORIZON} \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.01 \
    TTA.SPEC_TTA.BETA_FREQ 0.05 \
    TTA.SPEC_TTA.LAMBDA_PW 1.0 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
    TTA.SPEC_TTA.LR 0.001 \
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
tail -50 ${LOG_FILE} | grep -E "(Final MSE|Final MAE|use_safe_mode|mask_pt)" || echo "Check log for details"
echo ""
echo "Expected: MSE ≈ 0.264 (no change from before)"
echo "If MSE is significantly different, safe mode changes affected H<720 results!"
