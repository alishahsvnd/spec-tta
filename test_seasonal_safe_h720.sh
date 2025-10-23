#!/bin/bash
# Test Seasonal Envelope Adapter + Safe Mode on H=720
# This uses constrained seasonal adapter with holdout-validated mixing

set -e

echo "=========================================="
echo "Testing Seasonal Envelope + Safe Mode on H=720"
echo "=========================================="
echo ""

# Configuration
DATASET="ETTh2"
HORIZON=720
MODEL="iTransformer"
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
RESULT_TAG="SPEC_TTA_SEASONAL_SAFE"
RESULT_DIR="./results/${RESULT_TAG}/${MODEL}/"
LOG_FILE="${RESULT_TAG}_${MODEL}_${DATASET}_${HORIZON}.log"

echo "Dataset: ${DATASET}"
echo "Horizon: ${HORIZON}"
echo "Model: ${MODEL}"
echo "Result Dir: ${RESULT_DIR}"
echo ""
echo "Configuration:"
echo "  - SeasonalEnvelopeAdapter (auto-enabled for H>=720)"
echo "  - Safe Mode mixing (auto-enabled for H>=720)"
echo "  - beta_freq=0 (PETSA paper recommendation for long horizons)"
echo "  - Output-only adaptation"
echo "  - Tail damping enabled"
echo ""

# Run with seasonal envelope + safe mode
# For H>=720, manager.py automatically switches to SeasonalEnvelopeAdapter
# and adapter_wrapper.py automatically enables safe mode
echo "[1/1] Testing Seasonal Envelope + Safe Mode"
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
    TTA.SPEC_TTA.USE_OUTPUT_ONLY True \
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
echo "Expected behavior:"
echo "  - SeasonalEnvelopeAdapter used for output side (H>=720)"
echo "  - beta_freq automatically set to 0.0 (turned off for H>=720)"
echo "  - Safe mode mixing enabled (H>=720)"
echo "  - safe_gamma metrics logged"
echo "  - Final MSE <= 0.430 (no-regret guarantee)"
echo "  - Possible controlled improvements if gamma>0 on some updates"
echo ""
echo "Extracting final results..."
tail -100 ${LOG_FILE} | grep -E "(MSE|MAE|safe_gamma|Updates|seasonal|beta_freq)" || echo "Results not yet available"
echo ""
echo "To compare with baseline (MSE=0.430):"
echo "  - If MSE ≈ 0.430 → Safe mode worked (prevented harmful updates)"
echo "  - If MSE < 0.430 → Seasonal adapter provided improvements!"
echo "  - MSE should NEVER be > 0.430 (no-regret guarantee)"
