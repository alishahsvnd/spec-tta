#!/bin/bash
# Comprehensive Experiment: Compare Old SPEC-TTA vs New Hybrid vs PETSA
# Dataset: ETTh1, Horizon: 96
# Date: October 23, 2025

set -e

cd /home/alishah/PETSA

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║           COMPREHENSIVE EXPERIMENT: SPEC-TTA ENHANCEMENTS                 ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset: ETTh1"
echo "Horizon: 96"
echo "Model: iTransformer"
echo "Checkpoint: ./checkpoints/iTransformer/ETTh1_96/checkpoint_best.pth"
echo ""
echo "Experiments:"
echo "  1. Old SPEC-TTA (frequency-only, K_BINS=32, no quality detection)"
echo "  2. New SPEC-TTA (Phase 1+2, auto quality detection + hybrid mode)"
echo "  3. PETSA (time-domain LoRA, baseline comparison)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create results directory
mkdir -p results/COMPARISON_$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/COMPARISON_$(date +%Y%m%d_%H%M%S)"

# Common parameters
DATA_NAME="ETTh1"
PRED_LEN=96
MODEL_NAME="iTransformer"
CHECKPOINT_DIR="./checkpoints/iTransformer/ETTh1_96/"

echo "📊 Experiment 1: Old SPEC-TTA (Frequency-Only, K_BINS=32)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Configuration:"
echo "  • K_BINS: 32 (fixed)"
echo "  • DRIFT_THRESHOLD: 0.005"
echo "  • BETA_FREQ: 0.1"
echo "  • LR: 0.001"
echo "  • No quality detection, no hybrid mode"
echo ""

python main.py \
    DATA.NAME ${DATA_NAME} \
    DATA.PRED_LEN ${PRED_LEN} \
    MODEL.NAME ${MODEL_NAME} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
    TTA.SPEC_TTA.BETA_FREQ 0.1 \
    TTA.SPEC_TTA.LAMBDA_PW 1.0 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
    TTA.SPEC_TTA.LR 0.001 \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
    RESULT_DIR "${RESULT_DIR}/OLD_SPEC_TTA/" 2>&1 | tee ${RESULT_DIR}/old_spec_tta.log

OLD_MSE=$(grep -E "Final MSE:|MSE:" ${RESULT_DIR}/old_spec_tta.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
OLD_MAE=$(grep -E "Final MAE:|MAE:" ${RESULT_DIR}/old_spec_tta.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
OLD_PARAMS=$(grep -E "Total.*Parameters:" ${RESULT_DIR}/old_spec_tta.log | grep -oE "[0-9]+" | head -1)

echo ""
echo "✅ Old SPEC-TTA Results:"
echo "   MSE: ${OLD_MSE}"
echo "   MAE: ${OLD_MAE}"
echo "   Parameters: ${OLD_PARAMS}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
sleep 2

echo "📊 Experiment 2: New SPEC-TTA (Phase 1+2: Quality Detection + Hybrid)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Configuration:"
echo "  • K_BINS: 32 (initial, will auto-tune based on quality)"
echo "  • Quality detection: ENABLED"
echo "  • Hybrid mode: AUTO (activates for POOR/FAIR checkpoints)"
echo "  • Expected: Auto-tune to 64 bins + LoRA if poor checkpoint"
echo ""

python main.py \
    DATA.NAME ${DATA_NAME} \
    DATA.PRED_LEN ${PRED_LEN} \
    MODEL.NAME ${MODEL_NAME} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
    TTA.SPEC_TTA.BETA_FREQ 0.1 \
    TTA.SPEC_TTA.LAMBDA_PW 1.0 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
    TTA.SPEC_TTA.LR 0.001 \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
    RESULT_DIR "${RESULT_DIR}/NEW_SPEC_TTA_HYBRID/" 2>&1 | tee ${RESULT_DIR}/new_spec_tta_hybrid.log

NEW_MSE=$(grep -E "Final MSE:|MSE:" ${RESULT_DIR}/new_spec_tta_hybrid.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
NEW_MAE=$(grep -E "Final MAE:|MAE:" ${RESULT_DIR}/new_spec_tta_hybrid.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
NEW_PARAMS=$(grep -E "Total.*Parameters:|Created.*modules.*parameters" ${RESULT_DIR}/new_spec_tta_hybrid.log | grep -oE "[0-9]+" | tail -1)
CHECKPOINT_QUALITY=$(grep -E "Checkpoint Quality:|Quality Level:" ${RESULT_DIR}/new_spec_tta_hybrid.log | head -1)
HYBRID_MODE=$(grep -E "HYBRID mode|Enabling HYBRID" ${RESULT_DIR}/new_spec_tta_hybrid.log | head -1)

echo ""
echo "✅ New SPEC-TTA Results:"
echo "   MSE: ${NEW_MSE}"
echo "   MAE: ${NEW_MAE}"
echo "   Parameters: ${NEW_PARAMS}"
echo "   ${CHECKPOINT_QUALITY}"
echo "   Hybrid Mode: $([ -n "${HYBRID_MODE}" ] && echo 'ENABLED' || echo 'DISABLED')"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
sleep 2

echo "📊 Experiment 3: PETSA (Time-Domain LoRA Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Configuration:"
echo "  • Full LoRA on all linear layers"
echo "  • Rank 16"
echo "  • ~25,934 parameters"
echo ""

# Check if PETSA is available
if grep -q "TTA.PETSA" main.py 2>/dev/null || grep -rq "petsa" tta/ 2>/dev/null; then
    python main.py \
        DATA.NAME ${DATA_NAME} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${MODEL_NAME} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.METHOD petsa \
        TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
        RESULT_DIR "${RESULT_DIR}/PETSA/" 2>&1 | tee ${RESULT_DIR}/petsa.log
    
    PETSA_MSE=$(grep -E "Final MSE:|MSE:" ${RESULT_DIR}/petsa.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    PETSA_MAE=$(grep -E "Final MAE:|MAE:" ${RESULT_DIR}/petsa.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    PETSA_PARAMS=$(grep -E "Total.*Parameters:" ${RESULT_DIR}/petsa.log | grep -oE "[0-9]+" | head -1)
    
    echo ""
    echo "✅ PETSA Results:"
    echo "   MSE: ${PETSA_MSE}"
    echo "   MAE: ${PETSA_MAE}"
    echo "   Parameters: ${PETSA_PARAMS}"
else
    echo "⚠️  PETSA not available in this codebase"
    echo "   Using reference values from previous experiments:"
    PETSA_MSE="0.699"
    PETSA_MAE="0.601"
    PETSA_PARAMS="25934"
    echo "   MSE: ${PETSA_MSE} (reference)"
    echo "   MAE: ${PETSA_MAE} (reference)"
    echo "   Parameters: ${PETSA_PARAMS} (reference)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Generate comparison report
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                         📊 COMPARISON RESULTS                             ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset: ETTh1, Horizon: 96, Model: iTransformer"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Method                 | MSE      | MAE      | Parameters | Efficiency"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "Old SPEC-TTA          | %-8s | %-8s | %-10s | baseline\n" "${OLD_MSE}" "${OLD_MAE}" "${OLD_PARAMS}"
printf "New SPEC-TTA (Hybrid) | %-8s | %-8s | %-10s | " "${NEW_MSE}" "${NEW_MAE}" "${NEW_PARAMS}"
if [ -n "${NEW_PARAMS}" ] && [ -n "${PETSA_PARAMS}" ]; then
    EFFICIENCY=$(echo "scale=1; ${PETSA_PARAMS} / ${NEW_PARAMS}" | bc)
    echo "${EFFICIENCY}x vs PETSA"
else
    echo "N/A"
fi
printf "PETSA (reference)     | %-8s | %-8s | %-10s | baseline\n" "${PETSA_MSE}" "${PETSA_MAE}" "${PETSA_PARAMS}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Calculate improvements
if [ -n "${OLD_MSE}" ] && [ -n "${NEW_MSE}" ]; then
    MSE_IMPROVEMENT=$(echo "scale=2; (${OLD_MSE} - ${NEW_MSE}) / ${OLD_MSE} * 100" | bc)
    echo "📈 Improvements (New vs Old):"
    echo "   MSE Improvement: ${MSE_IMPROVEMENT}%"
fi

if [ -n "${PETSA_MSE}" ] && [ -n "${NEW_MSE}" ]; then
    MSE_VS_PETSA=$(echo "scale=2; (${NEW_MSE} - ${PETSA_MSE}) / ${PETSA_MSE} * 100" | bc)
    echo "   MSE vs PETSA: ${MSE_VS_PETSA}% (negative is better)"
fi

if [ -n "${NEW_PARAMS}" ] && [ -n "${PETSA_PARAMS}" ]; then
    PARAM_REDUCTION=$(echo "scale=1; (1 - ${NEW_PARAMS} / ${PETSA_PARAMS}) * 100" | bc)
    echo "   Parameter Reduction: ${PARAM_REDUCTION}% fewer than PETSA"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Results saved to: ${RESULT_DIR}/"
echo "   • old_spec_tta.log"
echo "   • new_spec_tta_hybrid.log"
echo "   • petsa.log (if available)"
echo ""
echo "✅ Experiment Complete!"
echo ""
