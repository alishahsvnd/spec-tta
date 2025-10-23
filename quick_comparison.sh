#!/bin/bash
# Quick Comparison: Old vs New SPEC-TTA on ETTh1

cd /home/alishah/PETSA || exit 1

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║      SPEC-TTA Comparison: Old vs New (Hybrid Mode)           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check data and checkpoint
if [ ! -f "data/ETT/ETTh1.csv" ]; then
    echo "❌ Error: data/ETT/ETTh1.csv not found!"
    exit 1
fi

if [ ! -f "checkpoints/iTransformer/ETTh1_96/checkpoint_best.pth" ]; then
    echo "❌ Error: checkpoint not found!"
    exit 1
fi

echo "✅ Data and checkpoint found"
echo ""

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/COMPARISON_${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Experiment 1: Old SPEC-TTA (Frequency-Only, Fixed K_BINS=32)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running..."
echo ""

python main.py \
    DATA.NAME ETTh1 \
    DATA.PRED_LEN 96 \
    MODEL.NAME iTransformer \
    MODEL.pred_len 96 \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
    TTA.SPEC_TTA.BETA_FREQ 0.1 \
    TTA.SPEC_TTA.LAMBDA_PW 1.0 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
    TTA.SPEC_TTA.LR 0.001 \
    TRAIN.CHECKPOINT_DIR "./checkpoints/iTransformer/ETTh1_96/" \
    RESULT_DIR "${RESULT_DIR}/OLD_SPEC_TTA/" > "${RESULT_DIR}/old.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Old SPEC-TTA completed"
    
    # Extract results
    OLD_MSE=$(grep "Final MSE:" "${RESULT_DIR}/old.log" | grep -oE "[0-9]+\.[0-9]+")
    OLD_MAE=$(grep "Final MAE:" "${RESULT_DIR}/old.log" | grep -oE "[0-9]+\.[0-9]+")
    OLD_PARAMS=$(grep "Total Trainable Parameters:" "${RESULT_DIR}/old.log" | grep -oE "[0-9]+")
    OLD_UPDATES=$(grep "Total Adaptation Updates:" "${RESULT_DIR}/old.log" | grep -oE "[0-9]+")
    
    echo "   MSE: ${OLD_MSE}"
    echo "   MAE: ${OLD_MAE}"
    echo "   Parameters: ${OLD_PARAMS}"
    echo "   Updates: ${OLD_UPDATES}"
else
    echo "❌ Old SPEC-TTA failed. Check ${RESULT_DIR}/old.log"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Experiment 2: New SPEC-TTA (Phase 1+2: Auto + Hybrid)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running..."
echo ""

python main.py \
    DATA.NAME ETTh1 \
    DATA.PRED_LEN 96 \
    MODEL.NAME iTransformer \
    MODEL.pred_len 96 \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
    TTA.SPEC_TTA.BETA_FREQ 0.1 \
    TTA.SPEC_TTA.LAMBDA_PW 1.0 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
    TTA.SPEC_TTA.LR 0.001 \
    TRAIN.CHECKPOINT_DIR "./checkpoints/iTransformer/ETTh1_96/" \
    RESULT_DIR "${RESULT_DIR}/NEW_SPEC_TTA/" > "${RESULT_DIR}/new.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ New SPEC-TTA completed"
    
    # Extract results
    NEW_MSE=$(grep "Final MSE:" "${RESULT_DIR}/new.log" | grep -oE "[0-9]+\.[0-9]+")
    NEW_MAE=$(grep "Final MAE:" "${RESULT_DIR}/new.log" | grep -oE "[0-9]+\.[0-9]+")
    NEW_PARAMS=$(grep "Created SPEC-TTA modules with" "${RESULT_DIR}/new.log" | grep -oE "[0-9]+" | head -1)
    NEW_UPDATES=$(grep "Total Adaptation Updates:" "${RESULT_DIR}/new.log" | grep -oE "[0-9]+")
    QUALITY=$(grep "Quality Level:" "${RESULT_DIR}/new.log" | grep -oE "POOR|FAIR|GOOD|EXCELLENT" | head -1)
    HYBRID=$(grep -c "Enabling HYBRID mode" "${RESULT_DIR}/new.log")
    
    echo "   MSE: ${NEW_MSE}"
    echo "   MAE: ${NEW_MAE}"
    echo "   Parameters: ${NEW_PARAMS}"
    echo "   Updates: ${NEW_UPDATES}"
    echo "   Quality: ${QUALITY}"
    echo "   Hybrid Mode: $([[ ${HYBRID} -gt 0 ]] && echo 'YES' || echo 'NO')"
else
    echo "❌ New SPEC-TTA failed. Check ${RESULT_DIR}/new.log"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 COMPARISON SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-25s | %-10s | %-10s | %-12s | %-8s\n" "Method" "MSE" "MAE" "Parameters" "Updates"
echo "───────────────────────────────────────────────────────────────"
printf "%-25s | %-10s | %-10s | %-12s | %-8s\n" "Old SPEC-TTA" "${OLD_MSE}" "${OLD_MAE}" "${OLD_PARAMS}" "${OLD_UPDATES}"
printf "%-25s | %-10s | %-10s | %-12s | %-8s\n" "New SPEC-TTA (Hybrid)" "${NEW_MSE}" "${NEW_MAE}" "${NEW_PARAMS}" "${NEW_UPDATES}"
printf "%-25s | %-10s | %-10s | %-12s | %-8s\n" "PETSA (reference)" "0.699" "0.601" "25934" "~77"
echo "───────────────────────────────────────────────────────────────"
echo ""

if [[ -n "${OLD_MSE}" && -n "${NEW_MSE}" ]]; then
    IMPROVEMENT=$(python3 -c "print(f'{((float('${OLD_MSE}') - float('${NEW_MSE}')) / float('${OLD_MSE}') * 100):.1f}')")
    echo "📈 MSE Improvement (New vs Old): ${IMPROVEMENT}%"
fi

if [[ -n "${NEW_MSE}" ]]; then
    VS_PETSA=$(python3 -c "print(f'{((float('${NEW_MSE}') - 0.699) / 0.699 * 100):.1f}')")
    echo "📊 MSE vs PETSA: ${VS_PETSA}%"
fi

if [[ -n "${NEW_PARAMS}" ]]; then
    EFFICIENCY=$(python3 -c "print(f'{(25934 / float('${NEW_PARAMS}')):.1f}')")
    echo "🎯 Parameter Efficiency: ${EFFICIENCY}x fewer than PETSA"
fi

echo ""
echo "📁 Logs saved to: ${RESULT_DIR}/"
echo "   • old.log"
echo "   • new.log"
echo ""
echo "✅ Comparison Complete!"
echo ""
