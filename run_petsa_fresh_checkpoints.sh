#!/bin/bash

# Run PETSA on Fresh Checkpoints
# Tests all 5 newly trained backbones with PETSA for comparison

DATASET="ETTh1"
PRED_LEN=96
RANK=4
LOSS_ALPHA=0.1
GATING_INIT=0.01

# Backbones to test
BACKBONES=("iTransformer" "DLinear" "PatchTST" "MICN" "FreTS")

# Result directory
RESULT_DIR="PETSA_FRESH_CHECKPOINTS_$(date +%Y%m%d_%H%M%S)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PETSA Test on Fresh Checkpoints                             ║"
echo "║  Dataset: ${DATASET}, Horizon: ${PRED_LEN}                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "results/${RESULT_DIR}"

# Store results
declare -A RESULTS_MSE
declare -A RESULTS_MAE
declare -A RESULTS_PARAMS

for MODEL in "${BACKBONES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Testing PETSA on: ${MODEL}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    
    # Check if checkpoint exists
    if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        echo "❌ Checkpoint not found: ${CHECKPOINT_DIR}/checkpoint_best.pth"
        echo ""
        continue
    fi
    
    echo "Checkpoint: ${CHECKPOINT_DIR}"
    echo "Running PETSA..."
    echo ""
    
    # Run PETSA
    LOG_FILE="results/${RESULT_DIR}/${MODEL}.log"
    
    python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.PETSA.RANK ${RANK} \
        TTA.PETSA.LOSS_ALPHA ${LOSS_ALPHA} \
        TTA.PETSA.GATING_INIT ${GATING_INIT} \
        RESULT_DIR ${RESULT_DIR} \
        2>&1 | tee "${LOG_FILE}"
    
    # Extract results
    if [ -f "${LOG_FILE}" ]; then
        MSE=$(grep "Test MSE:" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
        MAE=$(grep "Test MAE:" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
        PARAMS=$(grep "Total:" "${LOG_FILE}" | tail -1 | grep -oE "[0-9]+")
        
        RESULTS_MSE[$MODEL]=$MSE
        RESULTS_MAE[$MODEL]=$MAE
        RESULTS_PARAMS[$MODEL]=$PARAMS
        
        echo "✅ ${MODEL} Complete:"
        echo "   MSE: ${MSE}"
        echo "   MAE: ${MAE}"
        echo "   Params: ${PARAMS}"
    else
        echo "❌ ${MODEL} Failed - no log file"
        RESULTS_MSE[$MODEL]="FAILED"
        RESULTS_MAE[$MODEL]="FAILED"
        RESULTS_PARAMS[$MODEL]="FAILED"
    fi
    
    echo ""
done

# Print summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║            PETSA RESULTS - FRESH CHECKPOINTS                             ║"
echo "║            Dataset: ${DATASET}, Horizon: ${PRED_LEN}                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
printf "%-15s | %-10s | %-10s | %-10s\n" "Backbone" "MSE" "MAE" "Params"
echo "────────────────────────────────────────────────────────────────────────────"

for MODEL in "${BACKBONES[@]}"; do
    MSE=${RESULTS_MSE[$MODEL]:-"N/A"}
    MAE=${RESULTS_MAE[$MODEL]:-"N/A"}
    PARAMS=${RESULTS_PARAMS[$MODEL]:-"N/A"}
    printf "%-15s | %-10s | %-10s | %-10s\n" "$MODEL" "$MSE" "$MAE" "$PARAMS"
done

echo "────────────────────────────────────────────────────────────────────────────"
echo ""
echo "📁 Full logs: results/${RESULT_DIR}/"
echo ""

# Calculate average (excluding failures)
total_mse=0
total_mae=0
total_params=0
count=0

for MODEL in "${BACKBONES[@]}"; do
    MSE=${RESULTS_MSE[$MODEL]}
    MAE=${RESULTS_MAE[$MODEL]}
    PARAMS=${RESULTS_PARAMS[$MODEL]}
    
    if [[ "$MSE" != "FAILED" && "$MSE" != "" ]]; then
        total_mse=$(echo "$total_mse + $MSE" | bc)
        total_mae=$(echo "$total_mae + $MAE" | bc)
        total_params=$(echo "$total_params + $PARAMS" | bc)
        ((count++))
    fi
done

if [ $count -gt 0 ]; then
    avg_mse=$(echo "scale=6; $total_mse / $count" | bc)
    avg_mae=$(echo "scale=6; $total_mae / $count" | bc)
    avg_params=$(echo "scale=0; $total_params / $count" | bc)
    
    echo "📊 AVERAGE (${count} models):"
    echo "   MSE: ${avg_mse}"
    echo "   MAE: ${avg_mae}"
    echo "   Params: ${avg_params}"
    echo ""
fi

echo "✅ PETSA testing complete!"
echo ""
