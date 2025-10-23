#!/bin/bash
# Multi-Backbone Experiment: Test SPEC-TTA across different architectures
# Backbones: iTransformer, DLinear, PatchTST, MICN, FreTS
# Dataset: ETTh1, Horizon: 96

set -e

cd /home/alishah/PETSA

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║              MULTI-BACKBONE EXPERIMENT: SPEC-TTA PHASE 1+2                ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Testing SPEC-TTA with Phase 1+2 (Quality Detection + Hybrid Mode) across:"
echo "  • iTransformer (attention-based)"
echo "  • DLinear (linear decomposition)"
echo "  • PatchTST (patch-based transformer)"
echo "  • MICN (multi-scale convolution)"
echo "  • FreTS (frequency-based)"
echo ""
echo "Dataset: ETTh1, Horizon: 96"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/MULTI_BACKBONE_${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

# Common parameters
DATA_NAME="ETTh1"
PRED_LEN=96

# Define backbones to test
BACKBONES=("iTransformer" "DLinear" "PatchTST" "MICN" "FreTS")

# Results storage
declare -A RESULTS_MSE
declare -A RESULTS_MAE
declare -A RESULTS_PARAMS
declare -A RESULTS_UPDATES
declare -A RESULTS_QUALITY
declare -A RESULTS_HYBRID

# Function to run experiment for a backbone
run_experiment() {
    local backbone=$1
    local checkpoint_dir="./checkpoints/${backbone}/ETTh1_96/"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Testing: ${backbone}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Checkpoint: ${checkpoint_dir}"
    echo ""
    
    # Check if checkpoint exists
    if [ ! -f "${checkpoint_dir}/checkpoint_best.pth" ]; then
        echo "⚠️  Checkpoint not found for ${backbone}, skipping..."
        RESULTS_MSE[${backbone}]="N/A"
        RESULTS_MAE[${backbone}]="N/A"
        RESULTS_PARAMS[${backbone}]="N/A"
        RESULTS_UPDATES[${backbone}]="N/A"
        RESULTS_QUALITY[${backbone}]="N/A"
        RESULTS_HYBRID[${backbone}]="N/A"
        return
    fi
    
    echo "Running SPEC-TTA with Phase 1+2..."
    
    # Run experiment
    python main.py \
        DATA.NAME ${DATA_NAME} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${backbone} \
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
        TRAIN.CHECKPOINT_DIR "${checkpoint_dir}" \
        RESULT_DIR "${RESULT_DIR}/${backbone}/" > "${RESULT_DIR}/${backbone}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ ${backbone} completed"
        
        # Extract results
        MSE=$(grep "Final MSE:" "${RESULT_DIR}/${backbone}.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
        MAE=$(grep "Final MAE:" "${RESULT_DIR}/${backbone}.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
        PARAMS=$(grep -E "Created SPEC-TTA modules with|Total Trainable Parameters:" "${RESULT_DIR}/${backbone}.log" | grep -oE "[0-9]+" | head -1)
        UPDATES=$(grep "Total Adaptation Updates:" "${RESULT_DIR}/${backbone}.log" | grep -oE "[0-9]+" | head -1)
        QUALITY=$(grep "Quality Level:" "${RESULT_DIR}/${backbone}.log" | grep -oE "POOR|FAIR|GOOD|EXCELLENT" | head -1)
        HYBRID=$(grep -c "Enabling HYBRID mode" "${RESULT_DIR}/${backbone}.log")
        
        # Store results
        RESULTS_MSE[${backbone}]=${MSE:-"N/A"}
        RESULTS_MAE[${backbone}]=${MAE:-"N/A"}
        RESULTS_PARAMS[${backbone}]=${PARAMS:-"N/A"}
        RESULTS_UPDATES[${backbone}]=${UPDATES:-"N/A"}
        RESULTS_QUALITY[${backbone}]=${QUALITY:-"N/A"}
        RESULTS_HYBRID[${backbone}]=$([[ ${HYBRID} -gt 0 ]] && echo 'YES' || echo 'NO')
        
        echo "   MSE: ${RESULTS_MSE[${backbone}]}"
        echo "   MAE: ${RESULTS_MAE[${backbone}]}"
        echo "   Parameters: ${RESULTS_PARAMS[${backbone}]}"
        echo "   Updates: ${RESULTS_UPDATES[${backbone}]}"
        echo "   Quality: ${RESULTS_QUALITY[${backbone}]}"
        echo "   Hybrid Mode: ${RESULTS_HYBRID[${backbone}]}"
    else
        echo "❌ ${backbone} failed (check ${RESULT_DIR}/${backbone}.log)"
        RESULTS_MSE[${backbone}]="ERROR"
        RESULTS_MAE[${backbone}]="ERROR"
        RESULTS_PARAMS[${backbone}]="ERROR"
        RESULTS_UPDATES[${backbone}]="ERROR"
        RESULTS_QUALITY[${backbone}]="ERROR"
        RESULTS_HYBRID[${backbone}]="ERROR"
    fi
    
    echo ""
    sleep 1
}

# Run experiments for all backbones
for backbone in "${BACKBONES[@]}"; do
    run_experiment "${backbone}"
done

# Generate comparison report
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                   📊 MULTI-BACKBONE RESULTS SUMMARY                       ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset: ETTh1, Horizon: 96, Method: SPEC-TTA Phase 1+2"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-12s | %-8s\n" "Backbone" "MSE" "MAE" "Parameters" "Updates" "Quality" "Hybrid"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for backbone in "${BACKBONES[@]}"; do
    printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-12s | %-8s\n" \
        "${backbone}" \
        "${RESULTS_MSE[${backbone}]}" \
        "${RESULTS_MAE[${backbone}]}" \
        "${RESULTS_PARAMS[${backbone}]}" \
        "${RESULTS_UPDATES[${backbone}]}" \
        "${RESULTS_QUALITY[${backbone}]}" \
        "${RESULTS_HYBRID[${backbone}]}"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Find best and worst
echo "🏆 Performance Ranking (by MSE):"
echo ""

# Create sorted list (excluding N/A and ERROR)
SORTED_BACKBONES=()
for backbone in "${BACKBONES[@]}"; do
    mse=${RESULTS_MSE[${backbone}]}
    if [[ "${mse}" != "N/A" && "${mse}" != "ERROR" ]]; then
        SORTED_BACKBONES+=("${mse} ${backbone}")
    fi
done

# Sort and display
IFS=$'\n' SORTED=($(sort -n <<<"${SORTED_BACKBONES[*]}"))
unset IFS

rank=1
for entry in "${SORTED[@]}"; do
    mse=$(echo ${entry} | awk '{print $1}')
    backbone=$(echo ${entry} | awk '{print $2}')
    
    medal=""
    if [ ${rank} -eq 1 ]; then
        medal="🥇"
    elif [ ${rank} -eq 2 ]; then
        medal="🥈"
    elif [ ${rank} -eq 3 ]; then
        medal="🥉"
    else
        medal="${rank}."
    fi
    
    printf "%s %-15s: MSE=%-10s MAE=%-10s Params=%-10s\n" \
        "${medal}" \
        "${backbone}" \
        "${mse}" \
        "${RESULTS_MAE[${backbone}]}" \
        "${RESULTS_PARAMS[${backbone}]}"
    
    ((rank++))
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Quality distribution
echo "📊 Checkpoint Quality Distribution:"
echo ""
QUALITY_COUNTS=()
for quality in "EXCELLENT" "GOOD" "FAIR" "POOR"; do
    count=0
    for backbone in "${BACKBONES[@]}"; do
        if [[ "${RESULTS_QUALITY[${backbone}]}" == "${quality}" ]]; then
            ((count++))
        fi
    done
    if [ ${count} -gt 0 ]; then
        echo "   ${quality}: ${count} checkpoint(s)"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Hybrid mode activation
echo "🚀 Hybrid Mode Activation:"
echo ""
HYBRID_COUNT=0
for backbone in "${BACKBONES[@]}"; do
    if [[ "${RESULTS_HYBRID[${backbone}]}" == "YES" ]]; then
        echo "   ✅ ${backbone}: Hybrid mode activated"
        ((HYBRID_COUNT++))
    elif [[ "${RESULTS_HYBRID[${backbone}]}" == "NO" ]]; then
        echo "   ⚪ ${backbone}: Frequency-only mode"
    fi
done

echo ""
echo "   Total with hybrid mode: ${HYBRID_COUNT} / ${#BACKBONES[@]}"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Parameter efficiency
echo "💡 Parameter Efficiency (vs PETSA baseline ~25,934 params):"
echo ""
for backbone in "${BACKBONES[@]}"; do
    params=${RESULTS_PARAMS[${backbone}]}
    if [[ "${params}" != "N/A" && "${params}" != "ERROR" ]]; then
        efficiency=$(echo "scale=1; 25934 / ${params}" | bc 2>/dev/null || echo "N/A")
        printf "   %-15s: %5s params (%sx fewer than PETSA)\n" "${backbone}" "${params}" "${efficiency}"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Average performance
echo "📈 Average Performance Across All Backbones:"
echo ""

# Calculate averages
total_mse=0
total_mae=0
total_params=0
total_updates=0
count=0

for backbone in "${BACKBONES[@]}"; do
    mse=${RESULTS_MSE[${backbone}]}
    mae=${RESULTS_MAE[${backbone}]}
    params=${RESULTS_PARAMS[${backbone}]}
    updates=${RESULTS_UPDATES[${backbone}]}
    
    if [[ "${mse}" != "N/A" && "${mse}" != "ERROR" ]]; then
        total_mse=$(echo "${total_mse} + ${mse}" | bc)
        total_mae=$(echo "${total_mae} + ${mae}" | bc)
        total_params=$((total_params + params))
        total_updates=$((total_updates + updates))
        ((count++))
    fi
done

if [ ${count} -gt 0 ]; then
    avg_mse=$(echo "scale=6; ${total_mse} / ${count}" | bc)
    avg_mae=$(echo "scale=6; ${total_mae} / ${count}" | bc)
    avg_params=$((total_params / count))
    avg_updates=$((total_updates / count))
    
    echo "   Average MSE: ${avg_mse}"
    echo "   Average MAE: ${avg_mae}"
    echo "   Average Parameters: ${avg_params}"
    echo "   Average Updates: ${avg_updates}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Save detailed report
REPORT_FILE="${RESULT_DIR}/SUMMARY_REPORT.txt"
{
    echo "MULTI-BACKBONE EXPERIMENT RESULTS"
    echo "=================================="
    echo ""
    echo "Dataset: ETTh1"
    echo "Horizon: 96"
    echo "Method: SPEC-TTA Phase 1+2 (Quality Detection + Hybrid Mode)"
    echo "Date: $(date)"
    echo ""
    echo "Results:"
    echo "--------"
    echo ""
    printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-12s | %-8s\n" "Backbone" "MSE" "MAE" "Parameters" "Updates" "Quality" "Hybrid"
    echo "--------------------------------------------------------------------------------------------------------"
    for backbone in "${BACKBONES[@]}"; do
        printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-12s | %-8s\n" \
            "${backbone}" \
            "${RESULTS_MSE[${backbone}]}" \
            "${RESULTS_MAE[${backbone}]}" \
            "${RESULTS_PARAMS[${backbone}]}" \
            "${RESULTS_UPDATES[${backbone}]}" \
            "${RESULTS_QUALITY[${backbone}]}" \
            "${RESULTS_HYBRID[${backbone}]}"
    done
    echo ""
    echo "Averages (successful runs only):"
    echo "  MSE: ${avg_mse}"
    echo "  MAE: ${avg_mae}"
    echo "  Parameters: ${avg_params}"
    echo "  Updates: ${avg_updates}"
} > "${REPORT_FILE}"

echo "📁 Results saved to: ${RESULT_DIR}/"
echo "   • Individual logs: ${RESULT_DIR}/<backbone>.log"
echo "   • Summary report: ${REPORT_FILE}"
echo ""
echo "✅ Multi-Backbone Experiment Complete!"
echo ""
