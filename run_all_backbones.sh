#!/bin/bash
# Comprehensive Multi-Backbone SPEC-TTA Test

cd /home/alishah/PETSA || exit 1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_BASE="results/MULTI_BACKBONE_FINAL_${TIMESTAMP}"
mkdir -p "${RESULT_BASE}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Multi-Backbone SPEC-TTA Test - Phase 1+2 (Hybrid Mode)     ║"
echo "║  Dataset: ETTh1, Horizon: 96                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Array to store results
declare -A mse mae params quality hybrid_used
BACKBONES=("iTransformer" "DLinear" "PatchTST" "MICN" "FreTS")

# Test each backbone
for BACKBONE in "${BACKBONES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Testing: $BACKBONE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ ! -f "checkpoints/$BACKBONE/ETTh1_96/checkpoint_best.pth" ]; then
        echo "⚠️  Checkpoint not found: checkpoints/$BACKBONE/ETTh1_96/checkpoint_best.pth"
        echo "   Skipping..."
        echo ""
        continue
    fi
    
    LOG_FILE="${RESULT_BASE}/${BACKBONE}.log"
    
    echo "Running SPEC-TTA Phase 1+2..."
    python main.py \
        DATA.NAME ETTh1 \
        DATA.PRED_LEN 96 \
        MODEL.NAME $BACKBONE \
        MODEL.pred_len 96 \
        TRAIN.ENABLE False \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.SPEC_TTA.K_BINS 32 \
        TRAIN.CHECKPOINT_DIR "./checkpoints/$BACKBONE/ETTh1_96/" \
        RESULT_DIR "./results/SPEC_TTA_${BACKBONE}_${TIMESTAMP}/" \
        > "$LOG_FILE" 2>&1
    
    # Extract metrics
    if [ -f "$LOG_FILE" ]; then
        mse[$BACKBONE]=$(grep "Final MSE:" "$LOG_FILE" | grep -oE "[0-9]+\.[0-9]+" | head -1)
        mae[$BACKBONE]=$(grep "Final MAE:" "$LOG_FILE" | grep -oE "[0-9]+\.[0-9]+" | head -1)
        params[$BACKBONE]=$(grep "Total Trainable Parameters:" "$LOG_FILE" | grep -oE "[0-9]+" | head -1)
        quality[$BACKBONE]=$(grep "Quality Level:" "$LOG_FILE" | grep -oE "EXCELLENT|GOOD|FAIR|POOR" | head -1)
        
        # Check if hybrid mode was used
        if grep -q "HYBRID MODE ACTIVATED" "$LOG_FILE"; then
            hybrid_used[$BACKBONE]="YES"
        else
            hybrid_used[$BACKBONE]="NO"
        fi
        
        echo "✅ MSE: ${mse[$BACKBONE]:-N/A}"
        echo "✅ MAE: ${mae[$BACKBONE]:-N/A}"
        echo "✅ Params: ${params[$BACKBONE]:-N/A}"
        echo "✅ Quality: ${quality[$BACKBONE]:-N/A}"
        echo "✅ Hybrid: ${hybrid_used[$BACKBONE]:-N/A}"
    else
        echo "❌ Log file not found"
    fi
    
    echo ""
done

# Generate comprehensive report
REPORT="${RESULT_BASE}/SUMMARY_REPORT.txt"

{
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║            MULTI-BACKBONE SPEC-TTA RESULTS (Phase 1+2)                  ║"
    echo "║            Dataset: ETTh1, Horizon: 96                                    ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-8s\n" "Backbone" "MSE" "MAE" "Params" "Quality" "Hybrid"
    echo "──────────────────────────────────────────────────────────────────────────────────"
    
    for BACKBONE in "${BACKBONES[@]}"; do
        printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-8s\n" \
            "$BACKBONE" \
            "${mse[$BACKBONE]:-N/A}" \
            "${mae[$BACKBONE]:-N/A}" \
            "${params[$BACKBONE]:-N/A}" \
            "${quality[$BACKBONE]:-N/A}" \
            "${hybrid_used[$BACKBONE]:-N/A}"
    done
    
    echo "──────────────────────────────────────────────────────────────────────────────────"
    echo ""
    
    # Calculate statistics
    echo "📊 STATISTICS:"
    echo ""
    
    # Count quality levels
    excellent=0
    good=0
    fair=0
    poor=0
    hybrid_count=0
    
    for BACKBONE in "${BACKBONES[@]}"; do
        case "${quality[$BACKBONE]}" in
            EXCELLENT) ((excellent++)) ;;
            GOOD) ((good++)) ;;
            FAIR) ((fair++)) ;;
            POOR) ((poor++)) ;;
        esac
        
        if [ "${hybrid_used[$BACKBONE]}" = "YES" ]; then
            ((hybrid_count++))
        fi
    done
    
    echo "Quality Distribution:"
    echo "  • EXCELLENT: $excellent"
    echo "  • GOOD: $good"
    echo "  • FAIR: $fair"
    echo "  • POOR: $poor"
    echo ""
    
    echo "Hybrid Mode:"
    echo "  • Used: $hybrid_count / ${#BACKBONES[@]} backbones"
    echo ""
    
    # Performance ranking
    echo "🏆 PERFORMANCE RANKING (by MSE):"
    echo ""
    
    # Sort by MSE
    for BACKBONE in $(for b in "${BACKBONES[@]}"; do
        if [ -n "${mse[$b]}" ]; then
            echo "${mse[$b]} $b"
        fi
    done | sort -n | awk '{print $2}'); do
        echo "  $BACKBONE: MSE=${mse[$BACKBONE]}, MAE=${mae[$BACKBONE]}"
    done
    
    echo ""
    echo "📁 Full logs: $RESULT_BASE/"
    echo ""
    
} | tee "$REPORT"

cat "$REPORT"
