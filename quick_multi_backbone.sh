#!/bin/bash
# Quick Multi-Backbone Test - Sequential Execution

cd /home/alishah/PETSA

RESULT_DIR="results/MULTI_BACKBONE_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULT_DIR}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    Multi-Backbone SPEC-TTA Test (ETTh1 H=96)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Test each backbone
test_backbone() {
    local name=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Testing: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ ! -f "checkpoints/$name/ETTh1_96/checkpoint_best.pth" ]; then
        echo "⚠️  Checkpoint not found, skipping..."
        echo ""
        return
    fi
    
    python main.py \
        DATA.NAME ETTh1 \
        DATA.PRED_LEN 96 \
        MODEL.NAME $name \
        MODEL.pred_len 96 \
        TRAIN.ENABLE False \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.SPEC_TTA.K_BINS 32 \
        TRAIN.CHECKPOINT_DIR "./checkpoints/$name/ETTh1_96/" \
        RESULT_DIR "$RESULT_DIR/$name/" 2>&1 | tee "$RESULT_DIR/$name.log" | grep -E "Final MSE:|Final MAE:|Total.*Parameters:|Quality Level:|HYBRID"
    
    echo ""
}

# Run tests
test_backbone "iTransformer"
test_backbone "DLinear"
test_backbone "PatchTST"
test_backbone "MICN"
test_backbone "FreTS"

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    SUMMARY                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

printf "%-15s | %-10s | %-10s | %-10s\n" "Backbone" "MSE" "MAE" "Parameters"
echo "─────────────────────────────────────────────────────────────────"

for name in iTransformer DLinear PatchTST MICN FreTS; do
    if [ -f "$RESULT_DIR/$name.log" ]; then
        MSE=$(grep "Final MSE:" "$RESULT_DIR/$name.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
        MAE=$(grep "Final MAE:" "$RESULT_DIR/$name.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
        PARAMS=$(grep "Total Trainable Parameters:" "$RESULT_DIR/$name.log" | grep -oE "[0-9]+" | head -1)
        printf "%-15s | %-10s | %-10s | %-10s\n" "$name" "${MSE:-N/A}" "${MAE:-N/A}" "${PARAMS:-N/A}"
    fi
done

echo "─────────────────────────────────────────────────────────────────"
echo ""
echo "📁 Results: $RESULT_DIR/"
echo ""
