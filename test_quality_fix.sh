#!/bin/bash
# Test quality assessment fix

cd /home/alishah/PETSA || exit 1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Testing Quality Assessment Fix on iTransformer              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Expected: Quality should now be EXCELLENT (not POOR)"
echo "Expected: Hybrid mode should NOT activate for excellent checkpoint"
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
    TRAIN.CHECKPOINT_DIR "./checkpoints/iTransformer/ETTh1_96/" \
    RESULT_DIR "./results/SPEC_TTA_QUALITY_FIX_TEST/" \
    2>&1 | tee quality_fix_test.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Results:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -E "Quality Level:|Baseline MSE:|Final MSE:|Final MAE:|HYBRID|Total Trainable Parameters:" quality_fix_test.log | head -10
echo ""
