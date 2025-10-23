#!/bin/bash
# Sanity check: Verify SPEC-TTA works after reverting PT mode change
# Test on H=96 where it previously worked well

set -e

DATASET="ETTh2"
MODEL="iTransformer"
HORIZON=96
OUTPUT_DIR="results/SPEC_TTA_SANITY_CHECK"

echo "=== SPEC-TTA Sanity Check After Reverting PT Mode ==="
echo "Testing: $MODEL on $DATASET H=$HORIZON"
echo "Expected: MSE ~0.23 (14% improvement over baseline 0.266)"
echo ""

mkdir -p "$OUTPUT_DIR/$MODEL"

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"

if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
    echo "ERROR: Checkpoint not found"
    exit 1
fi

python -u main.py \
    DATA.NAME "$DATASET" \
    DATA.PRED_LEN "$HORIZON" \
    MODEL.NAME "$MODEL" \
    MODEL.pred_len "$HORIZON" \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR "$CHECKPOINT_DIR" \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TTA.SPEC_TTA.PATCH_LEN 24 \
    TTA.SPEC_TTA.HUBER_DELTA 0.5 \
    TTA.SPEC_TTA.BETA_FREQ 0.05 \
    TTA.SPEC_TTA.LAMBDA_PW 1.0 \
    TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
    TTA.SPEC_TTA.LAMBDA_HC 0.1 \
    TTA.SPEC_TTA.DRIFT_THRESHOLD 0.01 \
    TTA.SPEC_TTA.LR 0.001 \
    TTA.SPEC_TTA.GRAD_CLIP 1.0 \
    RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
    2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt"

echo ""
echo "=== Results ==="
grep -E "(Final MSE|Final MAE|Total Adaptation)" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt"

echo ""
echo "=== Comparison ==="
echo "Baseline:       MSE=0.2656"
echo "Previous SPEC:  MSE=0.2279 (+14.2% improvement)"
echo "Current run:    $(grep "Final MSE" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt")"

python3 << 'PYEOF'
import re

# Read current result
with open("results/SPEC_TTA_SANITY_CHECK/iTransformer/ETTh2_96.txt") as f:
    content = f.read()
    mse_match = re.search(r'Final MSE: ([\d.]+)', content)
    if mse_match:
        current_mse = float(mse_match.group(1))
        baseline_mse = 0.2656
        
        print(f"\n✓ Sanity Check:")
        if current_mse < 0.5:  # Should be around 0.23
            gain = (baseline_mse - current_mse) / baseline_mse * 100
            print(f"  PASS: MSE={current_mse:.4f} ({gain:+.1f}% vs baseline)")
            print(f"  SPEC-TTA restored to working state!")
        else:
            print(f"  FAIL: MSE={current_mse:.4f} (should be ~0.23)")
            print(f"  Something is still broken!")
PYEOF

echo ""
echo "Complete!"
