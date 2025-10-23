#!/usr/bin/env bash
# Run PETSA experiments for comparison with SPEC-TTA
# Usage: ./run_petsa_etth2.sh

DATASET="ETTh2"
RESULT_BASE="./results/PETSA_COMPARISON_ETTh2"
mkdir -p "$RESULT_BASE"

# PETSA hyperparameters (from paper)
RANK=4
LOSS_ALPHA=0.1
GATING_INIT=0.01
LR=0.001
WEIGHT_DECAY=0.0

echo "========================================"
echo "PETSA Experiments: $DATASET"
echo "========================================"
echo "Configuration:"
echo "  RANK: $RANK"
echo "  LOSS_ALPHA: $LOSS_ALPHA"
echo "  GATING_INIT: $GATING_INIT"
echo ""
echo "Models: iTransformer, DLinear, PatchTST"
echo "Horizons: 96, 192, 336, 720"
echo "Total: 12 experiments"
echo "========================================"
echo ""

COMPLETED=0
FAILED=0

for PRED_LEN in 96 192 336 720; do
    for MODEL in "iTransformer" "DLinear" "PatchTST"; do
        echo ""
        echo "Running: PETSA with $MODEL on $DATASET (horizon $PRED_LEN)"
        echo "----------------------------------------"
        
        CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
        RESULT_DIR="${RESULT_BASE}/${MODEL}/"
        mkdir -p "$RESULT_DIR"
        OUTPUT_FILE="${RESULT_BASE}/PETSA_${MODEL}_${DATASET}_${PRED_LEN}.txt"
        
        if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
            echo "⚠️  Checkpoint not found, skipping..."
            ((FAILED++))
            continue
        fi
        
        python -u main.py \
            DATA.NAME "$DATASET" \
            DATA.PRED_LEN "$PRED_LEN" \
            MODEL.NAME "$MODEL" \
            MODEL.pred_len "$PRED_LEN" \
            TRAIN.ENABLE False \
            TRAIN.CHECKPOINT_DIR "$CHECKPOINT_DIR" \
            TTA.ENABLE True \
            TTA.SOLVER.BASE_LR "$LR" \
            TTA.SOLVER.WEIGHT_DECAY "$WEIGHT_DECAY" \
            TTA.PETSA.GATING_INIT "$GATING_INIT" \
            TTA.PETSA.RANK "$RANK" \
            TTA.PETSA.LOSS_ALPHA "$LOSS_ALPHA" \
            RESULT_DIR "$RESULT_DIR" \
            2>&1 | tee "$OUTPUT_FILE"
        
        if [ $? -eq 0 ]; then
            echo "✓ Success"
            ((COMPLETED++))
        else
            echo "✗ Failed"
            ((FAILED++))
        fi
    done
done

echo ""
echo "========================================"
echo "PETSA Experiments complete for $DATASET"
echo "========================================"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Results: $RESULT_BASE"
echo "========================================"
