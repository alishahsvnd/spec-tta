#!/usr/bin/env bash
# Test adaptive K_BINS for long horizons
# Hypothesis: K should scale with horizon to capture more frequency components
# Formula: K = max(32, H // 8)

DATASET="ETTh2"
RESULT_BASE="./results/SPEC_TTA_ADAPTIVE_K_ETTh2"
mkdir -p "$RESULT_BASE"

# Standard hyperparameters
HUBER_DELTA=0.5
LAMBDA_PW=1.0
LAMBDA_PROX=0.0001
LAMBDA_HC=0.1
GRAD_CLIP=1.0
LR=0.001
PATCH_LEN=24
DRIFT_THRESHOLD=0.005
BETA_FREQ=0.1

echo "========================================"
echo "SPEC-TTA with Adaptive K: $DATASET"
echo "========================================"
echo "Configuration:"
echo "  K_BINS: Adaptive (32 for H=96, 42 for H=336, 90 for H=720)"
echo "  Formula: K = max(32, H // 8)"
echo "  DRIFT_THRESHOLD: $DRIFT_THRESHOLD"
echo "  BETA_FREQ: $BETA_FREQ"
echo ""
echo "Models: iTransformer, DLinear, PatchTST"
echo "Horizons: 336, 720 (testing long horizons)"
echo "Total: 6 experiments"
echo "========================================"
echo ""

COMPLETED=0
FAILED=0

for PRED_LEN in 336 720; do
    # Adaptive K: K = max(32, H // 8)
    K_BINS=$((PRED_LEN / 8))
    if [ $K_BINS -lt 32 ]; then
        K_BINS=32
    fi
    
    echo ""
    echo "=== Horizon $PRED_LEN: Using K_BINS=$K_BINS ==="
    echo ""
    
    for MODEL in "iTransformer" "DLinear" "PatchTST"; do
        echo ""
        echo "Running: $MODEL on $DATASET (horizon $PRED_LEN, K=$K_BINS)"
        echo "----------------------------------------"
        
        CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
        RESULT_DIR="${RESULT_BASE}/${MODEL}/"
        mkdir -p "$RESULT_DIR"
        OUTPUT_FILE="${RESULT_BASE}/ADAPTIVE_K${K_BINS}_${MODEL}_${DATASET}_${PRED_LEN}.txt"
        
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
            TTA.SPEC_TTA.K_BINS "$K_BINS" \
            TTA.SPEC_TTA.PATCH_LEN "$PATCH_LEN" \
            TTA.SPEC_TTA.HUBER_DELTA "$HUBER_DELTA" \
            TTA.SPEC_TTA.BETA_FREQ "$BETA_FREQ" \
            TTA.SPEC_TTA.LAMBDA_PW "$LAMBDA_PW" \
            TTA.SPEC_TTA.LAMBDA_PROX "$LAMBDA_PROX" \
            TTA.SPEC_TTA.LAMBDA_HC "$LAMBDA_HC" \
            TTA.SPEC_TTA.DRIFT_THRESHOLD "$DRIFT_THRESHOLD" \
            TTA.SPEC_TTA.LR "$LR" \
            TTA.SPEC_TTA.GRAD_CLIP "$GRAD_CLIP" \
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
echo "Adaptive K Experiments complete"
echo "========================================"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Results: $RESULT_BASE"
echo ""
echo "Summary:"
echo "  H=336: K=42 (vs original K=32)"
echo "  H=720: K=90 (vs original K=32)"
echo "========================================"
