#!/usr/bin/env bash
# Quick experiment runner: Single dataset, all horizons, all models
# Usage: ./run_single_dataset.sh <DATASET>
# Example: ./run_single_dataset.sh Weather

# Don't exit on error - we want to continue with other experiments
# set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <DATASET>"
    echo ""
    echo "Available datasets:"
    echo "  ETTh1      - 7 variables (tested)"
    echo "  ETTh2      - 7 variables"
    echo "  Weather    - 21 variables"
    echo "  Exchange   - 8 variables (aperiodic)"
    echo "  Electricity - 321 variables (high-dim)"
    echo "  Traffic    - 862 variables (very high-dim)"
    exit 1
fi

DATASET=$1
RESULT_BASE="./results/SPEC_TTA_BENCHMARK_${DATASET}"
mkdir -p "$RESULT_BASE"

# Standard hyperparameters
HUBER_DELTA=0.5
LAMBDA_PW=1.0
LAMBDA_PROX=0.0001
LAMBDA_HC=0.1
GRAD_CLIP=1.0
LR=0.001
PATCH_LEN=24

# Dataset-specific configurations
case $DATASET in
    "ETTh1"|"ETTh2"|"Weather")
        K_BINS=32
        DRIFT_THRESHOLD=0.005
        BETA_FREQ=0.1
        ;;
    "Exchange")
        K_BINS=16
        DRIFT_THRESHOLD=0.01
        BETA_FREQ=0.05
        ;;
    "Electricity")
        K_BINS=16
        DRIFT_THRESHOLD=0.005
        BETA_FREQ=0.1
        ;;
    "Traffic")
        K_BINS=8
        DRIFT_THRESHOLD=0.01
        BETA_FREQ=0.1
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

echo "========================================"
echo "SPEC-TTA Experiment: $DATASET"
echo "========================================"
echo "Configuration:"
echo "  K_BINS: $K_BINS"
echo "  DRIFT_THRESHOLD: $DRIFT_THRESHOLD"
echo "  BETA_FREQ: $BETA_FREQ"
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
        echo "Running: $MODEL on $DATASET (horizon $PRED_LEN)"
        echo "----------------------------------------"
        
        CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
        RESULT_DIR="${RESULT_BASE}/${MODEL}/"
        OUTPUT_FILE="${RESULT_BASE}/SPEC_TTA_${MODEL}_${DATASET}_${PRED_LEN}.txt"
        
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
echo "Experiment complete for $DATASET"
echo "========================================"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Results: $RESULT_BASE"
echo "========================================"
