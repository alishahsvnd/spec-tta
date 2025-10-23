#!/usr/bin/env bash
# Comprehensive SPEC-TTA Benchmark: All Datasets × All Horizons × Key Backbones
# Based on: spec_tta_complete_www2026.tex
# Author: Generated for WWW 2026 submission
# Date: October 23, 2025

set -e  # Exit on error

echo "=================================================="
echo "SPEC-TTA Comprehensive Benchmark Suite"
echo "=================================================="
echo "Datasets: ETTh1, ETTh2, Weather, Exchange, Electricity"
echo "Horizons: 96, 192, 336, 720"
echo "Backbones: iTransformer, DLinear, PatchTST"
echo "Total Experiments: 5 datasets × 4 horizons × 3 backbones = 60"
echo "=================================================="
echo ""

# Base result directory
RESULT_BASE="./results/SPEC_TTA_FULL_BENCHMARK"
mkdir -p "$RESULT_BASE"

# SPEC-TTA hyperparameters (standard across all experiments)
HUBER_DELTA=0.5
LAMBDA_PW=1.0
LAMBDA_PROX=0.0001
LAMBDA_HC=0.1
GRAD_CLIP=1.0
LR=0.001
PATCH_LEN=24

# Track progress
TOTAL_RUNS=60
COMPLETED=0
FAILED=0

# Function to run a single experiment
run_experiment() {
    local DATASET=$1
    local PRED_LEN=$2
    local MODEL=$3
    local K_BINS=$4
    local DRIFT_THRESHOLD=$5
    local BETA_FREQ=$6
    
    echo ""
    echo "========================================"
    echo "[$((COMPLETED+1))/$TOTAL_RUNS] Running: $MODEL on $DATASET (horizon $PRED_LEN)"
    echo "========================================"
    
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    RESULT_DIR="${RESULT_BASE}/${DATASET}/${MODEL}/"
    OUTPUT_FILE="${RESULT_BASE}/${DATASET}/SPEC_TTA_${MODEL}_${DATASET}_${PRED_LEN}.txt"
    
    # Check checkpoint
    if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        echo "⚠️  WARNING: Checkpoint not found at ${CHECKPOINT_DIR}/checkpoint_best.pth"
        echo "   Skipping this experiment..."
        ((FAILED++))
        return
    fi
    
    echo "✓ Checkpoint: ${CHECKPOINT_DIR}/checkpoint_best.pth"
    echo "✓ Config: K_BINS=$K_BINS, DRIFT_THRESHOLD=$DRIFT_THRESHOLD, BETA_FREQ=$BETA_FREQ"
    echo ""
    
    # Run experiment
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
        echo "✓ Completed successfully!"
        ((COMPLETED++))
    else
        echo "✗ FAILED!"
        ((FAILED++))
    fi
}

# =============================================================================
# PHASE 1: ETTh1 (Already tested, include for completeness)
# =============================================================================
DATASET="ETTh1"
K_BINS=32
DRIFT_THRESHOLD=0.005
BETA_FREQ=0.1

echo ""
echo "=========================================="
echo "PHASE 1: ETTh1 (7 variables)"
echo "=========================================="

for PRED_LEN in 96 192 336 720; do
    for MODEL in "iTransformer" "DLinear" "PatchTST"; do
        run_experiment "$DATASET" "$PRED_LEN" "$MODEL" "$K_BINS" "$DRIFT_THRESHOLD" "$BETA_FREQ"
    done
done

# =============================================================================
# PHASE 2: ETTh2
# =============================================================================
DATASET="ETTh2"
K_BINS=32
DRIFT_THRESHOLD=0.005
BETA_FREQ=0.1

echo ""
echo "=========================================="
echo "PHASE 2: ETTh2 (7 variables)"
echo "=========================================="

for PRED_LEN in 96 192 336 720; do
    for MODEL in "iTransformer" "DLinear" "PatchTST"; do
        run_experiment "$DATASET" "$PRED_LEN" "$MODEL" "$K_BINS" "$DRIFT_THRESHOLD" "$BETA_FREQ"
    done
done

# =============================================================================
# PHASE 3: Weather
# =============================================================================
DATASET="Weather"
K_BINS=32
DRIFT_THRESHOLD=0.005
BETA_FREQ=0.1

echo ""
echo "=========================================="
echo "PHASE 3: Weather (21 variables)"
echo "=========================================="

for PRED_LEN in 96 192 336 720; do
    for MODEL in "iTransformer" "DLinear" "PatchTST"; do
        run_experiment "$DATASET" "$PRED_LEN" "$MODEL" "$K_BINS" "$DRIFT_THRESHOLD" "$BETA_FREQ"
    done
done

# =============================================================================
# PHASE 4: Exchange Rate (Aperiodic - adjust params)
# =============================================================================
DATASET="Exchange"
K_BINS=16  # Fewer bins for less periodic signal
DRIFT_THRESHOLD=0.01  # Higher threshold
BETA_FREQ=0.05  # Lower frequency weight

echo ""
echo "=========================================="
echo "PHASE 4: Exchange Rate (8 variables, aperiodic)"
echo "=========================================="

for PRED_LEN in 96 192 336 720; do
    for MODEL in "iTransformer" "DLinear" "PatchTST"; do
        run_experiment "$DATASET" "$PRED_LEN" "$MODEL" "$K_BINS" "$DRIFT_THRESHOLD" "$BETA_FREQ"
    done
done

# =============================================================================
# PHASE 5: Electricity (High-dimensional - reduce K to control params)
# =============================================================================
DATASET="Electricity"
K_BINS=16  # 321 vars × 16 bins = ~20K params
DRIFT_THRESHOLD=0.005
BETA_FREQ=0.1

echo ""
echo "=========================================="
echo "PHASE 5: Electricity (321 variables, high-dim)"
echo "=========================================="

for PRED_LEN in 96 192 336 720; do
    for MODEL in "iTransformer" "DLinear" "PatchTST"; do
        run_experiment "$DATASET" "$PRED_LEN" "$MODEL" "$K_BINS" "$DRIFT_THRESHOLD" "$BETA_FREQ"
    done
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=================================================="
echo "BENCHMARK COMPLETE"
echo "=================================================="
echo "Total runs: $TOTAL_RUNS"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", 100*$COMPLETED/$TOTAL_RUNS}")%"
echo ""
echo "Results saved to: $RESULT_BASE"
echo "Next step: Run analyze_all_results.py to extract metrics"
echo "=================================================="
