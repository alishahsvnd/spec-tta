#!/bin/bash
# scripts/iTransformer/ETTh1_96/run_spec_tta_improved.sh
# Enhanced SPEC-TTA with adaptive threshold and other improvements

GPU=${1:-0}
K_BINS=${2:-16}
BETA_FREQ=${3:-0.05}
DRIFT_THRESH=${4:-0.001}  # Lower default threshold

CUDA_VISIBLE_DEVICES=$GPU python main.py \
  --config config.py \
  MODEL.NAME "iTransformer" \
  DATASET.DATA "ETTh1" \
  DATASET.LOOKBACK 96 \
  DATASET.HORIZON 96 \
  TTA.METHOD "SPEC_TTA" \
  TTA.SPEC_TTA.K_BINS $K_BINS \
  TTA.SPEC_TTA.BETA_FREQ $BETA_FREQ \
  TTA.SPEC_TTA.DRIFT_THRESHOLD $DRIFT_THRESH \
  TTA.SPEC_TTA.LR 0.005 \
  TTA.SPEC_TTA.LAMBDA_PW 1.0 \
  TTA.SPEC_TTA.LAMBDA_PROX 1e-4 \
  TTA.SPEC_TTA.LAMBDA_HC 0.1 \
  TTA.SPEC_TTA.GRAD_CLIP 1.0 \
  TTA.SPEC_TTA.HUBER_DELTA 0.5 \
  TTA.SPEC_TTA.PATCH_LEN 24 \
  TTA.SPEC_TTA.RESELECTION_EVERY 0 \
  CHECKPOINT.LOAD_DIR "./checkpoints/iTransformer/ETTh1_96" \
  RESULT_DIR "SPEC_TTA_IMPROVED_KBINS_${K_BINS}_BETA_${BETA_FREQ}_DRIFT_${DRIFT_THRESH}"

echo ""
echo "============================================"
echo "SPEC-TTA (IMPROVED) completed!"
echo "GPU: $GPU"
echo "K_BINS: $K_BINS"
echo "BETA_FREQ: $BETA_FREQ"
echo "DRIFT_THRESHOLD: $DRIFT_THRESH (lowered for more updates)"
echo "LR: 0.005 (increased for faster adaptation)"
echo "============================================"
