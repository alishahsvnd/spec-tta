#!/bin/bash

# ========================================================================
# HIGH-CAPACITY SPEC-TTA FOR PUBLICATION
# Goal: Beat PETSA's MSE for academic publication
# ========================================================================
#
# Configuration: ULTRA CAPACITY (36K params)
#   - Multi-scale spectral adapter (low/mid/high frequencies)
#   - Per-variable low-rank transformations (rank=24)
#   - Ensemble trend models (linear + quadratic + exponential)
#   - Learned gating between frequency scales
#
# Parameters: 36,131 (65% of PETSA's 55,296)
# Still 1.5x more efficient while targeting higher accuracy!
#
# Target: MSE < 0.10 (decisively beat PETSA's 0.112)
# ========================================================================

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
data_name=ETTh1
pred_len=96

# High-capacity multi-scale architecture
K_LOW=10                     # Low-frequency bins
K_MID=20                     # Mid-frequency bins  
K_HIGH=19                    # High-frequency bins (max safe value)
RANK=24                      # Low-rank dimension
GATING_DIM=128               # Gating network hidden dim

# Aggressive training for maximum accuracy
BATCH_SIZE=16                # Small batch = more updates
LR=2e-3                      # Higher LR for faster convergence
GRAD_CLIP=2.0                # Allow larger gradients

# Strong loss weights
BETA_FREQ=0.3                # Very strong frequency supervision
LAMBDA_PW=2.5                # Very strong structural loss
LAMBDA_PROX=5e-5             # Low proximal (less conservative)
LAMBDA_HC=0.25               # Strong horizon consistency

# Sensitive adaptation
DRIFT_THRESHOLD=0.002        # Very low threshold
RESELECTION_EVERY=20         # Frequent reselection

python -u main.py \
  --data_path data/$data_name/ \
  --checkpoints checkpoints/$model_name/ \
  --results_dir results/ \
  --model_id "${data_name}_${pred_len}_ULTRA" \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size $BATCH_SIZE \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3 \
  --TTA.ENABLE True \
  --TTA.MODULE_NAMES_TO_ADAPT spec_tta_high_capacity \
  --TTA.SPEC_TTA_HC.K_LOW $K_LOW \
  --TTA.SPEC_TTA_HC.K_MID $K_MID \
  --TTA.SPEC_TTA_HC.K_HIGH $K_HIGH \
  --TTA.SPEC_TTA_HC.RANK $RANK \
  --TTA.SPEC_TTA_HC.GATING_DIM $GATING_DIM \
  --TTA.SPEC_TTA_HC.LR $LR \
  --TTA.SPEC_TTA_HC.GRAD_CLIP $GRAD_CLIP \
  --TTA.SPEC_TTA_HC.BETA_FREQ $BETA_FREQ \
  --TTA.SPEC_TTA_HC.LAMBDA_PW $LAMBDA_PW \
  --TTA.SPEC_TTA_HC.LAMBDA_PROX $LAMBDA_PROX \
  --TTA.SPEC_TTA_HC.LAMBDA_HC $LAMBDA_HC \
  --TTA.SPEC_TTA_HC.DRIFT_THRESHOLD $DRIFT_THRESHOLD \
  --TTA.SPEC_TTA_HC.RESELECTION_EVERY $RESELECTION_EVERY 2>&1 | tee ultra_capacity_results.log

echo ""
echo "========================================================================"
echo "ULTRA-CAPACITY SPEC-TTA COMPLETED"
echo "========================================================================"
echo "Configuration:"
echo "  - Parameters: 36,131 (65% of PETSA)"
echo "  - Multi-scale: $K_LOW low + $K_MID mid + $K_HIGH high freq bins"
echo "  - Rank: $RANK (per-variable low-rank)"
echo "  - Gating: $GATING_DIM hidden dims"
echo "  - Batch: $BATCH_SIZE (maximum updates)"
echo ""
echo "This is the MAXIMUM ACCURACY configuration for publication."
echo "Should decisively beat PETSA's MSE=0.112 while using 35% fewer params!"
echo "========================================================================"
