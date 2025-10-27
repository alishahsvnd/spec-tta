#!/bin/bash

# ========================================================================
# SPEC-TTA MODERATE-CAPACITY CONFIGURATION
# Strategy: Minimal parameter increase, maximum algorithmic improvements
# ========================================================================
# 
# Configuration: MODERATE SCALING (k_bins=64)
#   - Parameters: 896 (1.6% of PETSA's 55,296)
#   - Only 4x increase from baseline (16 → 64)
#   - Focus on SMART training rather than brute-force capacity
#
# Philosophy: Prove SPEC-TTA's algorithmic superiority
#   - Better loss design
#   - Smarter adaptation triggers
#   - More efficient update schedule
#   - Can we beat PETSA with <2% of its parameters?
#
# Target: Best accuracy/efficiency ratio
# ========================================================================

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
data_name=ETTh1
pred_len=96

# Core SPEC-TTA settings - MODERATE CAPACITY
K_BINS=64                     # 4x baseline (16 → 64)
PATCH_LEN=24                  # Keep optimal value
HUBER_DELTA=0.5               # Keep optimal value
LR=1.5e-3                     # Slightly higher LR (1e-3 → 1.5e-3)
GRAD_CLIP=1.0                 # Keep same

# Loss weights - BALANCED AGGRESSION
BETA_FREQ=0.1                 # Moderate increase from 0.05 → 0.1
LAMBDA_PW=1.2                 # Slight increase from 1.0 → 1.2
LAMBDA_PROX=1e-4              # Keep same (good regularization)
LAMBDA_HC=0.12                # Slight increase from 0.1 → 0.12

# Adaptation control - SMART TRIGGERS
DRIFT_THRESHOLD=0.007         # Lower threshold (0.01 → 0.007)
RESELECTION_EVERY=100         # Conservative reselection (every 100 updates)

# Update schedule - BALANCED
BATCH_SIZE=48                 # 2x more updates (96 → 48)

python -u main.py \
  --data_path data/$data_name/ \
  --checkpoints checkpoints/$model_name/ \
  --results_dir results/ \
  --model_id "${data_name}_${pred_len}" \
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
  --TTA.MODULE_NAMES_TO_ADAPT spec_tta \
  --TTA.SPEC_TTA.K_BINS $K_BINS \
  --TTA.SPEC_TTA.PATCH_LEN $PATCH_LEN \
  --TTA.SPEC_TTA.HUBER_DELTA $HUBER_DELTA \
  --TTA.SPEC_TTA.BETA_FREQ $BETA_FREQ \
  --TTA.SPEC_TTA.LAMBDA_PW $LAMBDA_PW \
  --TTA.SPEC_TTA.LAMBDA_PROX $LAMBDA_PROX \
  --TTA.SPEC_TTA.LAMBDA_HC $LAMBDA_HC \
  --TTA.SPEC_TTA.DRIFT_THRESHOLD $DRIFT_THRESHOLD \
  --TTA.SPEC_TTA.LR $LR \
  --TTA.SPEC_TTA.GRAD_CLIP $GRAD_CLIP \
  --TTA.SPEC_TTA.RESELECTION_EVERY $RESELECTION_EVERY

echo ""
echo "========================================================================"
echo "SPEC-TTA MODERATE-CAPACITY COMPLETED"
echo "========================================================================"
echo "Configuration:"
echo "  - K_BINS: $K_BINS (4x baseline)"
echo "  - Parameters: ~896 (1.6% of PETSA)"
echo "  - Batch size: $BATCH_SIZE (2x more updates per epoch)"
echo "  - Adaptive bin reselection: every $RESELECTION_EVERY updates"
echo "  - Loss weights: β_freq=$BETA_FREQ, λ_pw=$LAMBDA_PW, λ_hc=$LAMBDA_HC"
echo "  - Drift threshold: $DRIFT_THRESHOLD"
echo ""
echo "Expected advantages:"
echo "  ✓ 62x more parameter-efficient than PETSA (896 vs 55,296 params)"
echo "  ✓ Balanced accuracy/efficiency trade-off"
echo "  ✓ Smart adaptation without brute-force capacity"
echo "  ✓ Best ROI (return on investment in parameters)"
echo "========================================================================"
