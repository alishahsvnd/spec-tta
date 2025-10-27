#!/bin/bash

# ========================================================================
# SPEC-TTA HIGH-CAPACITY CONFIGURATION
# Strategy: Match PETSA's accuracy with superior architecture
# ========================================================================
# 
# Configuration: AGGRESSIVE SCALING (k_bins=256)
#   - Parameters: 3,584 (6.5% of PETSA's 55,296)
#   - Still 15x more parameter-efficient than PETSA
#   - Better frequency coverage → better reconstruction
#   - Adaptive features enabled for dynamic adaptation
#
# Expected improvements over baseline SPEC-TTA (k_bins=16, MSE=0.186):
#   1. 16x more frequency bins (256 vs 16) → finer frequency control
#   2. Adaptive bin reselection → tracks distribution shifts
#   3. More aggressive loss weighting → stronger supervision
#   4. Lower drift threshold → earlier adaptation
#   5. Smaller batch size → more frequent updates
#
# Target: Beat PETSA's MSE while maintaining efficiency advantage
# ========================================================================

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
data_name=ETTh1
pred_len=96

# Core SPEC-TTA settings - HIGH CAPACITY
K_BINS=256                    # 16x more than baseline (16 → 256)
PATCH_LEN=24                  # Keep same (good for T=96)
HUBER_DELTA=0.5               # Keep same
LR=1e-3                       # Keep same (1e-3 is optimal)
GRAD_CLIP=1.0                 # Keep same

# Loss weights - MORE AGGRESSIVE
BETA_FREQ=0.15                # Increase from 0.05 → 0.15 (3x stronger frequency supervision)
LAMBDA_PW=1.5                 # Increase from 1.0 → 1.5 (stronger structural loss)
LAMBDA_PROX=1e-4              # Keep same (good regularization)
LAMBDA_HC=0.15                # Increase from 0.1 → 0.15 (horizon consistency)

# Adaptation control - DYNAMIC
DRIFT_THRESHOLD=0.005         # Lower from 0.01 → 0.005 (earlier adaptation)
RESELECTION_EVERY=50          # Enable adaptive bin reselection (was 0=disabled)

# Update schedule - MORE FREQUENT
BATCH_SIZE=32                 # Decrease from 96 → 32 (more updates)

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
echo "SPEC-TTA HIGH-CAPACITY COMPLETED"
echo "========================================================================"
echo "Configuration:"
echo "  - K_BINS: $K_BINS (16x baseline)"
echo "  - Parameters: ~3,584 (6.5% of PETSA)"
echo "  - Batch size: $BATCH_SIZE (3x more updates per epoch)"
echo "  - Adaptive bin reselection: every $RESELECTION_EVERY updates"
echo "  - Loss weights: β_freq=$BETA_FREQ, λ_pw=$LAMBDA_PW, λ_hc=$LAMBDA_HC"
echo "  - Drift threshold: $DRIFT_THRESHOLD (2x more sensitive)"
echo ""
echo "Expected advantages over PETSA:"
echo "  ✓ 15x more parameter-efficient (3,584 vs 55,296 params)"
echo "  ✓ Direct frequency control (no MLP overhead)"
echo "  ✓ Adaptive bin reselection tracks distribution shifts"
echo "  ✓ Finer frequency resolution (256 bins vs PETSA's implicit)"
echo "========================================================================"
