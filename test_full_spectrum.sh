#!/bin/bash

# ========================================================================
# QUICK DEMO: SPEC-TTA Full Spectrum with Aggressive Training
# ========================================================================
# 
# Test if using ALL available frequency bins (49) + aggressive training
# can beat PETSA while maintaining 79x parameter efficiency.
#
# Parameters: 700 (vs PETSA's 55,296)
# Strategy: Maximum frequency coverage + smart training
# ========================================================================

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
data_name=ETTh1
pred_len=96

# Use ALL available frequency bins
K_BINS=49                     # Maximum for L=96
BATCH_SIZE=32                 # More frequent updates
BETA_FREQ=0.15                # Strong frequency supervision
LAMBDA_PW=1.5                 # Strong structural loss
DRIFT_THRESHOLD=0.005         # Sensitive adaptation
RESELECTION_EVERY=50          # Adaptive reselection

python -u main.py \
  --data_path data/$data_name/ \
  --checkpoints checkpoints/$model_name/ \
  --results_dir results/ \
  --model_id "${data_name}_${pred_len}" \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size $BATCH_SIZE \
  --TTA.ENABLE True \
  --TTA.MODULE_NAMES_TO_ADAPT spec_tta \
  --TTA.SPEC_TTA.K_BINS $K_BINS \
  --TTA.SPEC_TTA.BETA_FREQ $BETA_FREQ \
  --TTA.SPEC_TTA.LAMBDA_PW $LAMBDA_PW \
  --TTA.SPEC_TTA.DRIFT_THRESHOLD $DRIFT_THRESHOLD \
  --TTA.SPEC_TTA.RESELECTION_EVERY $RESELECTION_EVERY 2>&1 | tee spec_tta_full_spectrum_test.log

echo ""
echo "========================================================================"
echo "Full Spectrum SPEC-TTA Test Complete"
echo "Parameters: 700 (79x more efficient than PETSA's 55,296)"
echo "Check results to see if we beat PETSA's MSE=0.112!"
echo "========================================================================"
