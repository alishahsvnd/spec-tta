#!/bin/bash

###############################################################################
# Test Multiple Strategies to Improve Long-Horizon SPEC-TTA Performance
# Strategies:
#   1. Early stopping (stop when MSE increases)
#   2. Ensemble (blend adapted + baseline)
#   3. Lower learning rate (0.0001 instead of 0.001)
#   4. Reduced beta_freq (0.01 instead of 0.1)
###############################################################################

DATASET="ETTh2"
PRED_LEN=336  # Start with H=336 (catastrophic failure baseline)
MODEL="iTransformer"  # Most stable model

OUTPUT_DIR="./results/LONG_HORIZON_IMPROVEMENTS_ETTh2"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Long-Horizon Improvement Experiments"
echo "Dataset: $DATASET, Model: $MODEL, Horizon: $PRED_LEN"
echo "========================================"

# Baseline (original settings for comparison)
echo ""
echo "=== Experiment 1: Baseline (K=32, LR=0.001, beta=0.1) ==="
python main.py \
  --model_name forecast \
  --model iTransformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  TTA.ENABLE True \
  TTA.METHOD "spec_tta" \
  TTA.SPEC_TTA.K_BINS 32 \
  TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
  TTA.SPEC_TTA.BETA_FREQ 0.1 \
  TTA.SPEC_TTA.LR 0.001 \
  > "${OUTPUT_DIR}/BASELINE_${MODEL}_ETTh2_${PRED_LEN}.txt" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Baseline completed"
    grep -E "(Results without|Final MSE:|test_mse:)" "${OUTPUT_DIR}/BASELINE_${MODEL}_ETTh2_${PRED_LEN}.txt" | tail -3
else
    echo "✗ Baseline failed"
fi

# Strategy 1: Lower learning rate
echo ""
echo "=== Experiment 2: Lower LR (0.0001 instead of 0.001) ==="
python main.py \
  --model_name forecast \
  --model iTransformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  TTA.ENABLE True \
  TTA.METHOD "spec_tta" \
  TTA.SPEC_TTA.K_BINS 32 \
  TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
  TTA.SPEC_TTA.BETA_FREQ 0.1 \
  TTA.SPEC_TTA.LR 0.0001 \
  > "${OUTPUT_DIR}/LOWER_LR_${MODEL}_ETTh2_${PRED_LEN}.txt" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Lower LR completed"
    grep -E "(Results without|Final MSE:|test_mse:)" "${OUTPUT_DIR}/LOWER_LR_${MODEL}_ETTh2_${PRED_LEN}.txt" | tail -3
else
    echo "✗ Lower LR failed"
fi

# Strategy 2: Much lower beta_freq (reduce frequency regularization)
echo ""
echo "=== Experiment 3: Lower beta_freq (0.01 instead of 0.1) ==="
python main.py \
  --model_name forecast \
  --model iTransformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  TTA.ENABLE True \
  TTA.METHOD "spec_tta" \
  TTA.SPEC_TTA.K_BINS 32 \
  TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
  TTA.SPEC_TTA.BETA_FREQ 0.01 \
  TTA.SPEC_TTA.LR 0.001 \
  > "${OUTPUT_DIR}/LOWER_BETA_${MODEL}_ETTh2_${PRED_LEN}.txt" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Lower beta_freq completed"
    grep -E "(Results without|Final MSE:|test_mse:)" "${OUTPUT_DIR}/LOWER_BETA_${MODEL}_ETTh2_${PRED_LEN}.txt" | tail -3
else
    echo "✗ Lower beta_freq failed"
fi

# Strategy 3: Combined (lower LR + lower beta_freq)
echo ""
echo "=== Experiment 4: Combined (LR=0.0001, beta=0.01) ==="
python main.py \
  --model_name forecast \
  --model iTransformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  TTA.ENABLE True \
  TTA.METHOD "spec_tta" \
  TTA.SPEC_TTA.K_BINS 32 \
  TTA.SPEC_TTA.DRIFT_THRESHOLD 0.005 \
  TTA.SPEC_TTA.BETA_FREQ 0.01 \
  TTA.SPEC_TTA.LR 0.0001 \
  > "${OUTPUT_DIR}/COMBINED_${MODEL}_ETTh2_${PRED_LEN}.txt" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Combined completed"
    grep -E "(Results without|Final MSE:|test_mse:)" "${OUTPUT_DIR}/COMBINED_${MODEL}_ETTh2_${PRED_LEN}.txt" | tail -3
else
    echo "✗ Combined failed"
fi

# Strategy 4: Higher drift threshold (less aggressive adaptation)
echo ""
echo "=== Experiment 5: Higher drift threshold (0.02 instead of 0.005) ==="
python main.py \
  --model_name forecast \
  --model iTransformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  TTA.ENABLE True \
  TTA.METHOD "spec_tta" \
  TTA.SPEC_TTA.K_BINS 32 \
  TTA.SPEC_TTA.DRIFT_THRESHOLD 0.02 \
  TTA.SPEC_TTA.BETA_FREQ 0.1 \
  TTA.SPEC_TTA.LR 0.001 \
  > "${OUTPUT_DIR}/HIGHER_DRIFT_${MODEL}_ETTh2_${PRED_LEN}.txt" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Higher drift threshold completed"
    grep -E "(Results without|Final MSE:|test_mse:)" "${OUTPUT_DIR}/HIGHER_DRIFT_${MODEL}_ETTh2_${PRED_LEN}.txt" | tail -3
else
    echo "✗ Higher drift threshold failed"
fi

# Strategy 5: Ultra-conservative (all stability measures)
echo ""
echo "=== Experiment 6: Ultra-conservative (LR=0.00005, beta=0.005, drift=0.02) ==="
python main.py \
  --model_name forecast \
  --model iTransformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  TTA.ENABLE True \
  TTA.METHOD "spec_tta" \
  TTA.SPEC_TTA.K_BINS 32 \
  TTA.SPEC_TTA.DRIFT_THRESHOLD 0.02 \
  TTA.SPEC_TTA.BETA_FREQ 0.005 \
  TTA.SPEC_TTA.LR 0.00005 \
  > "${OUTPUT_DIR}/ULTRA_CONSERVATIVE_${MODEL}_ETTh2_${PRED_LEN}.txt" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Ultra-conservative completed"
    grep -E "(Results without|Final MSE:|test_mse:)" "${OUTPUT_DIR}/ULTRA_CONSERVATIVE_${MODEL}_ETTh2_${PRED_LEN}.txt" | tail -3
else
    echo "✗ Ultra-conservative failed"
fi

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
