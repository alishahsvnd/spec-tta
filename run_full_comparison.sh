#!/bin/bash

# ========================================================================
# COMPREHENSIVE COMPARISON: SPEC-TTA vs PETSA
# Runs all configurations and generates comparison table
# ========================================================================

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
data_name=ETTh1
pred_len=96

RESULTS_FILE="comparison_results.txt"
echo "========================================================================" > $RESULTS_FILE
echo "SPEC-TTA vs PETSA COMPREHENSIVE COMPARISON" >> $RESULTS_FILE
echo "Dataset: $data_name, Horizon: $pred_len" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "========================================================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Function to run experiment and extract metrics
run_experiment() {
    local method=$1
    local config_name=$2
    local extra_args=$3
    
    echo "========================================================================"
    echo "Running: $config_name"
    echo "========================================================================"
    
    log_file="${method}_${config_name}_results.log"
    
    python -u main.py \
      --data_path data/$data_name/ \
      --checkpoints checkpoints/$model_name/ \
      --results_dir results/ \
      --model_id "${data_name}_${pred_len}_${config_name}" \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 4 \
      --d_model 512 \
      --d_ff 512 \
      --batch_size 256 \
      --TTA.ENABLE True \
      --TTA.MODULE_NAMES_TO_ADAPT $method \
      $extra_args 2>&1 | tee $log_file
    
    # Extract MSE, MAE, and param count from log
    mse=$(grep -oP "MSE:\s+\K[0-9.]+|mse.*?:\s*\K[0-9.]+" $log_file | tail -1)
    mae=$(grep -oP "MAE:\s+\K[0-9.]+|mae.*?:\s*\K[0-9.]+" $log_file | tail -1)
    params=$(grep -oP "TOTAL.*?:\s*\K[0-9,]+" $log_file | tail -1 | tr -d ',')
    updates=$(grep -oP "n_adapt.*?:\s*\K[0-9]+" $log_file | tail -1)
    
    echo "$config_name|$mse|$mae|$params|$updates" >> $RESULTS_FILE
    
    echo ""
    echo "Completed: $config_name"
    echo "MSE: $mse, MAE: $mae, Params: $params, Updates: $updates"
    echo ""
}

echo "Starting experiments..."
echo ""

# Baseline: No TTA
echo "========================================================================"
echo "1/5: No-TTA Baseline"
echo "========================================================================"
python -u main.py \
  --data_path data/$data_name/ \
  --checkpoints checkpoints/$model_name/ \
  --results_dir results/ \
  --model_id "${data_name}_${pred_len}_NoTTA" \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 256 \
  --TTA.ENABLE False 2>&1 | tee no_tta_results.log

mse=$(grep -oP "MSE:\s+\K[0-9.]+|mse.*?:\s*\K[0-9.]+" no_tta_results.log | tail -1)
mae=$(grep -oP "MAE:\s+\K[0-9.]+|mae.*?:\s*\K[0-9.]+" no_tta_results.log | tail -1)
echo "No-TTA|$mse|$mae|0|0" >> $RESULTS_FILE

echo ""
echo "========================================================================"
echo "2/5: PETSA Baseline"
echo "========================================================================"
run_experiment "petsa" "PETSA" ""

echo ""
echo "========================================================================"
echo "3/5: SPEC-TTA Medium Capacity (12K params)"
echo "========================================================================"
run_experiment "spec_tta_hc" "SPEC_Medium" "--TTA.SPEC_TTA_HC.MODE medium --TTA.SPEC_TTA_HC.K_LOW 6 --TTA.SPEC_TTA_HC.K_MID 12 --TTA.SPEC_TTA_HC.K_HIGH 20 --TTA.SPEC_TTA_HC.RANK 8 --TTA.SPEC_TTA_HC.GATING_DIM 32"

echo ""
echo "========================================================================"
echo "4/5: SPEC-TTA High Capacity (24K params)"
echo "========================================================================"
run_experiment "spec_tta_hc" "SPEC_High" "--TTA.SPEC_TTA_HC.MODE high --TTA.SPEC_TTA_HC.K_LOW 8 --TTA.SPEC_TTA_HC.K_MID 16 --TTA.SPEC_TTA_HC.K_HIGH 25 --TTA.SPEC_TTA_HC.RANK 16 --TTA.SPEC_TTA_HC.GATING_DIM 64"

echo ""
echo "========================================================================"
echo "5/5: SPEC-TTA Ultra Capacity (36K params)"
echo "========================================================================"
run_experiment "spec_tta_hc" "SPEC_Ultra" "--TTA.SPEC_TTA_HC.MODE ultra --TTA.SPEC_TTA_HC.K_LOW 10 --TTA.SPEC_TTA_HC.K_MID 20 --TTA.SPEC_TTA_HC.K_HIGH 19 --TTA.SPEC_TTA_HC.RANK 24 --TTA.SPEC_TTA_HC.GATING_DIM 128"

# Generate comparison table
echo "" >> $RESULTS_FILE
echo "========================================================================" >> $RESULTS_FILE
echo "RESULTS TABLE" >> $RESULTS_FILE
echo "========================================================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

python3 << 'EOF' >> $RESULTS_FILE
import sys

# Read results
with open('comparison_results.txt', 'r') as f:
    lines = f.readlines()

# Find data lines
data_lines = [l.strip() for l in lines if '|' in l and not l.startswith('=')]

print(f"{'Method':<20} {'MSE':<10} {'MAE':<10} {'Params':<12} {'Updates':<10} {'vs PETSA':<12}")
print("-" * 85)

petsa_mse = None
for line in data_lines:
    parts = line.split('|')
    if len(parts) >= 5:
        method, mse, mae, params, updates = parts[0], parts[1], parts[2], parts[3], parts[4]
        
        # Convert to numbers
        try:
            mse_val = float(mse) if mse else 0.0
            mae_val = float(mae) if mae else 0.0
            params_val = int(params) if params else 0
            updates_val = int(updates) if updates else 0
            
            # Track PETSA MSE
            if 'PETSA' in method:
                petsa_mse = mse_val
            
            # Calculate improvement
            if petsa_mse and mse_val > 0:
                improvement = ((petsa_mse - mse_val) / petsa_mse) * 100
                vs_petsa = f"{improvement:+.1f}%"
            else:
                vs_petsa = "N/A"
            
            print(f"{method:<20} {mse:<10} {mae:<10} {params_val:<12,} {updates_val:<10} {vs_petsa:<12}")
        except:
            pass

print()
print("Note: Positive % means better (lower MSE) than PETSA")
print("      Negative % means worse (higher MSE) than PETSA")
EOF

echo "" >> $RESULTS_FILE
echo "========================================================================" >> $RESULTS_FILE
echo "ANALYSIS" >> $RESULTS_FILE
echo "========================================================================" >> $RESULTS_FILE

# Display results
cat $RESULTS_FILE

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo "Results saved to: $RESULTS_FILE"
echo "Individual logs:"
echo "  - no_tta_results.log"
echo "  - petsa_PETSA_results.log"
echo "  - spec_tta_hc_SPEC_Medium_results.log"
echo "  - spec_tta_hc_SPEC_High_results.log"
echo "  - spec_tta_hc_SPEC_Ultra_results.log"
echo "========================================================================"
