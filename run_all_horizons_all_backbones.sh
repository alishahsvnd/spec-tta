#!/bin/bash

# Comprehensive SPEC-TTA vs PETSA Comparison
# Tests all backbones across all horizons for ETTh1 dataset

# Configuration
DATASET="ETTh1"
HORIZONS=(96 192 336 720)
BACKBONES=("iTransformer" "DLinear" "PatchTST" "MICN" "FreTS")

# PETSA configuration
PETSA_RANK=4
PETSA_LOSS_ALPHA=0.1
PETSA_GATING_INIT=0.01

# Result directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_BASE="FULL_COMPARISON_${TIMESTAMP}"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          SPEC-TTA vs PETSA: Full Comparison Experiment                      ║"
echo "║          Dataset: ${DATASET}                                                 ║"
echo "║          Horizons: 96, 192, 336, 720                                         ║"
echo "║          Backbones: iTransformer, DLinear, PatchTST, MICN, FreTS            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Create results directory
mkdir -p "results/${RESULT_BASE}"
mkdir -p "results/${RESULT_BASE}/spec_tta"
mkdir -p "results/${RESULT_BASE}/petsa"
mkdir -p "results/${RESULT_BASE}/training"

# Summary file
SUMMARY_FILE="results/${RESULT_BASE}/SUMMARY.txt"
echo "SPEC-TTA vs PETSA Full Comparison - ${TIMESTAMP}" > "$SUMMARY_FILE"
echo "Dataset: ${DATASET}" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Counters
TOTAL_EXPERIMENTS=$((${#HORIZONS[@]} * ${#BACKBONES[@]}))
COMPLETED=0
FAILED=0

echo "📊 Total experiments to run: ${TOTAL_EXPERIMENTS} per method (${TOTAL_EXPERIMENTS} x 2 = $((TOTAL_EXPERIMENTS * 2)) total)"
echo ""

# Function to train checkpoint if needed
train_checkpoint() {
    local model=$1
    local horizon=$2
    local checkpoint_dir="./checkpoints/${model}/${DATASET}_${horizon}/"
    
    if [ -f "${checkpoint_dir}/checkpoint_best.pth" ]; then
        echo "   ✅ Checkpoint exists: ${checkpoint_dir}"
        return 0
    fi
    
    echo "   🔨 Training checkpoint: ${model} H=${horizon}"
    local train_log="results/${RESULT_BASE}/training/${model}_${DATASET}_${horizon}_train.log"
    
    python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${horizon} \
        MODEL.NAME ${model} \
        MODEL.pred_len ${horizon} \
        TRAIN.ENABLE True \
        TRAIN.CHECKPOINT_DIR ${checkpoint_dir} \
        TEST.ENABLE False \
        TTA.ENABLE False \
        TRAIN.BASE_LR 0.001 \
        TRAIN.WEIGHT_DECAY 0.0 \
        2>&1 | tee "${train_log}" > /dev/null
    
    if [ -f "${checkpoint_dir}/checkpoint_best.pth" ]; then
        echo "   ✅ Training complete: ${checkpoint_dir}"
        return 0
    else
        echo "   ❌ Training failed: ${model} H=${horizon}"
        return 1
    fi
}

# Function to run SPEC-TTA
run_spec_tta() {
    local model=$1
    local horizon=$2
    local checkpoint_dir="./checkpoints/${model}/${DATASET}_${horizon}/"
    local log_file="results/${RESULT_BASE}/spec_tta/${model}_${DATASET}_${horizon}.log"
    local result_dir="SPEC_TTA_${model}_${horizon}"
    
    echo "   🔬 Running SPEC-TTA..."
    
    timeout 600 python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${horizon} \
        MODEL.NAME ${model} \
        MODEL.pred_len ${horizon} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${checkpoint_dir} \
        TEST.ENABLE False \
        TTA.ENABLE True \
        RESULT_DIR ${result_dir} \
        2>&1 | tee "${log_file}" > /dev/null
    
    # Extract results
    local mse=$(grep -E "(Final MSE:|Test MSE:)" "${log_file}" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    local mae=$(grep -E "(Final MAE:|Test MAE:)" "${log_file}" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    local params=$(grep -E "(Total Trainable Parameters:|Total:)" "${log_file}" | tail -1 | grep -oE "[0-9]+")
    
    echo "$model,$horizon,$mse,$mae,$params" >> "results/${RESULT_BASE}/spec_tta_results.csv"
    echo "   ✅ SPEC-TTA: MSE=${mse}, MAE=${mae}, Params=${params}"
}

# Function to run PETSA
run_petsa() {
    local model=$1
    local horizon=$2
    local checkpoint_dir="./checkpoints/${model}/${DATASET}_${horizon}/"
    local log_file="results/${RESULT_BASE}/petsa/${model}_${DATASET}_${horizon}.log"
    local result_dir="PETSA_${model}_${horizon}"
    
    echo "   🔬 Running PETSA..."
    
    timeout 600 python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${horizon} \
        MODEL.NAME ${model} \
        MODEL.pred_len ${horizon} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${checkpoint_dir} \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.PETSA.RANK ${PETSA_RANK} \
        TTA.PETSA.LOSS_ALPHA ${PETSA_LOSS_ALPHA} \
        TTA.PETSA.GATING_INIT ${PETSA_GATING_INIT} \
        RESULT_DIR ${result_dir} \
        2>&1 | tee "${log_file}" > /dev/null
    
    # Extract results
    local mse=$(grep "Test MSE:" "${log_file}" | tail -1 | sed -n 's/.*Test MSE: \([0-9.]*\).*/\1/p')
    local mae=$(grep "Test MAE:" "${log_file}" | tail -1 | sed -n 's/.*Test MAE: \([0-9.]*\).*/\1/p')
    local params=$(grep "Total:" "${log_file}" | tail -1 | grep -oE "[0-9]+")
    
    echo "$model,$horizon,$mse,$mae,$params" >> "results/${RESULT_BASE}/petsa_results.csv"
    echo "   ✅ PETSA: MSE=${mse}, MAE=${mae}, Params=${params}"
}

# Initialize CSV files
echo "Model,Horizon,MSE,MAE,Params" > "results/${RESULT_BASE}/spec_tta_results.csv"
echo "Model,Horizon,MSE,MAE,Params" > "results/${RESULT_BASE}/petsa_results.csv"

# Main experiment loop
START_TIME=$(date +%s)

for HORIZON in "${HORIZONS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📏 HORIZON: ${HORIZON}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    for MODEL in "${BACKBONES[@]}"; do
        echo "┌─────────────────────────────────────────────────────────────────────────┐"
        echo "│ 🧪 Testing: ${MODEL} (H=${HORIZON})                                    "
        echo "└─────────────────────────────────────────────────────────────────────────┘"
        
        # Check/train checkpoint
        if ! train_checkpoint "$MODEL" "$HORIZON"; then
            echo "   ❌ Skipping ${MODEL} H=${HORIZON} - training failed"
            ((FAILED++))
            continue
        fi
        
        # Run SPEC-TTA
        if run_spec_tta "$MODEL" "$HORIZON"; then
            : # Success logged in function
        else
            echo "   ⚠️  SPEC-TTA failed"
        fi
        
        # Run PETSA
        if run_petsa "$MODEL" "$HORIZON"; then
            : # Success logged in function
        else
            echo "   ⚠️  PETSA failed"
        fi
        
        ((COMPLETED++))
        PROGRESS=$((COMPLETED * 100 / TOTAL_EXPERIMENTS))
        echo ""
        echo "   📊 Progress: ${COMPLETED}/${TOTAL_EXPERIMENTS} (${PROGRESS}%)"
        echo ""
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         EXPERIMENT COMPLETE                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "✅ Completed: ${COMPLETED}/${TOTAL_EXPERIMENTS}"
echo "❌ Failed: ${FAILED}/${TOTAL_EXPERIMENTS}"
echo ""
echo "📁 Results saved to: results/${RESULT_BASE}/"
echo ""

# Display results table
echo "📊 Quick Results Summary:"
echo ""
python3 - "$RESULT_BASE" << 'PYTHON_END'
import sys
import csv

result_base = sys.argv[1]

try:
    print("\n" + "="*80)
    print("SPEC-TTA vs PETSA Results")
    print("="*80)
    
    # Read results
    spec_results = {}
    petsa_results = {}
    
    with open(f'results/{result_base}/spec_tta_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['Model']}_H{row['Horizon']}"
            spec_results[key] = row
    
    with open(f'results/{result_base}/petsa_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['Model']}_H{row['Horizon']}"
            petsa_results[key] = row
    
    # Print table
    print(f"\n{'Model':<15} {'H':<5} {'SPEC MSE':<12} {'PETSA MSE':<12} {'Improve':<10} {'Params Ratio':<12}")
    print("-" * 80)
    
    total_spec_mse = 0
    total_petsa_mse = 0
    count = 0
    
    for key in sorted(spec_results.keys()):
        if key in petsa_results:
            spec = spec_results[key]
            petsa = petsa_results[key]
            
            spec_mse = float(spec['MSE']) if spec['MSE'] else 0
            petsa_mse = float(petsa['MSE']) if petsa['MSE'] else 0
            
            if spec_mse > 0 and petsa_mse > 0:
                improvement = ((petsa_mse - spec_mse) / petsa_mse * 100)
                params_ratio = float(petsa['Params']) / float(spec['Params']) if spec['Params'] and petsa['Params'] else 0
                
                print(f"{spec['Model']:<15} {spec['Horizon']:<5} {spec_mse:<12.4f} {petsa_mse:<12.4f} {improvement:>8.1f}% {params_ratio:>9.1f}x")
                
                total_spec_mse += spec_mse
                total_petsa_mse += petsa_mse
                count += 1
    
    if count > 0:
        avg_improvement = ((total_petsa_mse - total_spec_mse) / total_petsa_mse * 100)
        print("-" * 80)
        print(f"{'AVERAGE':<15} {'':<5} {total_spec_mse/count:<12.4f} {total_petsa_mse/count:<12.4f} {avg_improvement:>8.1f}%")
    
    print("\n✅ Detailed results saved to: results/{}/detailed_comparison.csv".format(result_base))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

PYTHON_END

echo ""
echo "🎉 Full comparison experiment complete!"
echo ""
