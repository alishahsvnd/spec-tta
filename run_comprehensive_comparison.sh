#!/bin/bash

# ============================================================================
# COMPREHENSIVE SPEC-TTA vs PETSA COMPARISON
# Datasets: ETTh1, ETTh2, ETTm1, weather, electricity
# Backbones: iTransformer, PatchTST, DLinear
# Horizons: 96, 192, 336, 720
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║          COMPREHENSIVE SPEC-TTA vs PETSA BENCHMARK COMPARISON             ║"
echo "║                                                                            ║"
echo "║  Datasets:  ETTh1, ETTh2, ETTm1, weather, electricity                     ║"
echo "║  Backbones: iTransformer, PatchTST, DLinear                               ║"
echo "║  Horizons:  96, 192, 336, 720                                             ║"
echo "║                                                                            ║"
echo "║  SPEC-TTA Config: K_BINS=32, BETA_FREQ=0.1, DRIFT_THRESHOLD=0.005        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
DATASETS=("ETTh1" "ETTh2" "ETTm1" "weather" "electricity")
BACKBONES=("iTransformer" "PatchTST" "DLinear")
HORIZONS=(96 192 336 720)

# SPEC-TTA optimal configuration
K_BINS=32
BETA_FREQ=0.1
DRIFT_THRESHOLD=0.005
LAMBDA_PW=1.0
LAMBDA_PROX=0.0001
LR=0.001

# Results directory
RESULT_BASE="./results/COMPREHENSIVE_COMPARISON"
LOG_FILE="comprehensive_comparison.log"

# Create results directory
mkdir -p "$RESULT_BASE"

# Start logging
echo "Starting comprehensive comparison at $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to check if checkpoint exists
check_checkpoint() {
    local model=$1
    local dataset=$2
    local horizon=$3
    local checkpoint_path="./checkpoints/${model}/${dataset}_${horizon}/checkpoint_best.pth"
    
    if [ -f "$checkpoint_path" ]; then
        return 0
    else
        echo "⚠️  Checkpoint not found: $checkpoint_path" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Function to run SPEC-TTA experiment
run_spec_tta() {
    local dataset=$1
    local backbone=$2
    local horizon=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
    echo "🔬 SPEC-TTA: $backbone on $dataset (H=$horizon)" | tee -a "$LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
    
    local checkpoint_dir="./checkpoints/${backbone}/${dataset}_${horizon}/"
    local result_dir="${RESULT_BASE}/SPEC_TTA/${backbone}/${dataset}_${horizon}/"
    
    if ! check_checkpoint "$backbone" "$dataset" "$horizon"; then
        echo "❌ Skipping: Checkpoint not available" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        return 1
    fi
    
    mkdir -p "$result_dir"
    
    python main.py \
        DATA.NAME "$dataset" \
        DATA.PRED_LEN "$horizon" \
        MODEL.NAME "$backbone" \
        MODEL.pred_len "$horizon" \
        TRAIN.ENABLE False \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.SPEC_TTA.K_BINS "$K_BINS" \
        TTA.SPEC_TTA.DRIFT_THRESHOLD "$DRIFT_THRESHOLD" \
        TTA.SPEC_TTA.BETA_FREQ "$BETA_FREQ" \
        TTA.SPEC_TTA.LAMBDA_PW "$LAMBDA_PW" \
        TTA.SPEC_TTA.LAMBDA_PROX "$LAMBDA_PROX" \
        TTA.SPEC_TTA.LR "$LR" \
        TRAIN.CHECKPOINT_DIR "$checkpoint_dir" \
        RESULT_DIR "$result_dir" \
        2>&1 | tee "${result_dir}/run.log"
    
    # Extract results
    local mse=$(grep "Final MSE:" "${result_dir}/run.log" | tail -1 | awk '{print $3}')
    local mae=$(grep "Final MAE:" "${result_dir}/run.log" | tail -1 | awk '{print $3}')
    local updates=$(grep "Total Adaptation Updates:" "${result_dir}/run.log" | tail -1 | awk '{print $4}')
    
    echo "✅ SPEC-TTA Results: MSE=$mse, MAE=$mae, Updates=$updates" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Save summary
    echo "SPEC-TTA,$dataset,$backbone,$horizon,$mse,$mae,$updates" >> "${RESULT_BASE}/results_summary.csv"
}

# Function to run PETSA experiment
run_petsa() {
    local dataset=$1
    local backbone=$2
    local horizon=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
    echo "🔬 PETSA: $backbone on $dataset (H=$horizon)" | tee -a "$LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
    
    local checkpoint_dir="./checkpoints/${backbone}/${dataset}_${horizon}/"
    local result_dir="${RESULT_BASE}/PETSA/${backbone}/${dataset}_${horizon}/"
    
    if ! check_checkpoint "$backbone" "$dataset" "$horizon"; then
        echo "❌ Skipping: Checkpoint not available" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        return 1
    fi
    
    mkdir -p "$result_dir"
    
    python main.py \
        DATA.NAME "$dataset" \
        DATA.PRED_LEN "$horizon" \
        MODEL.NAME "$backbone" \
        MODEL.pred_len "$horizon" \
        TRAIN.ENABLE False \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TRAIN.CHECKPOINT_DIR "$checkpoint_dir" \
        RESULT_DIR "$result_dir" \
        2>&1 | tee "${result_dir}/run.log"
    
    # Extract results
    local mse=$(grep "Final MSE:" "${result_dir}/run.log" | tail -1 | awk '{print $3}')
    local mae=$(grep "Final MAE:" "${result_dir}/run.log" | tail -1 | awk '{print $3}')
    local updates=$(grep "Total Adaptation Updates:" "${result_dir}/run.log" | tail -1 | awk '{print $4}')
    
    echo "✅ PETSA Results: MSE=$mse, MAE=$mae, Updates=$updates" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Save summary
    echo "PETSA,$dataset,$backbone,$horizon,$mse,$mae,$updates" >> "${RESULT_BASE}/results_summary.csv"
}

# Initialize CSV
echo "Method,Dataset,Backbone,Horizon,MSE,MAE,Updates" > "${RESULT_BASE}/results_summary.csv"

# Main experiment loop
total_experiments=0
completed_experiments=0

for dataset in "${DATASETS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        for horizon in "${HORIZONS[@]}"; do
            total_experiments=$((total_experiments + 2))  # SPEC-TTA + PETSA
        done
    done
done

echo "Total experiments to run: $total_experiments" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

current=0

for dataset in "${DATASETS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "╔════════════════════════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
    echo "║  DATASET: $dataset" | tee -a "$LOG_FILE"
    echo "╚════════════════════════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    for backbone in "${BACKBONES[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "┌────────────────────────────────────────────────────────────────────────────┐" | tee -a "$LOG_FILE"
        echo "│  Backbone: $backbone" | tee -a "$LOG_FILE"
        echo "└────────────────────────────────────────────────────────────────────────────┘" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        for horizon in "${HORIZONS[@]}"; do
            current=$((current + 1))
            echo "Progress: [$current/$total_experiments]" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            
            # Run SPEC-TTA
            if run_spec_tta "$dataset" "$backbone" "$horizon"; then
                completed_experiments=$((completed_experiments + 1))
            fi
            
            current=$((current + 1))
            echo "Progress: [$current/$total_experiments]" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            
            # Run PETSA
            if run_petsa "$dataset" "$backbone" "$horizon"; then
                completed_experiments=$((completed_experiments + 1))
            fi
            
            echo "" | tee -a "$LOG_FILE"
        done
    done
done

# Generate comparison report
echo "" | tee -a "$LOG_FILE"
echo "╔════════════════════════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║                        GENERATING COMPARISON REPORT                        ║" | tee -a "$LOG_FILE"
echo "╚════════════════════════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create Python analysis script
cat > "${RESULT_BASE}/analyze_results.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# Read results
df = pd.read_csv('results_summary.csv')

# Convert MSE and MAE to float
df['MSE'] = pd.to_numeric(df['MSE'], errors='coerce')
df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')

# Remove rows with missing data
df = df.dropna(subset=['MSE', 'MAE'])

# Pivot to compare methods
comparison = []

for dataset in df['Dataset'].unique():
    for backbone in df['Backbone'].unique():
        for horizon in df['Horizon'].unique():
            spec_tta = df[(df['Method'] == 'SPEC-TTA') & 
                         (df['Dataset'] == dataset) & 
                         (df['Backbone'] == backbone) & 
                         (df['Horizon'] == horizon)]
            
            petsa = df[(df['Method'] == 'PETSA') & 
                      (df['Dataset'] == dataset) & 
                      (df['Backbone'] == backbone) & 
                      (df['Horizon'] == horizon)]
            
            if not spec_tta.empty and not petsa.empty:
                spec_mse = spec_tta['MSE'].values[0]
                petsa_mse = petsa['MSE'].values[0]
                spec_mae = spec_tta['MAE'].values[0]
                petsa_mae = petsa['MAE'].values[0]
                
                improvement_mse = ((petsa_mse - spec_mse) / petsa_mse) * 100
                improvement_mae = ((petsa_mae - spec_mae) / petsa_mae) * 100
                
                comparison.append({
                    'Dataset': dataset,
                    'Backbone': backbone,
                    'Horizon': horizon,
                    'SPEC-TTA MSE': spec_mse,
                    'PETSA MSE': petsa_mse,
                    'MSE Improvement (%)': improvement_mse,
                    'SPEC-TTA MAE': spec_mae,
                    'PETSA MAE': petsa_mae,
                    'MAE Improvement (%)': improvement_mae,
                    'Winner': 'SPEC-TTA' if spec_mse < petsa_mse else 'PETSA'
                })

comparison_df = pd.DataFrame(comparison)

# Save detailed comparison
comparison_df.to_csv('detailed_comparison.csv', index=False)

print("\n" + "="*100)
print("COMPREHENSIVE COMPARISON RESULTS")
print("="*100 + "\n")

# Summary by dataset
print("\n📊 SUMMARY BY DATASET:")
print("-" * 100)
for dataset in comparison_df['Dataset'].unique():
    dataset_df = comparison_df[comparison_df['Dataset'] == dataset]
    avg_mse_imp = dataset_df['MSE Improvement (%)'].mean()
    avg_mae_imp = dataset_df['MAE Improvement (%)'].mean()
    wins = len(dataset_df[dataset_df['Winner'] == 'SPEC-TTA'])
    total = len(dataset_df)
    
    print(f"\n{dataset}:")
    print(f"  Avg MSE Improvement: {avg_mse_imp:+.2f}%")
    print(f"  Avg MAE Improvement: {avg_mae_imp:+.2f}%")
    print(f"  Win Rate: {wins}/{total} ({100*wins/total:.1f}%)")

# Summary by backbone
print("\n\n🏗️  SUMMARY BY BACKBONE:")
print("-" * 100)
for backbone in comparison_df['Backbone'].unique():
    backbone_df = comparison_df[comparison_df['Backbone'] == backbone]
    avg_mse_imp = backbone_df['MSE Improvement (%)'].mean()
    avg_mae_imp = backbone_df['MAE Improvement (%)'].mean()
    wins = len(backbone_df[backbone_df['Winner'] == 'SPEC-TTA'])
    total = len(backbone_df)
    
    print(f"\n{backbone}:")
    print(f"  Avg MSE Improvement: {avg_mse_imp:+.2f}%")
    print(f"  Avg MAE Improvement: {avg_mae_imp:+.2f}%")
    print(f"  Win Rate: {wins}/{total} ({100*wins/total:.1f}%)")

# Summary by horizon
print("\n\n📏 SUMMARY BY HORIZON:")
print("-" * 100)
for horizon in sorted(comparison_df['Horizon'].unique()):
    horizon_df = comparison_df[comparison_df['Horizon'] == horizon]
    avg_mse_imp = horizon_df['MSE Improvement (%)'].mean()
    avg_mae_imp = horizon_df['MAE Improvement (%)'].mean()
    wins = len(horizon_df[horizon_df['Winner'] == 'SPEC-TTA'])
    total = len(horizon_df)
    
    print(f"\nH={horizon}:")
    print(f"  Avg MSE Improvement: {avg_mse_imp:+.2f}%")
    print(f"  Avg MAE Improvement: {avg_mae_imp:+.2f}%")
    print(f"  Win Rate: {wins}/{total} ({100*wins/total:.1f}%)")

# Overall summary
print("\n\n🎯 OVERALL SUMMARY:")
print("-" * 100)
avg_mse_imp = comparison_df['MSE Improvement (%)'].mean()
avg_mae_imp = comparison_df['MAE Improvement (%)'].mean()
wins = len(comparison_df[comparison_df['Winner'] == 'SPEC-TTA'])
total = len(comparison_df)

print(f"\nTotal Experiments: {total}")
print(f"SPEC-TTA Wins: {wins} ({100*wins/total:.1f}%)")
print(f"PETSA Wins: {total-wins} ({100*(total-wins)/total:.1f}%)")
print(f"\nAverage MSE Improvement: {avg_mse_imp:+.2f}%")
print(f"Average MAE Improvement: {avg_mae_imp:+.2f}%")

# Top improvements
print("\n\n🏆 TOP 10 IMPROVEMENTS (by MSE):")
print("-" * 100)
top_10 = comparison_df.nlargest(10, 'MSE Improvement (%)')
for idx, row in top_10.iterrows():
    print(f"{row['Dataset']:12} {row['Backbone']:15} H={row['Horizon']:3}  "
          f"MSE: {row['MSE Improvement (%)']:+6.2f}%  MAE: {row['MAE Improvement (%)']:+6.2f}%")

# Worst cases
print("\n\n⚠️  CASES WHERE PETSA WINS:")
print("-" * 100)
petsa_wins = comparison_df[comparison_df['Winner'] == 'PETSA'].sort_values('MSE Improvement (%)')
if len(petsa_wins) > 0:
    for idx, row in petsa_wins.iterrows():
        print(f"{row['Dataset']:12} {row['Backbone']:15} H={row['Horizon']:3}  "
              f"MSE: {row['MSE Improvement (%)']:+6.2f}%  MAE: {row['MAE Improvement (%)']:+6.2f}%")
else:
    print("None - SPEC-TTA wins all experiments! 🎉")

print("\n" + "="*100 + "\n")
print("Detailed results saved to: detailed_comparison.csv")
print("="*100 + "\n")
PYTHON_EOF

chmod +x "${RESULT_BASE}/analyze_results.py"

# Run analysis
cd "$RESULT_BASE"
python analyze_results.py | tee -a "../$LOG_FILE"
cd -

echo "" | tee -a "$LOG_FILE"
echo "╔════════════════════════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║                          BENCHMARK COMPLETE                                ║" | tee -a "$LOG_FILE"
echo "╚════════════════════════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Completed: $completed_experiments / $total_experiments experiments" | tee -a "$LOG_FILE"
echo "Results directory: $RESULT_BASE" | tee -a "$LOG_FILE"
echo "Summary CSV: ${RESULT_BASE}/results_summary.csv" | tee -a "$LOG_FILE"
echo "Detailed comparison: ${RESULT_BASE}/detailed_comparison.csv" | tee -a "$LOG_FILE"
echo "Full log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
