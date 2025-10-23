#!/bin/bash

# ============================================================================
# QUICK SPEC-TTA vs PETSA COMPARISON
# Focus: ETTh1, ETTh2 with iTransformer (most important baseline)
# Horizons: 96, 192, 336, 720
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              QUICK SPEC-TTA vs PETSA COMPARISON (ETTh1, ETTh2)            ║"
echo "║                        Backbone: iTransformer                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
DATASETS=("ETTh1" "ETTh2")
BACKBONE="iTransformer"
HORIZONS=(96 192 336 720)

# SPEC-TTA optimal configuration
K_BINS=32
BETA_FREQ=0.1
DRIFT_THRESHOLD=0.005
LAMBDA_PW=1.0
LAMBDA_PROX=0.0001
LR=0.001

# Results directory
RESULT_BASE="./results/QUICK_COMPARISON"
LOG_FILE="quick_comparison.log"

mkdir -p "$RESULT_BASE"

echo "Starting quick comparison at $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Initialize CSV
echo "Method,Dataset,Backbone,Horizon,MSE,MAE,Updates,Params" > "${RESULT_BASE}/results_summary.csv"

for dataset in "${DATASETS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "╔════════════════════════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
    echo "║  DATASET: $dataset" | tee -a "$LOG_FILE"
    echo "╚════════════════════════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    for horizon in "${HORIZONS[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
        echo "Horizon: $horizon" | tee -a "$LOG_FILE"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
        
        # SPEC-TTA
        echo "" | tee -a "$LOG_FILE"
        echo "🔬 Running SPEC-TTA..." | tee -a "$LOG_FILE"
        
        checkpoint_dir="./checkpoints/${BACKBONE}/${dataset}_${horizon}/"
        result_dir="${RESULT_BASE}/SPEC_TTA/${BACKBONE}/${dataset}_${horizon}/"
        mkdir -p "$result_dir"
        
        python main.py \
            DATA.NAME "$dataset" \
            DATA.PRED_LEN "$horizon" \
            MODEL.NAME "$BACKBONE" \
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
        
        spec_mse=$(grep "Final MSE:" "${result_dir}/run.log" | tail -1 | awk '{print $3}')
        spec_mae=$(grep "Final MAE:" "${result_dir}/run.log" | tail -1 | awk '{print $3}')
        spec_updates=$(grep "Total Adaptation Updates:" "${result_dir}/run.log" | tail -1 | awk '{print $4}')
        spec_params=$(grep "Total Trainable Parameters:" "${result_dir}/run.log" | tail -1 | awk '{print $4}')
        
        echo "SPEC-TTA,$dataset,$BACKBONE,$horizon,$spec_mse,$spec_mae,$spec_updates,$spec_params" >> "${RESULT_BASE}/results_summary.csv"
        echo "✅ SPEC-TTA: MSE=$spec_mse, MAE=$spec_mae, Updates=$spec_updates, Params=$spec_params" | tee -a "$LOG_FILE"
        
        # PETSA
        echo "" | tee -a "$LOG_FILE"
        echo "🔬 Running PETSA..." | tee -a "$LOG_FILE"
        
        result_dir="${RESULT_BASE}/PETSA/${BACKBONE}/${dataset}_${horizon}/"
        mkdir -p "$result_dir"
        
        python main.py \
            DATA.NAME "$dataset" \
            DATA.PRED_LEN "$horizon" \
            MODEL.NAME "$BACKBONE" \
            MODEL.pred_len "$horizon" \
            TRAIN.ENABLE False \
            TEST.ENABLE False \
            TTA.ENABLE True \
            TRAIN.CHECKPOINT_DIR "$checkpoint_dir" \
            RESULT_DIR "$result_dir" \
            2>&1 | tee "${result_dir}/run.log"
        
        petsa_mse=$(grep "Final MSE:" "${result_dir}/run.log" | tail -1 | awk '{print $3}')
        petsa_mae=$(grep "Final MAE:" "${result_dir}/run.log" | tail -1 | awk '{print $3}')
        petsa_updates=$(grep "Total Adaptation Updates:" "${result_dir}/run.log" | tail -1 | awk '{print $4}')
        petsa_params=$(grep "Total Trainable Parameters:" "${result_dir}/run.log" | tail -1 | awk '{print $4}')
        
        echo "PETSA,$dataset,$BACKBONE,$horizon,$petsa_mse,$petsa_mae,$petsa_updates,$petsa_params" >> "${RESULT_BASE}/results_summary.csv"
        echo "✅ PETSA: MSE=$petsa_mse, MAE=$petsa_mae, Updates=$petsa_updates, Params=$petsa_params" | tee -a "$LOG_FILE"
        
        # Calculate improvement
        if command -v bc &> /dev/null && [ -n "$spec_mse" ] && [ -n "$petsa_mse" ]; then
            improvement=$(echo "scale=2; (($petsa_mse - $spec_mse) / $petsa_mse) * 100" | bc)
            echo "📊 Improvement: ${improvement}% (SPEC-TTA vs PETSA)" | tee -a "$LOG_FILE"
        fi
        
        echo "" | tee -a "$LOG_FILE"
    done
done

# Generate summary
echo "" | tee -a "$LOG_FILE"
echo "╔════════════════════════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║                            SUMMARY TABLE                                   ║" | tee -a "$LOG_FILE"
echo "╚════════════════════════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 << 'PYTHON_EOF' | tee -a "$LOG_FILE"
import pandas as pd
import sys

try:
    df = pd.read_csv('./results/QUICK_COMPARISON/results_summary.csv')
    df['MSE'] = pd.to_numeric(df['MSE'], errors='coerce')
    df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')
    df = df.dropna(subset=['MSE', 'MAE'])
    
    print("\n" + "="*100)
    print("QUICK COMPARISON RESULTS: SPEC-TTA vs PETSA")
    print("="*100)
    
    for dataset in df['Dataset'].unique():
        print(f"\n{'─'*100}")
        print(f"Dataset: {dataset}")
        print(f"{'─'*100}")
        print(f"{'Horizon':<10} {'Method':<12} {'MSE':<12} {'MAE':<12} {'Updates':<10} {'Params':<10}")
        print(f"{'─'*100}")
        
        for horizon in sorted(df['Horizon'].unique()):
            data = df[(df['Dataset'] == dataset) & (df['Horizon'] == horizon)]
            
            for _, row in data.iterrows():
                print(f"{row['Horizon']:<10} {row['Method']:<12} {row['MSE']:<12.6f} {row['MAE']:<12.6f} "
                      f"{int(row['Updates']) if pd.notna(row['Updates']) else 'N/A':<10} "
                      f"{int(row['Params']) if pd.notna(row['Params']) else 'N/A':<10}")
            
            # Calculate improvement
            spec = data[data['Method'] == 'SPEC-TTA']
            petsa = data[data['Method'] == 'PETSA']
            
            if not spec.empty and not petsa.empty:
                spec_mse = spec['MSE'].values[0]
                petsa_mse = petsa['MSE'].values[0]
                improvement = ((petsa_mse - spec_mse) / petsa_mse) * 100
                
                winner = "SPEC-TTA ✅" if spec_mse < petsa_mse else "PETSA ⚠️"
                print(f"{'':10} {'→ Winner:':<12} {winner:<12} Improvement: {improvement:+.2f}%")
            
            print()
    
    print("="*100)
    
    # Overall summary
    spec_tta_df = df[df['Method'] == 'SPEC-TTA']
    petsa_df = df[df['Method'] == 'PETSA']
    
    print("\n📊 OVERALL STATISTICS:")
    print(f"{'─'*100}")
    print(f"{'Method':<15} {'Avg MSE':<15} {'Avg MAE':<15} {'Avg Params':<15}")
    print(f"{'─'*100}")
    print(f"{'SPEC-TTA':<15} {spec_tta_df['MSE'].mean():<15.6f} {spec_tta_df['MAE'].mean():<15.6f} "
          f"{spec_tta_df['Params'].mean():<15.0f}")
    print(f"{'PETSA':<15} {petsa_df['MSE'].mean():<15.6f} {petsa_df['MAE'].mean():<15.6f} "
          f"{petsa_df['Params'].mean():<15.0f}")
    
    avg_improvement = ((petsa_df['MSE'].mean() - spec_tta_df['MSE'].mean()) / petsa_df['MSE'].mean()) * 100
    print(f"{'─'*100}")
    print(f"Average Improvement: {avg_improvement:+.2f}%")
    print(f"{'─'*100}")
    
    print("\n✅ Results saved to: ./results/QUICK_COMPARISON/results_summary.csv")
    print("="*100 + "\n")
    
except Exception as e:
    print(f"Error generating summary: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

echo "" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
