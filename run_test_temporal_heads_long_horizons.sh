#!/bin/bash
# Test SPEC-TTA with TimeShiftHead + PolyTrendHead on LONG horizons
# Check if temporal heads fix the catastrophic failures (H=336, H=720)

set -e

DATASET="ETTh2"
MODEL="iTransformer"
OUTPUT_DIR="results/SPEC_TTA_TEMPORAL_HEADS_LONG_HORIZONS"
LOG_FILE="spec_tta_temporal_long_horizons.log"

echo "=== Testing Temporal Heads on Long Horizons ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET, Model: $MODEL" | tee -a "$LOG_FILE"
echo "Testing: H=336 and H=720 (previously catastrophic)" | tee -a "$LOG_FILE"
echo "Setup: TimeShiftHead + PolyTrendHead + full-horizon mask (reverted)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

mkdir -p "$OUTPUT_DIR/$MODEL"

for HORIZON in 336 720; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running: $MODEL on $DATASET H=$HORIZON" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
    
    if [ ! -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        echo "ERROR: Checkpoint not found at $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
        continue
    fi
    
    python -u main.py \
        DATA.NAME "$DATASET" \
        DATA.PRED_LEN "$HORIZON" \
        MODEL.NAME "$MODEL" \
        MODEL.pred_len "$HORIZON" \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR "$CHECKPOINT_DIR" \
        TTA.ENABLE True \
        TTA.SPEC_TTA.K_BINS 32 \
        TTA.SPEC_TTA.PATCH_LEN 24 \
        TTA.SPEC_TTA.HUBER_DELTA 0.5 \
        TTA.SPEC_TTA.BETA_FREQ 0.05 \
        TTA.SPEC_TTA.LAMBDA_PW 1.0 \
        TTA.SPEC_TTA.LAMBDA_PROX 0.0001 \
        TTA.SPEC_TTA.LAMBDA_HC 0.1 \
        TTA.SPEC_TTA.DRIFT_THRESHOLD 0.01 \
        TTA.SPEC_TTA.LR 0.001 \
        TTA.SPEC_TTA.GRAD_CLIP 1.0 \
        RESULT_DIR "$OUTPUT_DIR/$MODEL/" \
        2>&1 | tee "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt"
    
    echo "" | tee -a "$LOG_FILE"
    echo "Results:" | tee -a "$LOG_FILE"
    grep -E "(Final MSE|Final MAE|Total Adaptation)" "$OUTPUT_DIR/$MODEL/${DATASET}_${HORIZON}.txt" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "=== Complete Comparison Analysis ===" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

python3 << 'PYEOF' | tee -a "$LOG_FILE"
import re

def read_metrics(filepath):
    try:
        with open(filepath) as f:
            content = f.read()
            mse = re.search(r'(?:Final MSE|test_mse): ([\d.]+)', content)
            mae = re.search(r'(?:Final MAE|test_mae): ([\d.]+)', content)
            updates = re.search(r'Total Adaptation Updates: (\d+)', content)
            return {
                'mse': float(mse.group(1)) if mse else None,
                'mae': float(mae.group(1)) if mae else None,
                'updates': int(updates.group(1)) if updates else 0
            }
    except:
        return {'mse': None, 'mae': None, 'updates': 0}

results = {}
for h in [336, 720]:
    results[h] = {}
    
    # Baseline
    baseline = read_metrics(f"results/SPEC_TTA_BENCHMARK_ETTh2/iTransformer/iTransformer/ETTh2_{h}/test.txt")
    results[h]['baseline'] = baseline
    
    # Previous SPEC-TTA (linear trend, catastrophic on long horizons)
    import glob
    files = glob.glob(f"results/SPEC_TTA_BENCHMARK_ETTh2/*{h}*.txt")
    for f in files:
        if 'iTransformer' in open(f).read():
            results[h]['old_spec'] = read_metrics(f)
            break
    
    # New SPEC-TTA (temporal heads)
    results[h]['new_spec'] = read_metrics(f"results/SPEC_TTA_TEMPORAL_HEADS_LONG_HORIZONS/iTransformer/ETTh2_{h}.txt")

print("\n" + "="*80)
print("LONG HORIZON COMPARISON: TimeShiftHead + PolyTrendHead vs Linear TrendHead")
print("="*80)

for h in [336, 720]:
    r = results[h]
    print(f"\n{'='*80}")
    print(f"H={h}")
    print(f"{'='*80}")
    
    base = r['baseline']
    old = r.get('old_spec', {})
    new = r['new_spec']
    
    if base['mse'] and old.get('mse') and new['mse']:
        print(f"\n{'Method':<35} {'MSE':>12} {'MAE':>12} {'Updates':>8} {'vs Baseline':>15}")
        print(f"{'-'*80}")
        print(f"{'Baseline (no TTA)':<35} {base['mse']:>12.4f} {base['mae']:>12.4f} {'-':>8} {'-':>15}")
        
        old_change = ((base['mse'] - old['mse']) / base['mse'] * 100)
        old_status = "✓" if old_change > 0 else "✗"
        print(f"{'Old SPEC-TTA (linear trend)':<35} {old['mse']:>12.4f} {old['mae']:>12.4f} {old['updates']:>8} {old_change:>+14.1f}% {old_status}")
        
        new_change = ((base['mse'] - new['mse']) / base['mse'] * 100)
        new_status = "✓" if new_change > 0 else "✗"
        print(f"{'NEW SPEC-TTA (temporal heads)':<35} {new['mse']:>12.4f} {new['mae']:>12.4f} {new['updates']:>8} {new_change:>+14.1f}% {new_status}")
        
        print(f"\n{'Improvement Analysis:'}")
        if old['mse'] > base['mse'] and new['mse'] < old['mse']:
            improvement = ((old['mse'] - new['mse']) / old['mse'] * 100)
            print(f"  • Temporal heads BETTER than old linear trend: {improvement:+.1f}%")
            if new['mse'] < base['mse']:
                print(f"  • ✓ FIXES catastrophic failure! Now improves over baseline.")
            else:
                print(f"  • ⚠ Partially fixes: still worse than baseline but much better than old.")
        elif old['mse'] > base['mse'] and new['mse'] > old['mse']:
            print(f"  • ✗ Temporal heads WORSE than old linear trend")
            print(f"  • Both fail catastrophically on long horizons")
        elif new_change > old_change:
            print(f"  • ✓ Temporal heads provide additional improvement")
        else:
            print(f"  • ⚠ Temporal heads slightly regress vs old linear trend")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}\n")

# Overall assessment
all_new_better = True
any_new_fixes = False
for h in [336, 720]:
    r = results[h]
    base = r['baseline']
    old = r.get('old_spec', {})
    new = r['new_spec']
    if base['mse'] and old.get('mse') and new['mse']:
        if new['mse'] < old['mse']:
            any_new_fixes = True
        if new['mse'] > old['mse']:
            all_new_better = False

if any_new_fixes:
    print("✓ Temporal heads (TimeShiftHead + PolyTrendHead) show improvement over old linear trend")
    print("  on long horizons!")
else:
    print("✗ Temporal heads do not fix long-horizon catastrophic failures.")
    print("  Alternative approach needed (e.g., lower drift threshold, early stopping).")

PYEOF

echo "" | tee -a "$LOG_FILE"
echo "Complete! Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
