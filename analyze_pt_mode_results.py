#!/usr/bin/env python
"""
Analyze SPEC-TTA behavior with true PT mode by examining adaptation metrics.
"""
import re
import sys

def analyze_log(logfile):
    print(f"\n=== Analyzing {logfile} ===\n")
    
    try:
        with open(logfile, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {logfile}")
        return
    
    # Extract key metrics
    mse_match = re.search(r'Final MSE: ([\d.]+)', content)
    mae_match = re.search(r'Final MAE: ([\d.]+)', content)
    updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
    baseline_match = re.search(r'test_mse: ([\d.]+), test_mae: ([\d.]+)', content)
    
    if mse_match and mae_match and updates_match:
        final_mse = float(mse_match.group(1))
        final_mae = float(mae_match.group(1))
        updates = int(updates_match.group(1))
        
        print(f"Final Results:")
        print(f"  MSE: {final_mse:.4f}")
        print(f"  MAE: {final_mae:.4f}")
        print(f"  Total Updates: {updates}")
        
        if baseline_match:
            baseline_mse = float(baseline_match.group(1))
            baseline_mae = float(baseline_match.group(2))
            print(f"\nBaseline (no TTA):")
            print(f"  MSE: {baseline_mse:.4f}")
            print(f"  MAE: {baseline_mae:.4f}")
            print(f"\nDegradation:")
            print(f"  MSE: {final_mse/baseline_mse:.1f}× worse")
            print(f"  MAE: {final_mae/baseline_mae:.1f}× worse")
            
            if final_mse > 10 * baseline_mse:
                print(f"\n⚠️  CATASTROPHIC FAILURE: MSE exploded by {final_mse/baseline_mse:.0f}×")
                print("Likely causes:")
                print("  - Overfitting to sparse PT observations (10% prefix)")
                print("  - Learning rate too high")
                print("  - Proximal regularization too weak")
                print("  - Loss imbalance between different components")
        
        # Check if adaptation actually occurred
        if updates == 0:
            print("\n⚠️  NO ADAPTATION: Drift threshold never exceeded")
        elif updates < 5:
            print(f"\n⚠️  MINIMAL ADAPTATION: Only {updates} updates")
        else:
            print(f"\n✓ Adaptation active: {updates} updates performed")

if __name__ == "__main__":
    # Analyze both horizons
    for h in [336, 720]:
        analyze_log(f"results/SPEC_TTA_TRUE_PT_MODE_ETTh2/iTransformer/ETTh2_{h}.txt")
        print("\n" + "="*60)
