#!/usr/bin/env python3
"""
Analyze SPEC-TTA full backbone comparison results.
Extracts MSE/MAE, baseline performance, and improvement metrics.
"""

import re
import os
from pathlib import Path

# Result files
RESULT_DIR = Path("./results/SPEC_TTA_FULL_COMPARISON")
MODELS = ["iTransformer", "DLinear", "FreTS", "MICN", "PatchTST"]

def extract_metrics(filepath):
    """Extract key metrics from SPEC-TTA output file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract SPEC-TTA results
    mse_match = re.search(r'Final MSE: ([\d.]+)', content)
    mae_match = re.search(r'Final MAE: ([\d.]+)', content)
    updates_match = re.search(r'Total Adaptation Updates: (\d+)', content)
    params_match = re.search(r'Total Trainable Parameters:\s+(\d+)', content)
    
    # Extract baseline (no TTA) results
    baseline_mse_match = re.search(r'Results without TSF-TTA:.*?test_mse: ([\d.]+)', content, re.DOTALL)
    baseline_mae_match = re.search(r'Results without TSF-TTA:.*?test_mae: ([\d.]+)', content, re.DOTALL)
    
    if mse_match:
        metrics['spec_tta_mse'] = float(mse_match.group(1))
    if mae_match:
        metrics['spec_tta_mae'] = float(mae_match.group(1))
    if updates_match:
        metrics['num_updates'] = int(updates_match.group(1))
    if params_match:
        metrics['num_params'] = int(params_match.group(1))
    if baseline_mse_match:
        metrics['baseline_mse'] = float(baseline_mse_match.group(1))
    if baseline_mae_match:
        metrics['baseline_mae'] = float(baseline_mae_match.group(1))
    
    # Calculate improvement
    if 'spec_tta_mse' in metrics and 'baseline_mse' in metrics:
        improvement = ((metrics['baseline_mse'] - metrics['spec_tta_mse']) / metrics['baseline_mse']) * 100
        metrics['improvement_pct'] = improvement
    
    return metrics

def main():
    print("=" * 80)
    print("SPEC-TTA Full Backbone Comparison Results")
    print("Dataset: ETTh1, Horizon: 96")
    print("Configuration: K_BINS=32, DRIFT_THRESHOLD=0.005")
    print("=" * 80)
    print()
    
    results = {}
    for model in MODELS:
        filepath = RESULT_DIR / f"SPEC_TTA_{model}_ETTh1_96.txt"
        if filepath.exists():
            results[model] = extract_metrics(filepath)
        else:
            print(f"⚠️  Warning: Results for {model} not found at {filepath}")
    
    # Table 1: Main Results
    print("Table 1: SPEC-TTA Performance Comparison")
    print("-" * 120)
    print(f"{'Model':<15} {'Baseline MSE':<15} {'SPEC-TTA MSE':<15} {'Baseline MAE':<15} {'SPEC-TTA MAE':<15} {'Improvement':<12} {'Updates':<10}")
    print("-" * 120)
    
    for model in MODELS:
        if model in results:
            r = results[model]
            baseline_mse = r.get('baseline_mse', 0)
            spec_mse = r.get('spec_tta_mse', 0)
            baseline_mae = r.get('baseline_mae', 0)
            spec_mae = r.get('spec_tta_mae', 0)
            improvement = r.get('improvement_pct', 0)
            updates = r.get('num_updates', 0)
            
            print(f"{model:<15} {baseline_mse:<15.4f} {spec_mse:<15.4f} {baseline_mae:<15.4f} {spec_mae:<15.4f} {improvement:>10.1f}% {updates:>9}")
    
    print("-" * 120)
    print()
    
    # Table 2: Parameter Efficiency
    print("Table 2: Parameter Efficiency")
    print("-" * 80)
    print(f"{'Model':<15} {'Trainable Params':<20} {'MSE':<15} {'Params/MSE Ratio':<20}")
    print("-" * 80)
    
    for model in MODELS:
        if model in results:
            r = results[model]
            params = r.get('num_params', 0)
            mse = r.get('spec_tta_mse', 1)
            ratio = params / mse if mse > 0 else 0
            
            print(f"{model:<15} {params:<20} {mse:<15.4f} {ratio:<20.1f}")
    
    print("-" * 80)
    print()
    
    # Best results
    print("Summary Statistics:")
    print("-" * 80)
    
    if results:
        best_mse_model = min(results.items(), key=lambda x: x[1].get('spec_tta_mse', float('inf')))
        best_improvement_model = max(results.items(), key=lambda x: x[1].get('improvement_pct', 0))
        
        print(f"✓ Best MSE: {best_mse_model[0]} ({best_mse_model[1]['spec_tta_mse']:.4f})")
        print(f"✓ Best Improvement: {best_improvement_model[0]} ({best_improvement_model[1]['improvement_pct']:.1f}%)")
        
        avg_improvement = sum(r.get('improvement_pct', 0) for r in results.values()) / len(results)
        print(f"✓ Average Improvement: {avg_improvement:.1f}%")
        
        avg_params = sum(r.get('num_params', 0) for r in results.values()) / len(results)
        print(f"✓ Average Parameters: {avg_params:.0f}")
        
        avg_updates = sum(r.get('num_updates', 0) for r in results.values()) / len(results)
        print(f"✓ Average Updates: {avg_updates:.1f}")
    
    print("-" * 80)
    print()
    
    # LaTeX table format
    print("LaTeX Table Format:")
    print("-" * 80)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{SPEC-TTA Performance on ETTh1 (Horizon=96)}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Model & Baseline MSE & SPEC-TTA MSE & Baseline MAE & SPEC-TTA MAE & Improvement \\\\")
    print("\\midrule")
    
    for model in MODELS:
        if model in results:
            r = results[model]
            baseline_mse = r.get('baseline_mse', 0)
            spec_mse = r.get('spec_tta_mse', 0)
            baseline_mae = r.get('baseline_mae', 0)
            spec_mae = r.get('spec_tta_mae', 0)
            improvement = r.get('improvement_pct', 0)
            
            # Bold the best MSE
            if model == best_mse_model[0]:
                spec_mse_str = f"\\textbf{{{spec_mse:.4f}}}"
            else:
                spec_mse_str = f"{spec_mse:.4f}"
            
            print(f"{model} & {baseline_mse:.4f} & {spec_mse_str} & {baseline_mae:.4f} & {spec_mae:.4f} & {improvement:+.1f}\\% \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

if __name__ == "__main__":
    main()
