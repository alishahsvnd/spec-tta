#!/usr/bin/env python3
"""
visualize_spec_tta_results.py
Analyze and visualize SPEC-TTA performance across different configurations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

def load_results(result_dir: str, model: str = "iTransformer", dataset: str = "ETTh1_96") -> Dict:
    """Load results from a specific experiment directory."""
    base_path = Path(result_dir) / model / dataset
    
    results = {}
    
    # Load test.txt
    test_file = base_path / "test.txt"
    if test_file.exists():
        with open(test_file, 'r') as f:
            content = f.read().strip()
            parts = content.split(',')
            for part in parts:
                key, val = part.split(':')
                results[key.strip()] = float(val.strip())
    
    # Load time series results
    try:
        results['test_mse_all'] = np.load(base_path / "test_mse_all.npy")
    except:
        pass
    
    return results


def compare_variants(results_base: str = "results") -> None:
    """Compare different SPEC-TTA variants."""
    variants = [
        ("Baseline\n(thresh=0.01)", "SPEC_TTA_KBINS_16_BETA_0.05_DRIFT_0.01"),
        ("Low Threshold\n(thresh=0.001)", "SPEC_TTA_KBINS_16_BETA_0.05_DRIFT_0.001"),
        ("Low Thresh\n+ High LR", "SPEC_TTA_LOWTHRESH_HIGHLR"),
        ("More Bins\n(K=32)", "SPEC_TTA_KBINS_32_BETA_0.05_DRIFT_0.001"),
        ("Optimized", "SPEC_TTA_OPTIMIZED")
    ]
    
    # Collect results
    all_results = {}
    for name, dirname in variants:
        path = os.path.join(results_base, dirname)
        if os.path.exists(path):
            all_results[name] = load_results(path)
    
    if not all_results:
        print("No results found. Please run experiments first.")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SPEC-TTA Variant Comparison', fontsize=16, fontweight='bold')
    
    # 1. Bar chart: Test MSE comparison
    ax = axes[0, 0]
    names = list(all_results.keys())
    mse_values = [all_results[name].get('test_mse', 0) for name in names]
    mae_values = [all_results[name].get('test_mae', 0) for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mse_values, width, label='Test MSE', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, mae_values, width, label='Test MAE', alpha=0.8, color='coral')
    
    ax.set_xlabel('Variant', fontweight='bold')
    ax.set_ylabel('Error', fontweight='bold')
    ax.set_title('Test Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Improvement over baseline
    ax = axes[0, 1]
    baseline_mse = mse_values[0] if mse_values else 1.0
    improvements = [(baseline_mse - mse) / baseline_mse * 100 for mse in mse_values]
    
    colors = ['red' if imp < 0 else 'green' for imp in improvements]
    bars = ax.barh(names, improvements, color=colors, alpha=0.7)
    
    ax.set_xlabel('Improvement over Baseline (%)', fontweight='bold')
    ax.set_title('Relative Improvement in Test MSE')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        ax.text(imp + 0.5 if imp > 0 else imp - 0.5, i,
               f'{imp:+.2f}%', va='center', fontweight='bold',
               fontsize=9)
    
    # 3. Train vs Test comparison
    ax = axes[1, 0]
    train_mse = [all_results[name].get('train_mse', 0) for name in names]
    test_mse = [all_results[name].get('test_mse', 0) for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, train_mse, width, label='Train MSE', alpha=0.8, color='lightblue')
    ax.bar(x + width/2, test_mse, width, label='Test MSE', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Variant', fontweight='bold')
    ax.set_ylabel('MSE', fontweight='bold')
    ax.set_title('Train vs Test Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Time series plot (if available)
    ax = axes[1, 1]
    has_timeseries = False
    
    for name, color in zip(names[:3], ['blue', 'green', 'red']):  # Plot first 3 variants
        if 'test_mse_all' in all_results[name]:
            mse_series = all_results[name]['test_mse_all']
            ax.plot(mse_series, label=name, alpha=0.7, linewidth=2, color=color)
            has_timeseries = True
    
    if has_timeseries:
        ax.set_xlabel('Test Sample Index', fontweight='bold')
        ax.set_ylabel('MSE', fontweight='bold')
        ax.set_title('MSE Over Test Samples')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Time series data not available',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(results_base, 'spec_tta_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def print_summary_table(results_base: str = "results") -> None:
    """Print a formatted summary table."""
    variants = [
        ("Baseline (thresh=0.01)", "SPEC_TTA_KBINS_16_BETA_0.05_DRIFT_0.01"),
        ("Lower threshold (0.001)", "SPEC_TTA_KBINS_16_BETA_0.05_DRIFT_0.001"),
        ("Low thresh + high LR", "SPEC_TTA_LOWTHRESH_HIGHLR"),
        ("More bins (K=32)", "SPEC_TTA_KBINS_32_BETA_0.05_DRIFT_0.001"),
        ("Optimized", "SPEC_TTA_OPTIMIZED")
    ]
    
    print("\n" + "="*90)
    print("SPEC-TTA Performance Summary")
    print("="*90)
    print(f"{'Variant':<30} {'Train MSE':>12} {'Test MSE':>12} {'Test MAE':>12} {'Improvement':>12}")
    print("-"*90)
    
    baseline_mse = None
    for name, dirname in variants:
        path = os.path.join(results_base, dirname)
        if os.path.exists(path):
            results = load_results(path)
            
            train_mse = results.get('train_mse', 0)
            test_mse = results.get('test_mse', 0)
            test_mae = results.get('test_mae', 0)
            
            if baseline_mse is None:
                baseline_mse = test_mse
                improvement = "baseline"
            else:
                if baseline_mse > 0:
                    improvement = f"{(baseline_mse - test_mse) / baseline_mse * 100:+.2f}%"
                else:
                    improvement = "N/A"
            
            print(f"{name:<30} {train_mse:>12.4f} {test_mse:>12.4f} {test_mae:>12.4f} {improvement:>12}")
        else:
            print(f"{name:<30} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    
    print("="*90)
    print()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SPEC-TTA results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Base directory containing results')
    parser.add_argument('--table_only', action='store_true',
                       help='Only print summary table (no plots)')
    
    args = parser.parse_args()
    
    # Print summary table
    print_summary_table(args.results_dir)
    
    # Generate plots (unless table_only)
    if not args.table_only:
        try:
            compare_variants(args.results_dir)
        except Exception as e:
            print(f"\nWarning: Could not generate plots: {e}")
            print("You may need to install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
