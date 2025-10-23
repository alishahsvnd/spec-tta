#!/usr/bin/env python3
"""
Analyze all SPEC-TTA benchmark results and generate paper-ready tables.
Extracts MSE/MAE, parameters, updates from all experiment logs.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Result directory
RESULT_DIR = Path("./results/SPEC_TTA_FULL_BENCHMARK")

# Datasets and their variable counts
DATASETS = {
    "ETTh1": 7,
    "ETTh2": 7,
    "Weather": 21,
    "Exchange": 8,
    "Electricity": 321,
}

# Models to analyze
MODELS = ["iTransformer", "DLinear", "PatchTST"]

# Horizons
HORIZONS = [96, 192, 336, 720]


def extract_metrics(log_file: Path) -> Dict:
    """Extract MSE, MAE, params, updates from experiment log."""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract baseline MSE/MAE
    baseline_mse = re.search(r"Baseline.*MSE[:\s]+([0-9.]+)", content, re.IGNORECASE)
    baseline_mae = re.search(r"Baseline.*MAE[:\s]+([0-9.]+)", content, re.IGNORECASE)
    
    # Extract SPEC-TTA MSE/MAE
    spectta_mse = re.search(r"(?:SPEC-TTA|Final|Test).*MSE[:\s]+([0-9.]+)", content, re.IGNORECASE)
    spectta_mae = re.search(r"(?:SPEC-TTA|Final|Test).*MAE[:\s]+([0-9.]+)", content, re.IGNORECASE)
    
    # Extract parameters
    params = re.search(r"(?:Trainable parameters|Total params)[:\s]+([0-9,]+)", content, re.IGNORECASE)
    
    # Extract update count
    updates = re.search(r"(?:Updates|Adaptations|Drift triggers)[:\s]+([0-9]+)", content, re.IGNORECASE)
    
    if baseline_mse:
        metrics['baseline_mse'] = float(baseline_mse.group(1))
    if baseline_mae:
        metrics['baseline_mae'] = float(baseline_mae.group(1))
    if spectta_mse:
        metrics['spectta_mse'] = float(spectta_mse.group(1))
    if spectta_mae:
        metrics['spectta_mae'] = float(spectta_mae.group(1))
    if params:
        metrics['params'] = int(params.group(1).replace(',', ''))
    if updates:
        metrics['updates'] = int(updates.group(1))
    
    # Compute improvement
    if 'baseline_mse' in metrics and 'spectta_mse' in metrics:
        baseline = metrics['baseline_mse']
        spectta = metrics['spectta_mse']
        if baseline > 0:
            metrics['improvement'] = (baseline - spectta) / baseline * 100
    
    return metrics if metrics else None


def generate_summary_table():
    """Generate summary table for all datasets and horizons."""
    print("\n" + "="*80)
    print("SPEC-TTA Comprehensive Benchmark Results")
    print("="*80 + "\n")
    
    results = {}
    
    for dataset in DATASETS.keys():
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for horizon in HORIZONS:
                log_file = RESULT_DIR / dataset / f"SPEC_TTA_{model}_{dataset}_{horizon}.txt"
                metrics = extract_metrics(log_file)
                results[dataset][model][horizon] = metrics
    
    return results


def print_latex_table(results: Dict):
    """Generate LaTeX table for paper."""
    print("\n" + "="*80)
    print("LaTeX Table (paste into paper)")
    print("="*80 + "\n")
    
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{SPEC-TTA performance across all benchmarks. Best in \textbf{bold}.}")
    print(r"\label{tab:full_results}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Model} & \textbf{H=96} & \textbf{H=192} & \textbf{H=336} & \textbf{H=720} & \textbf{Avg Improv} & \textbf{Params} \\")
    print(r"\midrule")
    
    for dataset in DATASETS.keys():
        print(f"\\multirow{{3}}{{*}}{{{dataset}}}")
        
        for i, model in enumerate(MODELS):
            improvements = []
            params_val = None
            
            for horizon in HORIZONS:
                metrics = results.get(dataset, {}).get(model, {}).get(horizon)
                if metrics and 'spectta_mse' in metrics:
                    mse = metrics['spectta_mse']
                    improvement = metrics.get('improvement', 0)
                    improvements.append(improvement)
                    params_val = metrics.get('params')
                    print(f" & {model if i == 0 else ''} & {mse:.4f} ", end='')
                else:
                    print(f" & {model if i == 0 else ''} & --- ", end='')
            
            avg_improv = sum(improvements) / len(improvements) if improvements else 0
            print(f"& {avg_improv:+.1f}\\% & {params_val if params_val else '---'} \\\\")
        
        print(r"\midrule")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")
    print()


def print_summary_statistics(results: Dict):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80 + "\n")
    
    all_improvements = []
    all_params = []
    dataset_stats = {}
    
    for dataset in DATASETS.keys():
        dataset_improvements = []
        
        for model in MODELS:
            for horizon in HORIZONS:
                metrics = results.get(dataset, {}).get(model, {}).get(horizon)
                if metrics and 'improvement' in metrics:
                    improvement = metrics['improvement']
                    all_improvements.append(improvement)
                    dataset_improvements.append(improvement)
                if metrics and 'params' in metrics:
                    all_params.append(metrics['params'])
        
        if dataset_improvements:
            dataset_stats[dataset] = {
                'avg': sum(dataset_improvements) / len(dataset_improvements),
                'min': min(dataset_improvements),
                'max': max(dataset_improvements)
            }
    
    # Overall statistics
    if all_improvements:
        print(f"Overall Average Improvement: {sum(all_improvements)/len(all_improvements):.1f}%")
        print(f"Best Improvement: {max(all_improvements):.1f}%")
        print(f"Worst Improvement: {min(all_improvements):.1f}%")
        print()
    
    # Per-dataset statistics
    print("Per-Dataset Statistics:")
    for dataset, stats in dataset_stats.items():
        print(f"  {dataset:12s}: Avg={stats['avg']:+.1f}%, Min={stats['min']:+.1f}%, Max={stats['max']:+.1f}%")
    print()
    
    # Parameter statistics
    if all_params:
        print(f"Parameter Range: {min(all_params)} to {max(all_params)}")
        print(f"Average Parameters: {sum(all_params)/len(all_params):.0f}")
        print()


def print_best_results(results: Dict):
    """Print best result for each dataset."""
    print("\n" + "="*80)
    print("Best Results Per Dataset (for paper highlights)")
    print("="*80 + "\n")
    
    for dataset in DATASETS.keys():
        best_mse = float('inf')
        best_config = None
        
        for model in MODELS:
            for horizon in HORIZONS:
                metrics = results.get(dataset, {}).get(model, {}).get(horizon)
                if metrics and 'spectta_mse' in metrics:
                    if metrics['spectta_mse'] < best_mse:
                        best_mse = metrics['spectta_mse']
                        best_config = (model, horizon, metrics)
        
        if best_config:
            model, horizon, metrics = best_config
            improvement = metrics.get('improvement', 0)
            print(f"{dataset:12s}: {model:15s} H={horizon:3d} → MSE={best_mse:.4f} ({improvement:+.1f}%)")


def save_json_results(results: Dict):
    """Save results to JSON for further analysis."""
    output_file = RESULT_DIR / "all_results.json"
    
    # Convert Path objects to strings for JSON serialization
    json_results = {}
    for dataset, models in results.items():
        json_results[dataset] = {}
        for model, horizons in models.items():
            json_results[dataset][model] = {}
            for horizon, metrics in horizons.items():
                if metrics:
                    json_results[dataset][model][str(horizon)] = metrics
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main analysis pipeline."""
    print("Analyzing SPEC-TTA benchmark results...")
    print(f"Result directory: {RESULT_DIR}")
    
    # Extract all results
    results = generate_summary_table()
    
    # Generate outputs
    print_summary_statistics(results)
    print_best_results(results)
    print_latex_table(results)
    
    # Save to JSON
    save_json_results(results)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
