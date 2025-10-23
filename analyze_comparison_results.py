#!/usr/bin/env python3
"""
Analyze and aggregate comparison results from PETSA vs SPEC-TTA HC experiments.
Generates tables and visualizations suitable for publication.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_result_file(filepath):
    """
    Parse a result text file to extract MSE, MAE, and parameter count.
    
    Returns:
        dict with keys: 'mse', 'mae', 'params', 'n_adapt'
    """
    results = {
        'mse': None,
        'mae': None,
        'params': None,
        'n_adapt': None
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract MSE
            mse_match = re.search(r'Test MSE:\s*([0-9.]+)', content)
            if mse_match:
                results['mse'] = float(mse_match.group(1))
            
            # Extract MAE
            mae_match = re.search(r'Test MAE:\s*([0-9.]+)', content)
            if mae_match:
                results['mae'] = float(mae_match.group(1))
            
            # Extract parameter count
            param_match = re.search(r'Total(?:\s+Trainable)?\s+Parameters?:\s*([0-9,]+)', content)
            if param_match:
                results['params'] = int(param_match.group(1).replace(',', ''))
            
            # Extract number of adaptations
            adapt_match = re.search(r'Number of adaptations:\s*([0-9]+)', content)
            if adapt_match:
                results['n_adapt'] = int(adapt_match.group(1))
                
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return results


def parse_filename(filename):
    """
    Parse filename to extract config type, config name, and model.
    
    Example: SPEC_TTA_HC_Medium_iTransformer_ETTh1_96.txt
    Returns: ('SPEC_TTA_HC', 'Medium', 'iTransformer')
    """
    parts = filename.replace('.txt', '').split('_')
    
    # Handle PETSA files
    if 'PETSA' in filename:
        config_type = 'PETSA'
        # Find RANK position
        rank_idx = [i for i, p in enumerate(parts) if 'RANK' in p][0]
        config_name = parts[rank_idx]
        model = parts[rank_idx + 1]
        return config_type, config_name, model
    
    # Handle SPEC_TTA_HC files
    if 'SPEC' in filename and 'TTA' in filename and 'HC' in filename:
        config_type = 'SPEC_TTA_HC'
        # Find config name (Medium/High/Ultra)
        config_idx = None
        for i, p in enumerate(parts):
            if p in ['Medium', 'High', 'Ultra']:
                config_idx = i
                break
        
        if config_idx is not None:
            config_name = parts[config_idx]
            model = parts[config_idx + 1]
            return config_type, config_name, model
    
    return None, None, None


def aggregate_results(results_dir):
    """
    Aggregate all results from a results directory.
    
    Returns:
        dict: {model: {config: {metric: value}}}
    """
    results_dir = Path(results_dir)
    data = defaultdict(lambda: defaultdict(dict))
    
    # Find all result files
    result_files = list(results_dir.glob('*.txt'))
    
    print(f"Found {len(result_files)} result files in {results_dir}")
    print()
    
    for filepath in result_files:
        config_type, config_name, model = parse_filename(filepath.name)
        
        if config_type is None:
            print(f"Skipping unrecognized file: {filepath.name}")
            continue
        
        results = parse_result_file(filepath)
        
        # Create unique key for configuration
        if config_type == 'PETSA':
            config_key = 'PETSA'
        else:
            config_key = f"SPEC-HC-{config_name}"
        
        data[model][config_key] = results
        
        mse_str = f"{results['mse']:.4f}" if results['mse'] is not None else 'N/A'
        print(f"Parsed: {model:15s} | {config_key:20s} | MSE={mse_str}")
    
    return dict(data)


def generate_table(data, metric='mse'):
    """
    Generate a comparison table for a specific metric.
    
    Args:
        data: dict from aggregate_results()
        metric: 'mse' or 'mae'
    """
    # Get all models and configs
    models = sorted(data.keys())
    
    # Determine all configs
    all_configs = set()
    for model in models:
        all_configs.update(data[model].keys())
    
    # Order configs: PETSA first, then Medium, High, Ultra
    config_order = ['PETSA', 'SPEC-HC-Medium', 'SPEC-HC-High', 'SPEC-HC-Ultra']
    configs = [c for c in config_order if c in all_configs]
    
    # Print header
    print(f"\n{'='*100}")
    print(f"{metric.upper()} Comparison Table")
    print(f"{'='*100}")
    print(f"{'Model':<15s}", end='')
    for config in configs:
        print(f" | {config:<20s}", end='')
    print()
    print('-' * 100)
    
    # Print rows
    for model in models:
        print(f"{model:<15s}", end='')
        for config in configs:
            if config in data[model] and data[model][config][metric] is not None:
                value = data[model][config][metric]
                print(f" | {value:<20.4f}", end='')
            else:
                print(f" | {'N/A':<20s}", end='')
        print()
    
    print('=' * 100)


def generate_improvement_table(data):
    """
    Generate a table showing improvement over PETSA baseline.
    """
    models = sorted(data.keys())
    configs = ['SPEC-HC-Medium', 'SPEC-HC-High', 'SPEC-HC-Ultra']
    
    print(f"\n{'='*100}")
    print("MSE Improvement over PETSA (%)")
    print(f"{'='*100}")
    print(f"{'Model':<15s}", end='')
    for config in configs:
        print(f" | {config:<20s}", end='')
    print()
    print('-' * 100)
    
    for model in models:
        if 'PETSA' not in data[model] or data[model]['PETSA']['mse'] is None:
            continue
        
        baseline_mse = data[model]['PETSA']['mse']
        print(f"{model:<15s}", end='')
        
        for config in configs:
            if config in data[model] and data[model][config]['mse'] is not None:
                new_mse = data[model][config]['mse']
                improvement = ((baseline_mse - new_mse) / baseline_mse) * 100
                sign = '+' if improvement > 0 else ''
                print(f" | {sign}{improvement:<19.2f}", end='')
            else:
                print(f" | {'N/A':<20s}", end='')
        print()
    
    print('=' * 100)


def generate_parameter_table(data):
    """
    Generate a table showing parameter counts.
    """
    models = sorted(data.keys())
    config_order = ['PETSA', 'SPEC-HC-Medium', 'SPEC-HC-High', 'SPEC-HC-Ultra']
    configs = [c for c in config_order if any(c in data[m] for m in models)]
    
    print(f"\n{'='*100}")
    print("Parameter Count Comparison")
    print(f"{'='*100}")
    print(f"{'Model':<15s}", end='')
    for config in configs:
        print(f" | {config:<20s}", end='')
    print()
    print('-' * 100)
    
    for model in models:
        print(f"{model:<15s}", end='')
        for config in configs:
            if config in data[model] and data[model][config]['params'] is not None:
                params = data[model][config]['params']
                print(f" | {params:<20,}", end='')
            else:
                print(f" | {'N/A':<20s}", end='')
        print()
    
    print('=' * 100)


def generate_summary_stats(data):
    """
    Generate summary statistics across all backbones.
    """
    configs = ['PETSA', 'SPEC-HC-Medium', 'SPEC-HC-High', 'SPEC-HC-Ultra']
    
    print(f"\n{'='*100}")
    print("Average Performance Across All Backbones")
    print(f"{'='*100}")
    
    for config in configs:
        mse_values = []
        mae_values = []
        param_values = []
        
        for model in data.keys():
            if config in data[model]:
                if data[model][config]['mse'] is not None:
                    mse_values.append(data[model][config]['mse'])
                if data[model][config]['mae'] is not None:
                    mae_values.append(data[model][config]['mae'])
                if data[model][config]['params'] is not None:
                    param_values.append(data[model][config]['params'])
        
        if mse_values:
            print(f"\n{config}:")
            print(f"  MSE: {np.mean(mse_values):.4f} ± {np.std(mse_values):.4f}")
            print(f"  MAE: {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
            if param_values:
                print(f"  Params: {int(np.mean(param_values)):,} ± {int(np.std(param_values)):,}")
            print(f"  Tested on {len(mse_values)} backbones")
    
    print('=' * 100)


def generate_publication_latex(data, output_file='comparison_table.tex'):
    """
    Generate LaTeX table suitable for publication.
    """
    models = sorted(data.keys())
    
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Comparison of PETSA and High-Capacity SPEC-TTA on ETTh1 (Pred Len=96)}")
    latex.append("\\label{tab:comparison}")
    latex.append("\\begin{tabular}{l|cccc|cccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{Backbone} & \\multicolumn{4}{c|}{MSE} & \\multicolumn{4}{c}{Parameters} \\\\")
    latex.append(" & PETSA & Medium & High & Ultra & PETSA & Medium & High & Ultra \\\\")
    latex.append("\\hline")
    
    for model in models:
        row = [model]
        
        # MSE values
        for config in ['PETSA', 'SPEC-HC-Medium', 'SPEC-HC-High', 'SPEC-HC-Ultra']:
            if config in data[model] and data[model][config]['mse'] is not None:
                mse = data[model][config]['mse']
                row.append(f"{mse:.4f}")
            else:
                row.append("--")
        
        # Parameter counts
        for config in ['PETSA', 'SPEC-HC-Medium', 'SPEC-HC-High', 'SPEC-HC-Ultra']:
            if config in data[model] and data[model][config]['params'] is not None:
                params = data[model][config]['params']
                row.append(f"{params//1000}K")
            else:
                row.append("--")
        
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex_str = "\n".join(latex)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(latex_str)
    
    print(f"\nLaTeX table saved to: {output_file}")
    print(latex_str)


def main():
    parser = argparse.ArgumentParser(description='Analyze comparison results')
    parser.add_argument('results_dir', help='Directory containing result .txt files')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX table')
    parser.add_argument('--output', default='comparison_table.tex', help='Output file for LaTeX table')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    print(f"Analyzing results from: {args.results_dir}")
    print()
    
    # Aggregate results
    data = aggregate_results(args.results_dir)
    
    if not data:
        print("No valid results found!")
        sys.exit(1)
    
    # Generate tables
    generate_table(data, metric='mse')
    generate_table(data, metric='mae')
    generate_parameter_table(data)
    generate_improvement_table(data)
    generate_summary_stats(data)
    
    # Generate LaTeX if requested
    if args.latex:
        generate_publication_latex(data, args.output)
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
