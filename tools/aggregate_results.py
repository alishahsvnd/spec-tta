#!/usr/bin/env python
"""
Result Aggregation Script

Collects individual run logs and aggregates them into a single CSV
for protocol evaluation.

Usage:
    python tools/aggregate_results.py --results_dir results/ --output aggregated_results.csv
"""

import os
import argparse
import pandas as pd
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional


def parse_run_log(log_path: str) -> Optional[Dict]:
    """
    Parse a single run log file.
    
    Supports formats:
    - JSON: {"dataset": "ETTh1", "horizon": 96, "method": "SPEC-TTA", "mse": 0.186, ...}
    - TXT: Lines like "MSE: 0.186", "MAE: 0.340", etc.
    
    Args:
        log_path: Path to log file
    
    Returns:
        Dict with run metadata and metrics, or None if parsing fails
    """
    try:
        # Try JSON first
        if log_path.endswith('.json'):
            with open(log_path, 'r') as f:
                data = json.load(f)
                return data
        
        # Try TXT format
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # Parse TXT (format: "Key: Value")
        data = {}
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Try to convert to float
                try:
                    value = float(value)
                except ValueError:
                    pass
                
                data[key] = value
        
        return data if data else None
        
    except Exception as e:
        print(f"Warning: Failed to parse {log_path}: {e}")
        return None


def parse_filename_metadata(filename: str) -> Dict:
    """
    Extract metadata from filename.
    
    Expected format: {METHOD}_{MODEL}_{DATASET}_{HORIZON}.txt
    Example: SPEC_TTA_iTransformer_ETTh1_96.txt
    
    Args:
        filename: Base filename (without path)
    
    Returns:
        Dict with extracted metadata
    """
    parts = filename.replace('.txt', '').replace('.json', '').split('_')
    
    metadata = {}
    
    # Try to extract dataset (ETTh1, ETTh2, ETTm1, ETTm2, etc.)
    for part in parts:
        if part.startswith('ETT') or part in ['Exchange', 'Weather', 'Electricity']:
            metadata['dataset'] = part
        
        # Extract horizon (numeric)
        try:
            horizon = int(part)
            if horizon in [96, 192, 336, 720]:
                metadata['horizon'] = horizon
        except ValueError:
            pass
    
    # Extract method (PETSA, SPEC-TTA, etc.)
    for part in parts:
        if part in ['PETSA', 'TAFAS', 'NoTTA'] or 'SPEC' in part:
            metadata['method'] = part.replace('_', '-')
    
    # Extract model
    for part in parts:
        if part in ['iTransformer', 'PatchTST', 'DLinear', 'FreTS', 'MICN']:
            metadata['model'] = part
    
    return metadata


def aggregate_from_directory(
    results_dir: str,
    pattern: str = "*.txt"
) -> pd.DataFrame:
    """
    Aggregate all result files from directory.
    
    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern for result files (default: "*.txt")
    
    Returns:
        DataFrame with aggregated results
    """
    results = []
    
    # Find all matching files
    search_path = os.path.join(results_dir, '**', pattern)
    files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(files)} result files matching pattern '{pattern}'")
    
    for file_path in files:
        # Parse file content
        data = parse_run_log(file_path)
        
        if data is None:
            continue
        
        # Extract metadata from filename
        filename = os.path.basename(file_path)
        metadata = parse_filename_metadata(filename)
        
        # Merge data and metadata
        result = {**metadata, **data}
        
        # Normalize column names
        result = {
            'dataset': result.get('dataset', 'unknown'),
            'horizon': result.get('horizon', 0),
            'model': result.get('model', 'unknown'),
            'method': result.get('method', 'unknown'),
            'mse': result.get('mse', result.get('final_mse', None)),
            'mae': result.get('mae', result.get('final_mae', None)),
            'n_params': result.get('n_params', result.get('total_trainable_parameters', 0)),
        }
        
        # Validate required fields
        if result['mse'] is not None:
            results.append(result)
        else:
            print(f"Warning: Skipping {filename} - missing MSE")
    
    df = pd.DataFrame(results)
    
    print(f"Successfully aggregated {len(df)} results")
    
    return df


def aggregate_from_csv_files(csv_paths: List[str]) -> pd.DataFrame:
    """
    Aggregate multiple CSV files into single DataFrame.
    
    Args:
        csv_paths: List of CSV file paths
    
    Returns:
        Combined DataFrame
    """
    dfs = []
    
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
    
    if not dfs:
        raise ValueError("No valid CSV files found")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate individual run results for protocol evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all TXT files from results directory
  python tools/aggregate_results.py --results_dir results/ --output aggregated.csv
  
  # Aggregate specific CSV files
  python tools/aggregate_results.py --csv_files run1.csv run2.csv --output aggregated.csv
  
  # Aggregate with custom pattern
  python tools/aggregate_results.py --results_dir results/ --pattern "*_ETTh1_*.txt" --output etth1_results.csv
        """
    )
    
    parser.add_argument(
        '--results_dir',
        help='Directory containing result files'
    )
    parser.add_argument(
        '--pattern',
        default='*.txt',
        help='Glob pattern for result files (default: *.txt)'
    )
    parser.add_argument(
        '--csv_files',
        nargs='+',
        help='List of CSV files to aggregate'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV path'
    )
    
    args = parser.parse_args()
    
    try:
        # Aggregate from directory or CSV files
        if args.results_dir:
            df = aggregate_from_directory(args.results_dir, args.pattern)
        elif args.csv_files:
            df = aggregate_from_csv_files(args.csv_files)
        else:
            print("Error: Must specify --results_dir or --csv_files")
            return 1
        
        if len(df) == 0:
            print("Error: No results found")
            return 1
        
        # Sort by dataset, horizon, method
        df = df.sort_values(['dataset', 'horizon', 'method'])
        
        # Save
        df.to_csv(args.output, index=False)
        print(f"\nAggregated results saved to {args.output}")
        
        # Print summary
        print("\nSummary:")
        print(f"  Total results: {len(df)}")
        print(f"  Datasets: {', '.join(df['dataset'].unique())}")
        print(f"  Horizons: {', '.join(map(str, sorted(df['horizon'].unique())))}")
        print(f"  Methods: {', '.join(df['method'].unique())}")
        print(f"  Models: {', '.join(df['model'].unique())}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
