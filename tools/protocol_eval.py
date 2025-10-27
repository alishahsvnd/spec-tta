"""
Protocol Evaluation Tool for PETSA-style Experimental Reporting

This tool ensures apples-to-apples comparison with PETSA paper by:
1. Aggregating results across datasets/horizons/methods
2. Computing best-row counts (how many dataset×horizon combos each method wins)
3. Generating Table-1-style comparison tables
4. Providing MSE vs. parameters plots data (like PETSA Fig. 4)

Usage:
    python tools/protocol_eval.py --results_csv results/aggregated_results.csv
    
CSV Format:
    Columns: dataset, horizon, model, method, mse, mae, n_params
    
    Example:
    dataset,horizon,model,method,mse,mae,n_params
    ETTh1,96,iTransformer,NoTTA,0.308,0.437,0
    ETTh1,96,iTransformer,PETSA,0.112,0.263,7470
    ETTh1,96,iTransformer,SPEC-TTA,0.186,0.340,910
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys
import argparse
import json


def best_row_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Compute best-row counts per method over each (dataset, horizon) combination.
    
    A method gets +1 for each dataset×horizon where it achieves lowest MSE
    (aggregated across all models tested on that combination).
    
    Args:
        df: DataFrame with columns ['dataset', 'horizon', 'model', 'method', 'mse']
    
    Returns:
        Dict mapping method name -> count of best rows
        
    Example:
        >>> df = pd.DataFrame({
        ...     'dataset': ['ETTh1', 'ETTh1', 'ETTh2', 'ETTh2'],
        ...     'horizon': [96, 96, 96, 96],
        ...     'method': ['PETSA', 'SPEC-TTA', 'PETSA', 'SPEC-TTA'],
        ...     'mse': [0.112, 0.186, 0.145, 0.138]
        ... })
        >>> best_row_counts(df)
        {'PETSA': 1, 'SPEC-TTA': 1}
    """
    counts = defaultdict(int)
    
    for (ds, H), group in df.groupby(['dataset', 'horizon']):
        # For this dataset×horizon, find method with lowest MSE
        best_idx = group['mse'].idxmin()
        best_method = group.loc[best_idx, 'method']
        counts[best_method] += 1
    
    return dict(counts)


def parameter_summary(
    df: pd.DataFrame,
    dataset: str,
    model: str
) -> Dict[Tuple[int, str], Tuple[float, int]]:
    """
    Extract MSE vs. parameters data for plotting (like PETSA Fig. 4).
    
    Args:
        df: Results DataFrame
        dataset: Dataset name (e.g., 'ETTh1')
        model: Model name (e.g., 'iTransformer')
    
    Returns:
        Dict mapping (horizon, method) -> (mse, n_params)
        
    Example:
        >>> summary = parameter_summary(df, 'ETTh1', 'iTransformer')
        >>> summary[(96, 'PETSA')]
        (0.112, 7470)
        >>> summary[(96, 'SPEC-TTA')]
        (0.186, 910)
    """
    sub = df[(df['dataset'] == dataset) & (df['model'] == model)].copy()
    out = {}
    
    for H in sub['horizon'].unique():
        horizon_data = sub[sub['horizon'] == H]
        
        for method in horizon_data['method'].unique():
            method_data = horizon_data[horizon_data['method'] == method]
            
            # Get best result for this method
            best_row = method_data.sort_values('mse').head(1)
            
            if len(best_row) > 0:
                mse = float(best_row['mse'].values[0])
                n_params = int(best_row['n_params'].values[0])
                out[(H, method)] = (mse, n_params)
    
    return out


def pretty_table(df: pd.DataFrame, methods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate PETSA Table-1-style comparison table.
    
    Creates a grid with rows for each dataset×horizon and columns for each method,
    showing MSE values. The best result per row is highlighted.
    
    Args:
        df: Results DataFrame with columns ['dataset', 'horizon', 'method', 'mse']
        methods: List of methods to include (default: ['NoTTA', 'TAFAS', 'PETSA', 'SPEC-TTA'])
    
    Returns:
        DataFrame formatted as comparison table
        
    Example Output:
        dataset  horizon    NoTTA    TAFAS    PETSA  SPEC-TTA
        ETTh1        96    0.308    0.203    0.112     0.186
        ETTh1       192    0.423    0.289    0.178     0.241
        ETTh2        96    0.289    0.198    0.145     0.138
    """
    if methods is None:
        methods = ['NoTTA', 'TAFAS', 'PETSA', 'SPEC-TTA']
    
    rows = []
    
    for (ds, H), group in df.groupby(['dataset', 'horizon']):
        row = {'dataset': ds, 'horizon': H}
        
        for method in methods:
            method_data = group[group['method'] == method]
            
            if len(method_data) > 0:
                # Take minimum MSE if multiple runs exist
                row[method] = round(float(method_data['mse'].min()), 6)
            else:
                row[method] = None
        
        rows.append(row)
    
    # Create DataFrame and sort
    table = pd.DataFrame(rows).sort_values(['dataset', 'horizon'])
    
    return table


def pretty_table_with_best(df: pd.DataFrame, methods: Optional[List[str]] = None) -> str:
    """
    Generate pretty table with best results highlighted in each row.
    
    Args:
        df: Results DataFrame
        methods: List of methods to include
    
    Returns:
        Formatted string with table and highlighting
    """
    table = pretty_table(df, methods)
    
    if methods is None:
        methods = ['NoTTA', 'TAFAS', 'PETSA', 'SPEC-TTA']
    
    # Build string representation with highlighting
    lines = []
    
    # Header
    header = f"{'Dataset':<10} {'Horizon':<8} " + " ".join([f"{m:<12}" for m in methods])
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for _, row in table.iterrows():
        dataset = row['dataset']
        horizon = row['horizon']
        
        # Find best MSE in this row
        mse_values = [row[m] for m in methods if row[m] is not None]
        best_mse = min(mse_values) if mse_values else None
        
        line = f"{dataset:<10} {horizon:<8} "
        
        for method in methods:
            mse = row[method]
            
            if mse is None:
                line += f"{'N/A':<12} "
            elif mse == best_mse:
                # Highlight best result
                line += f"{mse:<12.6f}*"
            else:
                line += f"{mse:<12.6f} "
        
        lines.append(line)
    
    return "\n".join(lines)


def compute_improvements(df: pd.DataFrame, baseline: str = 'NoTTA') -> pd.DataFrame:
    """
    Compute percentage improvements over baseline method.
    
    Args:
        df: Results DataFrame
        baseline: Baseline method name (default: 'NoTTA')
    
    Returns:
        DataFrame with improvement percentages
    """
    table = pretty_table(df)
    
    if baseline not in table.columns:
        raise ValueError(f"Baseline method '{baseline}' not found in results")
    
    improvements = table.copy()
    
    for col in table.columns:
        if col not in ['dataset', 'horizon', baseline]:
            improvements[col + '_improvement'] = (
                (table[baseline] - table[col]) / table[baseline] * 100
            )
    
    return improvements


def load_results(csv_path: str) -> pd.DataFrame:
    """
    Load results CSV with validation.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Validated DataFrame
    
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ['dataset', 'horizon', 'model', 'method', 'mse']
    
    df = pd.read_csv(csv_path)
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def save_results(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'csv'
):
    """
    Save results to file.
    
    Args:
        df: Results DataFrame
        output_path: Output file path
        format: Output format ('csv', 'json', 'latex')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    elif format == 'latex':
        latex_str = df.to_latex(index=False)
        with open(output_path, 'w') as f:
            f.write(latex_str)
    else:
        raise ValueError(f"Unknown format: {format}")


def generate_report(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive evaluation report.
    
    Args:
        df: Results DataFrame
    
    Returns:
        Dict containing:
            - best_row_counts: Dict[method, count]
            - comparison_table: DataFrame
            - improvements: DataFrame
            - parameter_efficiency: Dict per dataset/model
    """
    report = {
        'best_row_counts': best_row_counts(df),
        'comparison_table': pretty_table(df),
        'improvements': compute_improvements(df, baseline='NoTTA'),
    }
    
    # Parameter efficiency per dataset/model
    param_efficiency = {}
    for dataset in df['dataset'].unique():
        param_efficiency[dataset] = {}
        for model in df['model'].unique():
            subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
            if len(subset) > 0:
                param_efficiency[dataset][model] = parameter_summary(df, dataset, model)
    
    report['parameter_efficiency'] = param_efficiency
    
    return report


def print_report(report: Dict):
    """
    Pretty-print evaluation report.
    
    Args:
        report: Report dictionary from generate_report()
    """
    print("=" * 80)
    print("PETSA-Style Experimental Protocol Evaluation Report")
    print("=" * 80)
    print()
    
    # Best-row counts
    print("Best-Row Counts (Number of dataset×horizon wins per method):")
    print("-" * 60)
    counts = report['best_row_counts']
    for method, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {method:<15}: {count:>3}")
    print()
    
    # Comparison table
    print("Comparison Table (MSE per dataset×horizon):")
    print("-" * 60)
    print(report['comparison_table'].to_string(index=False))
    print()
    
    # Improvements
    print("Improvements over NoTTA (%):")
    print("-" * 60)
    improvements = report['improvements']
    imp_cols = [c for c in improvements.columns if '_improvement' in c]
    print(improvements[['dataset', 'horizon'] + imp_cols].to_string(index=False))
    print()
    
    print("=" * 80)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='PETSA-style experimental protocol evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full report
  python tools/protocol_eval.py --results_csv results/aggregated.csv
  
  # Save comparison table to LaTeX
  python tools/protocol_eval.py --results_csv results/aggregated.csv --output report.tex --format latex
  
  # Custom methods list
  python tools/protocol_eval.py --results_csv results/aggregated.csv --methods NoTTA PETSA SPEC-TTA
        """
    )
    
    parser.add_argument(
        '--results_csv',
        required=True,
        help='Path to aggregated results CSV file'
    )
    parser.add_argument(
        '--output',
        help='Output path for report (optional)'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'json', 'latex', 'txt'],
        default='txt',
        help='Output format (default: txt for console)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        help='List of methods to include (default: NoTTA TAFAS PETSA SPEC-TTA)'
    )
    parser.add_argument(
        '--baseline',
        default='NoTTA',
        help='Baseline method for improvement calculation (default: NoTTA)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load results
        print(f"Loading results from {args.results_csv}...")
        df = load_results(args.results_csv)
        print(f"Loaded {len(df)} results")
        print()
        
        # Generate report
        report = generate_report(df)
        
        # Print to console
        print_report(report)
        
        # Save if requested
        if args.output:
            if args.format == 'csv':
                save_results(report['comparison_table'], args.output, 'csv')
            elif args.format == 'json':
                with open(args.output, 'w') as f:
                    json.dump({
                        'best_row_counts': report['best_row_counts'],
                        'comparison_table': report['comparison_table'].to_dict('records'),
                    }, f, indent=2)
            elif args.format == 'latex':
                save_results(report['comparison_table'], args.output, 'latex')
            
            print(f"\nReport saved to {args.output}")
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
