"""
Result Logging Utilities

Helper functions to save individual run results in standardized format
for later aggregation.
"""

import json
import os
from typing import Dict, Optional


def save_run_result(
    output_path: str,
    dataset: str,
    horizon: int,
    model: str,
    method: str,
    mse: float,
    mae: float,
    n_params: int,
    n_updates: Optional[int] = None,
    extra_metrics: Optional[Dict] = None,
    format: str = 'json'
):
    """
    Save individual run result in standardized format.
    
    Args:
        output_path: Output file path
        dataset: Dataset name (e.g., 'ETTh1')
        horizon: Prediction horizon (e.g., 96)
        model: Model name (e.g., 'iTransformer')
        method: Method name (e.g., 'SPEC-TTA')
        mse: Mean squared error
        mae: Mean absolute error
        n_params: Number of trainable parameters
        n_updates: Number of adaptation updates (optional)
        extra_metrics: Additional metrics to save (optional)
        format: Output format ('json' or 'txt')
    
    Example:
        >>> save_run_result(
        ...     'results/SPEC_TTA_iTransformer_ETTh1_96.json',
        ...     dataset='ETTh1',
        ...     horizon=96,
        ...     model='iTransformer',
        ...     method='SPEC-TTA',
        ...     mse=0.185735,
        ...     mae=0.340007,
        ...     n_params=910,
        ...     n_updates=30
        ... )
    """
    # Build result dictionary
    result = {
        'dataset': dataset,
        'horizon': horizon,
        'model': model,
        'method': method,
        'mse': float(mse),
        'mae': float(mae),
        'n_params': int(n_params),
    }
    
    if n_updates is not None:
        result['n_updates'] = int(n_updates)
    
    if extra_metrics:
        result.update(extra_metrics)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save in requested format
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    elif format == 'txt':
        with open(output_path, 'w') as f:
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Horizon: {horizon}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Method: {method}\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"N_Params: {n_params}\n")
            
            if n_updates is not None:
                f.write(f"N_Updates: {n_updates}\n")
            
            if extra_metrics:
                for key, value in extra_metrics.items():
                    f.write(f"{key}: {value}\n")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def append_to_csv(
    csv_path: str,
    dataset: str,
    horizon: int,
    model: str,
    method: str,
    mse: float,
    mae: float,
    n_params: int,
    n_updates: Optional[int] = None,
):
    """
    Append result to CSV file (creates file with header if doesn't exist).
    
    Args:
        csv_path: Path to CSV file
        dataset: Dataset name
        horizon: Prediction horizon
        model: Model name
        method: Method name
        mse: Mean squared error
        mae: Mean absolute error
        n_params: Number of trainable parameters
        n_updates: Number of adaptation updates (optional)
    
    Example:
        >>> append_to_csv(
        ...     'results/all_results.csv',
        ...     dataset='ETTh1',
        ...     horizon=96,
        ...     model='iTransformer',
        ...     method='SPEC-TTA',
        ...     mse=0.185735,
        ...     mae=0.340007,
        ...     n_params=910,
        ...     n_updates=30
        ... )
    """
    import csv
    
    # Create directory if needed
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Open in append mode
    with open(csv_path, 'a', newline='') as f:
        fieldnames = ['dataset', 'horizon', 'model', 'method', 'mse', 'mae', 'n_params']
        
        if n_updates is not None:
            fieldnames.append('n_updates')
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if new file
        if not file_exists:
            writer.writeheader()
        
        # Write row
        row = {
            'dataset': dataset,
            'horizon': horizon,
            'model': model,
            'method': method,
            'mse': mse,
            'mae': mae,
            'n_params': n_params,
        }
        
        if n_updates is not None:
            row['n_updates'] = n_updates
        
        writer.writerow(row)


def format_result_filename(
    dataset: str,
    horizon: int,
    model: str,
    method: str,
    extension: str = 'json'
) -> str:
    """
    Generate standardized filename for result.
    
    Args:
        dataset: Dataset name
        horizon: Prediction horizon
        model: Model name
        method: Method name
        extension: File extension (default: 'json')
    
    Returns:
        Formatted filename
    
    Example:
        >>> format_result_filename('ETTh1', 96, 'iTransformer', 'SPEC-TTA')
        'SPEC-TTA_iTransformer_ETTh1_96.json'
    """
    # Sanitize method name
    method_clean = method.replace('-', '_').replace(' ', '_')
    
    return f"{method_clean}_{model}_{dataset}_{horizon}.{extension}"
