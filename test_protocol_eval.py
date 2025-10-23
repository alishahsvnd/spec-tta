#!/usr/bin/env python
"""
Quick integration test for Fix #6: Protocol Evaluation

Tests:
1. Result logging helpers
2. Aggregation from files
3. Protocol evaluation report generation
"""

import os
import sys
import tempfile
import shutil
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools.result_logger import save_run_result, append_to_csv, format_result_filename
from tools.protocol_eval import best_row_counts, pretty_table, generate_report


def test_result_logger():
    """Test result logging utilities."""
    print("=" * 60)
    print("Test 1: Result Logger")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test JSON saving
        json_path = os.path.join(tmpdir, 'test_result.json')
        save_run_result(
            output_path=json_path,
            dataset='ETTh1',
            horizon=96,
            model='iTransformer',
            method='SPEC-TTA',
            mse=0.185735,
            mae=0.340007,
            n_params=910,
            n_updates=30,
            format='json'
        )
        
        assert os.path.exists(json_path), "JSON file not created"
        
        # Test CSV appending
        csv_path = os.path.join(tmpdir, 'results.csv')
        for i in range(3):
            append_to_csv(
                csv_path=csv_path,
                dataset=f'ETTh{i+1}',
                horizon=96,
                model='iTransformer',
                method='SPEC-TTA',
                mse=0.18 + i * 0.01,
                mae=0.34 + i * 0.01,
                n_params=910,
                n_updates=30
            )
        
        assert os.path.exists(csv_path), "CSV file not created"
        
        df = pd.read_csv(csv_path)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        
        # Test filename formatting
        filename = format_result_filename('ETTh1', 96, 'iTransformer', 'SPEC-TTA')
        assert filename == 'SPEC_TTA_iTransformer_ETTh1_96.json', f"Unexpected filename: {filename}"
        
        print("✅ Result logger tests passed\n")


def test_protocol_evaluation():
    """Test protocol evaluation utilities."""
    print("=" * 60)
    print("Test 2: Protocol Evaluation")
    print("=" * 60)
    
    # Create test data
    data = {
        'dataset': ['ETTh1', 'ETTh1', 'ETTh1', 'ETTh2', 'ETTh2', 'ETTh2'],
        'horizon': [96, 96, 96, 96, 96, 96],
        'model': ['iTransformer'] * 6,
        'method': ['NoTTA', 'PETSA', 'SPEC-TTA'] * 2,
        'mse': [0.308, 0.112, 0.186, 0.289, 0.145, 0.138],
        'mae': [0.437, 0.263, 0.340, 0.408, 0.289, 0.276],
        'n_params': [0, 7470, 910, 0, 7470, 910],
    }
    
    df = pd.DataFrame(data)
    
    # Test best-row counts
    counts = best_row_counts(df)
    assert counts['PETSA'] == 1, f"Expected PETSA=1, got {counts.get('PETSA', 0)}"
    assert counts['SPEC-TTA'] == 1, f"Expected SPEC-TTA=1, got {counts.get('SPEC-TTA', 0)}"
    print(f"Best-row counts: {counts}")
    
    # Test comparison table
    table = pretty_table(df)
    assert len(table) == 2, f"Expected 2 rows, got {len(table)}"
    assert 'NoTTA' in table.columns, "NoTTA column missing"
    assert 'PETSA' in table.columns, "PETSA column missing"
    assert 'SPEC-TTA' in table.columns, "SPEC-TTA column missing"
    print("\nComparison table:")
    print(table.to_string(index=False))
    
    # Test full report
    report = generate_report(df)
    assert 'best_row_counts' in report, "Missing best_row_counts"
    assert 'comparison_table' in report, "Missing comparison_table"
    assert 'improvements' in report, "Missing improvements"
    
    print("\n✅ Protocol evaluation tests passed\n")


def test_end_to_end():
    """Test end-to-end workflow."""
    print("=" * 60)
    print("Test 3: End-to-End Workflow")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'aggregated.csv')
        
        # Simulate multiple runs
        runs = [
            ('ETTh1', 96, 'NoTTA', 0.308, 0.437, 0),
            ('ETTh1', 96, 'PETSA', 0.112, 0.263, 7470),
            ('ETTh1', 96, 'SPEC-TTA', 0.186, 0.340, 910),
            ('ETTh2', 96, 'NoTTA', 0.289, 0.408, 0),
            ('ETTh2', 96, 'PETSA', 0.145, 0.289, 7470),
            ('ETTh2', 96, 'SPEC-TTA', 0.138, 0.276, 910),
        ]
        
        for dataset, horizon, method, mse, mae, n_params in runs:
            append_to_csv(
                csv_path=csv_path,
                dataset=dataset,
                horizon=horizon,
                model='iTransformer',
                method=method,
                mse=mse,
                mae=mae,
                n_params=n_params,
                n_updates=30 if method == 'SPEC-TTA' else 143
            )
        
        # Load and evaluate
        df = pd.read_csv(csv_path)
        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"
        
        # Generate report
        report = generate_report(df)
        
        print("Best-row counts:")
        for method, count in sorted(report['best_row_counts'].items(), key=lambda x: -x[1]):
            print(f"  {method}: {count}")
        
        print("\nComparison table:")
        print(report['comparison_table'].to_string(index=False))
        
        print("\n✅ End-to-end test passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Fix #6: Protocol Evaluation — Integration Tests")
    print("=" * 60 + "\n")
    
    try:
        test_result_logger()
        test_protocol_evaluation()
        test_end_to_end()
        
        print("=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)
        print("\nFix #6 is production-ready!")
        print("\nNext steps:")
        print("1. Run: bash scripts/iTransformer/ETTh1_96/run_spec_tta.sh 0 32 0.1 0.005")
        print("2. Check: results/aggregated_results.csv")
        print("3. Report: python tools/protocol_eval.py --results_csv results/aggregated_results.csv")
        print()
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
