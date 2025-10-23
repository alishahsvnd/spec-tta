"""
Direct comparison on ETTh1 dataset without full training pipeline.
Loads pre-trained iTransformer and tests TTA methods.
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, '/home/alishah/PETSA')

from tta.spec_tta.multi_scale_adapter import HighCapacitySpectralAdapter
from config import get_cfg_defaults
from models.build import build_model
from datasets.loader import get_test_dataloader
import pandas as pd

def load_model_and_data():
    """Load pre-trained model and test data."""
    cfg = get_cfg_defaults()
    
    # ETTh1 configuration
    cfg.DATA.NAME = 'ETTh1'
    cfg.DATA.N_VAR = 7
    cfg.DATA.SEQ_LEN = 96
    cfg.DATA.PRED_LEN = 96
    cfg.MODEL.NAME = 'iTransformer'
    cfg.MODEL.enc_in = 7
    cfg.TEST.BATCH_SIZE = 32
    
    # Build model
    model = build_model(cfg)
    
    # Load checkpoint if exists
    checkpoint_path = f'checkpoints/iTransformer/ETTh1_96/checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found, using random weights")
    
    model.eval()
    
    # Get test data
    test_loader = get_test_dataloader(cfg)
    
    return model, test_loader, cfg

def test_adapter(name, adapter, model, test_loader, cfg, n_batches=10):
    """Test an adapter on real data."""
    print(f"\nTesting: {name}")
    print("=" * 70)
    
    mse_list = []
    mae_list = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if adapter is not None:
        adapter = adapter.to(device)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            
            batch_x = batch[0].to(device)  # [B, L, V]
            batch_y = batch[1].to(device)  # [B, L_label + T, V]
            
            # Get base prediction
            outputs = model(batch_x, None, None, None)
            pred = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            pred = pred[:, -cfg.DATA.PRED_LEN:, :]  # [B, T, V]
            
            # Apply adapter if provided
            if adapter is not None:
                pred = adapter(pred)
            
            # Ground truth
            gt = batch_y[:, -cfg.DATA.PRED_LEN:, :]
            
            # Metrics
            mse = F.mse_loss(pred, gt).item()
            mae = F.l1_loss(pred, gt).item()
            
            mse_list.append(mse)
            mae_list.append(mae)
    
    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)
    params = sum(p.numel() for p in adapter.parameters()) if adapter else 0
    
    print(f"MSE: {avg_mse:.6f}")
    print(f"MAE: {avg_mae:.6f}")
    print(f"Parameters: {params:,}")
    print("=" * 70)
    
    return {
        'name': name,
        'mse': avg_mse,
        'mae': avg_mae,
        'params': params
    }

def main():
    print("\n" + "=" * 70)
    print("ETTh1 REAL DATA COMPARISON")
    print("High-Capacity SPEC-TTA Configurations")
    print("=" * 70)
    
    # Load model and data
    print("\nLoading model and data...")
    model, test_loader, cfg = load_model_and_data()
    
    results = []
    
    # Test 1: No adaptation (baseline)
    print("\n[1/4] No Adaptation (Baseline)")
    results.append(test_adapter("No-TTA", None, model, test_loader, cfg))
    
    # Test 2: Medium capacity
    print("\n[2/4] SPEC-TTA Medium Capacity (12K params)")
    medium = HighCapacitySpectralAdapter(
        L=96, V=7, k_low=6, k_mid=12, k_high=20,
        rank=8, gating_dim=32, init_scale=0.01
    )
    results.append(test_adapter("SPEC-TTA Medium", medium, model, test_loader, cfg))
    
    # Test 3: High capacity
    print("\n[3/4] SPEC-TTA High Capacity (24K params)")
    high = HighCapacitySpectralAdapter(
        L=96, V=7, k_low=8, k_mid=16, k_high=25,
        rank=16, gating_dim=64, init_scale=0.01
    )
    results.append(test_adapter("SPEC-TTA High", high, model, test_loader, cfg))
    
    # Test 4: Ultra capacity
    print("\n[4/4] SPEC-TTA Ultra Capacity (36K params)")
    ultra = HighCapacitySpectralAdapter(
        L=96, V=7, k_low=10, k_mid=20, k_high=19,
        rank=24, gating_dim=128, init_scale=0.01
    )
    results.append(test_adapter("SPEC-TTA Ultra", ultra, model, test_loader, cfg))
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'MSE':<12} {'MAE':<12} {'Params':<12} {'vs No-TTA':<12}")
    print("-" * 70)
    
    baseline_mse = results[0]['mse']
    for r in results:
        improvement = ((baseline_mse - r['mse']) / baseline_mse) * 100 if baseline_mse > 0 else 0
        print(f"{r['name']:<25} {r['mse']:<12.6f} {r['mae']:<12.6f} {r['params']:<12,} {improvement:+.2f}%")
    
    print("=" * 70)
    
    # Best performer
    best = min(results[1:], key=lambda x: x['mse'])
    print(f"\n✅ BEST TTA METHOD: {best['name']}")
    print(f"   MSE: {best['mse']:.6f}")
    print(f"   Improvement: {((baseline_mse - best['mse']) / baseline_mse) * 100:+.2f}%")
    print(f"   Parameters: {best['params']:,}")
    
    print("\n📊 COMPARISON WITH PETSA (expected ~55K params):")
    for r in results[1:]:
        efficiency = 55296 / r['params'] if r['params'] > 0 else 0
        print(f"   {r['name']}: {efficiency:.1f}x more efficient")
    
    print("\n" + "=" * 70)
    print("Note: These are untrained adapters (random initialization)")
    print("Results show architectural capacity, not optimized performance")
    print("Run full training for publication-ready results")
    print("=" * 70)

if __name__ == "__main__":
    main()
