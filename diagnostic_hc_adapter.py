"""
Diagnostic script to check what's happening with the high-capacity adapter.
"""

import torch
import torch.nn as nn
import numpy as np
from tta.spec_tta.multi_scale_adapter import HighCapacitySpectralAdapter

# Create synthetic data similar to ETTh1
B, L, V = 32, 96, 7
torch.manual_seed(42)

# Simulate predictions from a model (with some error)
y_true = torch.randn(B, L, V)
y_pred = y_true + 0.1 * torch.randn(B, L, V)  # Add noise

print("=" * 80)
print("Diagnostic: High-Capacity Spectral Adapter")
print("=" * 80)

# Test all three configurations
configs = {
    'Medium': {'k_low': 6, 'k_mid': 12, 'k_high': 20, 'rank': 8, 'gating_dim': 32},
    'High': {'k_low': 8, 'k_mid': 16, 'k_high': 25, 'rank': 16, 'gating_dim': 64},
    'Ultra': {'k_low': 10, 'k_mid': 20, 'k_high': 19, 'rank': 24, 'gating_dim': 128},
}

for name, config in configs.items():
    print(f"\n{'='*80}")
    print(f"Testing {name} Configuration")
    print(f"{'='*80}")
    
    # Create adapter
    adapter = HighCapacitySpectralAdapter(
        L=L, V=V,
        k_low=config['k_low'],
        k_mid=config['k_mid'],
        k_high=config['k_high'],
        rank=config['rank'],
        gating_dim=config['gating_dim'],
        init_scale=0.01
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in adapter.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Initial MSE
    initial_mse = torch.nn.functional.mse_loss(y_pred, y_true).item()
    print(f"\nInitial MSE (before adaptation): {initial_mse:.6f}")
    
    # Test forward pass without trend
    y_adapted_no_trend = adapter(y_pred, apply_trend=False)
    mse_no_trend = torch.nn.functional.mse_loss(y_adapted_no_trend, y_true).item()
    print(f"MSE after adaptation (no trend): {mse_no_trend:.6f}")
    improvement_no_trend = ((initial_mse - mse_no_trend) / initial_mse) * 100
    print(f"  → {'Improvement' if improvement_no_trend > 0 else 'Degradation'}: {improvement_no_trend:+.2f}%")
    
    # Test forward pass with trend
    y_adapted_with_trend = adapter(y_pred, apply_trend=True)
    mse_with_trend = torch.nn.functional.mse_loss(y_adapted_with_trend, y_true).item()
    print(f"MSE after adaptation (with trend): {mse_with_trend:.6f}")
    improvement_with_trend = ((initial_mse - mse_with_trend) / initial_mse) * 100
    print(f"  → {'Improvement' if improvement_with_trend > 0 else 'Degradation'}: {improvement_with_trend:+.2f}%")
    
    # Now try to optimize the adapter
    print(f"\nTraining adapter for 10 steps...")
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    
    for step in range(10):
        optimizer.zero_grad()
        y_adapted = adapter(y_pred, apply_trend=True)
        loss = torch.nn.functional.mse_loss(y_adapted, y_true)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: MSE = {loss.item():.6f}")
    
    # Final MSE after training
    with torch.no_grad():
        y_adapted_final = adapter(y_pred, apply_trend=True)
        final_mse = torch.nn.functional.mse_loss(y_adapted_final, y_true).item()
    
    print(f"\nFinal MSE (after 10 steps): {final_mse:.6f}")
    final_improvement = ((initial_mse - final_mse) / initial_mse) * 100
    print(f"  → {'Improvement' if final_improvement > 0 else 'Degradation'}: {final_improvement:+.2f}%")

print(f"\n{'='*80}")
print("Diagnostic Complete")
print(f"{'='*80}")
print("\nKey Observations:")
print("1. If adapter helps without training → Architecture is good")
print("2. If adapter hurts without training → Initialization issue")
print("3. If adapter improves with training → Learning works")
print("4. If adapter doesn't improve with training → Optimization issue")
print("")
