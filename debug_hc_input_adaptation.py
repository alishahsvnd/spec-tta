"""
Debug script to see what's happening with the HC adapter on inputs.
"""

import torch
from tta.spec_tta.multi_scale_adapter import HighCapacitySpectralAdapter

# Create synthetic input data
B, L, V = 32, 96, 7
torch.manual_seed(42)

# Simulate input sequences
X = torch.randn(B, L, V)

print("="*80)
print("DEBUG: HC Adapter on Input Sequences")
print("="*80)

# Create Ultra config adapter
adapter = HighCapacitySpectralAdapter(
    L=L, V=V,
    k_low=10, k_mid=20, k_high=19,
    rank=24, gating_dim=128,
    init_scale=0.01
)

print(f"\nInput X shape: {X.shape}")
print(f"Input X mean: {X.mean():.6f}, std: {X.std():.6f}")

# Apply adapter without trend (as we do for inputs)
X_adapted = adapter(X, apply_trend=False)

print(f"\nAdapted X shape: {X_adapted.shape}")
print(f"Adapted X mean: {X_adapted.mean():.6f}, std: {X_adapted.std():.6f}")

# Check if adaptation is too strong
diff = (X_adapted - X).abs()
print(f"\nDifference |X_adapted - X|:")
print(f"  Mean: {diff.mean():.6f}")
print(f"  Max: {diff.max():.6f}")
print(f"  Relative change: {(diff.mean() / X.abs().mean() * 100):.2f}%")

# Check gradient flow
X.requires_grad = True
X_adapted = adapter(X, apply_trend=False)
loss = X_adapted.sum()
loss.backward()

if X.grad is not None:
    print(f"\nGradient check:")
    print(f"  X.grad mean: {X.grad.mean():.6f}")
    print(f"  X.grad max: {X.grad.abs().max():.6f}")
    print(f"  ✅ Gradients flow correctly")
else:
    print(f"\n❌ No gradients!")

print("\n" + "="*80)
print("Key Issue Check:")
print("="*80)

# The problem might be that we're not freezing the model properly
# Or the gradient isn't flowing through the model to the adapter

print("\nHypothesis: The model is frozen, so gradients from Y_hat don't reach adapter")
print("Solution: Need to use .requires_grad_(True) on model temporarily during adaptation")

print("\n" + "="*80)
