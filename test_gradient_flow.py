"""
Debug: Check if gradients are actually flowing to adapter_out
"""

import torch
import torch.nn as nn
from tta.spec_tta.spectral_adapter import SpectralAdapter, TrendHead
from tta.spec_tta.multi_scale_adapter import HighCapacitySpectralAdapter

B, L, T, V = 32, 96, 96, 7
torch.manual_seed(42)

print("="*80)
print("Gradient Flow Test: Two-Adapter Design")
print("="*80)

# Create the two adapters
adapter_in = SpectralAdapter(L, V, list(range(32)))
adapter_out = HighCapacitySpectralAdapter(T, V, k_low=10, k_mid=20, k_high=19, rank=24, gating_dim=128)
trend_head = TrendHead(T, V)

# Create fake model (identity for simplicity)
class FakeModel(nn.Module):
    def forward(self, x):
        return x  # Just pass through

model = FakeModel()

# Setup optimizer
params = list(adapter_in.parameters()) + list(adapter_out.parameters()) + list(trend_head.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

# Simulate one adaptation step
X = torch.randn(B, L, V)
Y_true = torch.randn(B, T, V)

print(f"\nInput X: {X.shape}")
print(f"Target Y_true: {Y_true.shape}")

# Forward pass (like in the wrapper)
with torch.no_grad():
    X_cal = adapter_in(X)
    Y_hat = model(X_cal)  # Frozen model

print(f"Model output Y_hat: {Y_hat.shape}")

# Adapt output (WITH gradients)
Y_adapted = adapter_out(Y_hat, apply_trend=False)
Y_final = trend_head(Y_adapted)

print(f"Adapted Y_final: {Y_final.shape}")

# Compute loss
loss = torch.nn.functional.mse_loss(Y_final, Y_true)
print(f"\nLoss: {loss.item():.6f}")

# Backward
optimizer.zero_grad()
loss.backward()

# Check gradients
print(f"\nGradient Check:")
print(f"  adapter_in gradients: {sum(p.grad.abs().sum().item() for p in adapter_in.parameters() if p.grad is not None):.6f}")
print(f"  adapter_out gradients: {sum(p.grad.abs().sum().item() for p in adapter_out.parameters() if p.grad is not None):.6f}")
print(f"  trend_head gradients: {sum(p.grad.abs().sum().item() for p in trend_head.parameters() if p.grad is not None):.6f}")

# Optimize
optimizer.step()

# Check if parameters changed
print(f"\nAfter optimizer step:")
for name, module in [('adapter_in', adapter_in), ('adapter_out', adapter_out), ('trend_head', trend_head)]:
    param_sum = sum(p.abs().sum().item() for p in module.parameters())
    print(f"  {name} param sum: {param_sum:.6f}")

# Try another forward pass to see if loss decreases
with torch.no_grad():
    X_cal2 = adapter_in(X)
    Y_hat2 = model(X_cal2)
    Y_adapted2 = adapter_out(Y_hat2, apply_trend=False)
    Y_final2 = trend_head(Y_adapted2)
    loss2 = torch.nn.functional.mse_loss(Y_final2, Y_true)

print(f"\nLoss after 1 step: {loss2.item():.6f}")
print(f"Loss improvement: {(loss.item() - loss2.item()):.6f}")

if loss2.item() < loss.item():
    print("✅ Learning is working!")
else:
    print("❌ No learning!")

print("="*80)
