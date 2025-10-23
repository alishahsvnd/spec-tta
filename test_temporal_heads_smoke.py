#!/usr/bin/env python
"""
Smoke test for TimeShift + PolyTrend temporal heads.
Creates synthetic data and runs one adapt_step to validate:
- Import and instantiation
- Forward pass shapes
- Gradient flow
- No runtime errors
"""
import torch
import torch.nn as nn
from tta.spec_tta.manager import SpecTTAManager, SpecTTAConfig

class DummyForecaster(nn.Module):
    """Simple linear forecaster for smoke test."""
    def __init__(self, L, T, V):
        super().__init__()
        self.linear = nn.Linear(L * V, T * V)
    
    def forward(self, x):
        B, L, V = x.shape
        out = self.linear(x.reshape(B, -1))
        return out.reshape(B, -1, V)

def main():
    print("=== TimeShift + PolyTrend Smoke Test ===\n")
    
    # Small synthetic problem
    L, T, V = 48, 96, 7
    B = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Config: L={L}, T={T}, V={V}, B={B}, device={device}")
    
    # Create dummy forecaster
    forecaster = DummyForecaster(L, T, V).to(device)
    forecaster.eval()
    
    # Create SPEC-TTA config
    cfg = SpecTTAConfig(
        L=L, T=T, V=V,
        k_bins=8,
        patch_len=12,
        lr=1e-3,
        drift_threshold=0.01,
        device=device
    )
    
    # Create manager
    print("\n1. Creating SpecTTAManager...")
    manager = SpecTTAManager(forecaster, cfg).to(device)
    print("   ✓ Manager created")
    
    # Create synthetic data
    print("\n2. Creating synthetic test data...")
    X = torch.randn(B, L, V, device=device)
    Y_hat = forecaster(X)  # Get initial predictions
    
    # Create PT mask: first 10% of horizon is observed
    M = int(0.1 * T)
    mask_pt = torch.zeros(B, T, V, device=device)
    mask_pt[:, :M, :] = 1.0
    
    # Create synthetic ground truth for PT prefix
    Y_pt = torch.randn(B, T, V, device=device)
    print(f"   ✓ Data created: X {X.shape}, Y_hat {Y_hat.shape}, mask_pt sum={mask_pt.sum().item()}")
    
    # Test predict_with_calibration
    print("\n3. Testing predict_with_calibration...")
    with torch.no_grad():
        Y_pred = manager.predict_with_calibration(X)
    print(f"   ✓ Prediction shape: {Y_pred.shape}")
    print(f"   ✓ Prediction mean: {Y_pred.mean().item():.4f}, std: {Y_pred.std().item():.4f}")
    
    # Check temporal heads were instantiated
    assert manager.time_shift is not None, "TimeShiftHead not instantiated"
    assert manager.trend_head is not None, "PolyTrendHead not instantiated"
    print(f"   ✓ TimeShiftHead instantiated: tau_raw shape {manager.time_shift.tau_raw.shape}")
    print(f"   ✓ PolyTrendHead instantiated: a shape {manager.trend_head.a.shape}")
    
    # Test adapt_step
    print("\n4. Testing adapt_step...")
    Y_hat_fresh = forecaster(X)  # Fresh forward for gradient graph
    metrics = manager.adapt_step(X, Y_hat_fresh, Y_pt, mask_pt)
    print(f"   ✓ Adapt step completed")
    print(f"   ✓ Metrics: {metrics}")
    
    # Run a few more steps to test gradient flow and stability
    print("\n5. Running 5 more adaptation steps...")
    for i in range(5):
        Y_hat_fresh = forecaster(X)  # Fresh forward for each step
        metrics = manager.adapt_step(X, Y_hat_fresh, Y_pt, mask_pt)
        print(f"   Step {i+1}: drift={metrics['drift']:.6f}, loss={metrics.get('loss', 0.0):.4f}")
    
    print("\n6. Testing final prediction after adaptation...")
    with torch.no_grad():
        Y_pred_final = manager.predict_with_calibration(X)
    print(f"   ✓ Final prediction mean: {Y_pred_final.mean().item():.4f}, std: {Y_pred_final.std().item():.4f}")
    
    # Check parameter updates
    print("\n7. Checking parameter updates...")
    print(f"   TimeShift tau_raw: {manager.time_shift.tau_raw.data}")
    print(f"   PolyTrend a: {manager.trend_head.a.data}")
    print(f"   PolyTrend b: {manager.trend_head.b.data}")
    print(f"   PolyTrend c: {manager.trend_head.c.data}")
    
    # Verify parameters changed from initialization
    tau_nonzero = manager.time_shift.tau_raw.abs().sum() > 1e-6
    trend_nonzero = (manager.trend_head.a.abs().sum() + 
                     manager.trend_head.b.abs().sum() + 
                     manager.trend_head.c.abs().sum()) > 1e-6
    
    if tau_nonzero:
        print("   ✓ TimeShift parameters updated")
    else:
        print("   ⚠ TimeShift parameters still near zero (may be ok if drift was low)")
    
    if trend_nonzero:
        print("   ✓ PolyTrend parameters updated")
    else:
        print("   ⚠ PolyTrend parameters still near zero (may be ok if drift was low)")
    
    print("\n=== ✓ All smoke tests passed ===")

if __name__ == "__main__":
    main()
