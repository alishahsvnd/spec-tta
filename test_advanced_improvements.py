#!/usr/bin/env python
"""
Test script for advanced SPEC-TTA improvements.
Tests each improvement individually to ensure no regression on short horizons.
"""
import sys
import torch
import torch.nn as nn

# Simulated test setup
class DummyForecaster(nn.Module):
    def __init__(self, L, T, V):
        super().__init__()
        self.fc = nn.Linear(L * V, T * V)
    
    def forward(self, x):
        B, L, V = x.shape
        return self.fc(x.reshape(B, -1)).reshape(B, -1, V)


def test_tail_damping():
    """Test B: Tail damping"""
    print("\n=== Testing B: Tail Damping ===")
    from tta.spec_tta.advanced_improvements import TailDampingHead
    
    T, V = 336, 7
    damping = TailDampingHead(T, V, damping_start=0.6, damping_strength=0.3)
    
    y_base = torch.randn(4, T, V)
    y_correction = y_base + torch.randn(4, T, V) * 0.5
    
    y_damped = damping(y_correction, y_base)
    
    # Check that early timesteps are less damped than late
    early_correction = (y_correction[:, :T//3, :] - y_base[:, :T//3, :]).abs().mean()
    late_correction = (y_correction[:, 2*T//3:, :] - y_base[:, 2*T//3:, :]).abs().mean()
    
    early_damped = (y_damped[:, :T//3, :] - y_base[:, :T//3, :]).abs().mean()
    late_damped = (y_damped[:, 2*T//3:, :] - y_base[:, 2*T//3:, :]).abs().mean()
    
    print(f"  Early correction: {early_correction:.4f} -> {early_damped:.4f} (damped)")
    print(f"  Late correction:  {late_correction:.4f} -> {late_damped:.4f} (damped)")
    print(f"  Late damping ratio: {late_damped / late_correction:.3f} (should be < early ratio)")
    
    assert late_damped < early_damped, "Late timesteps should be more damped"
    print("  ✓ Tail damping works correctly")
    return True


def test_adaptive_loss_schedule():
    """Test E: Adaptive loss weights"""
    print("\n=== Testing E: Adaptive Loss Schedule ===")
    from tta.spec_tta.advanced_improvements import get_adaptive_loss_weights
    
    for h in [96, 192, 336, 720]:
        weights = get_adaptive_loss_weights(h)
        print(f"  H={h:3d}: beta_freq={weights['beta_freq']:.4f}, "
              f"lambda_pw={weights['lambda_pw']:.2f}, "
              f"lambda_prox={weights['lambda_prox']:.5f}, "
              f"lr={weights['lr']:.5f}")
    
    # Verify decreasing frequency loss with horizon
    w96 = get_adaptive_loss_weights(96)
    w720 = get_adaptive_loss_weights(720)
    assert w720['beta_freq'] < w96['beta_freq'], "Longer horizons should have lower freq loss"
    assert w720['lambda_prox'] > w96['lambda_prox'], "Longer horizons should have stronger regularization"
    print("  ✓ Adaptive schedule correct")
    return True


def test_output_only_criterion():
    """Test D: Output-only mode selection"""
    print("\n=== Testing D: Output-Only Mode Criterion ===")
    from tta.spec_tta.advanced_improvements import should_use_output_only
    
    for h in [96, 192, 240, 336, 720]:
        output_only = should_use_output_only(h)
        print(f"  H={h:3d}: Output-only mode = {output_only}")
    
    assert not should_use_output_only(96), "Short horizons should use full adaptation"
    assert should_use_output_only(336), "Long horizons should use output-only"
    print("  ✓ Output-only criterion works")
    return True


def test_safe_update_manager():
    """Test F: Safe update mechanism"""
    print("\n=== Testing F: Safe Update Manager ===")
    from tta.spec_tta.advanced_improvements import SafeUpdateManager
    
    manager = SafeUpdateManager(max_param_norm=5.0, patience=3)
    
    # Create dummy modules
    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(10, 10))
    
    modules = {'test': DummyModule()}
    
    # Test 1: Normal update (loss improves)
    manager.before_update(modules)
    metrics = manager.after_update(current_loss=1.0, modules=modules)
    print(f"  Normal update: {metrics}")
    assert metrics['safe_update_applied'], "Should apply normal update"
    
    # Test 2: Parameter norm clipping
    modules['test'].weight.data.fill_(100.0)  # Large values
    manager.before_update(modules)
    metrics = manager.after_update(current_loss=0.95, modules=modules)
    print(f"  After large params: {metrics}")
    assert metrics['params_clipped'], "Should clip large parameters"
    assert modules['test'].weight.data.norm() <= 5.1, "Norm should be clipped"
    
    # Test 3: Rollback on no improvement
    manager.best_loss = 0.5
    manager.no_improve_count = 0
    rollback_happened = False
    for i in range(manager.patience + 1):
        manager.before_update(modules)
        metrics = manager.after_update(current_loss=0.6, modules=modules)  # No improvement
        if metrics['rollback_occurred']:
            rollback_happened = True
            break
    print(f"  After {manager.patience}+ bad updates: rollback={rollback_happened}")
    assert rollback_happened, "Should rollback after patience exhausted"
    
    print("  ✓ Safe update manager works")
    return True


def test_local_spectral_adapter():
    """Test C: Local spectral adapter"""
    print("\n=== Testing C: Local Spectral Adapter ===")
    from tta.spec_tta.advanced_improvements import LocalSpectralAdapter
    
    T, V = 336, 7
    k_bins = list(range(8))  # First 8 bins
    
    adapter = LocalSpectralAdapter(T, V, k_bins, split_ratio=0.5)
    
    x = torch.randn(4, T, V)
    y = adapter(x)
    
    assert y.shape == x.shape, "Output shape should match input"
    
    # Check that parameters exist for both segments
    assert adapter.g_early_real.shape == (V, len(k_bins))
    assert adapter.g_late_real.shape == (V, len(k_bins))
    
    print(f"  Split point: t={adapter.split_t}")
    print(f"  Crossfade width: {adapter.crossfade_width}")
    print(f"  Early weight at t=0: {adapter.early_weight[0, 0, 0]:.3f}")
    print(f"  Early weight at t={T-1}: {adapter.early_weight[0, T-1, 0]:.3f}")
    print(f"  Late weight at t=0: {adapter.late_weight[0, 0, 0]:.3f}")
    print(f"  Late weight at t={T-1}: {adapter.late_weight[0, T-1, 0]:.3f}")
    
    # Verify crossfade
    assert adapter.early_weight[0, 0, 0].item() > 0.9, "Early should dominate at start"
    assert adapter.late_weight[0, T-1, 0].item() > 0.9, "Late should dominate at end"
    
    print("  ✓ Local spectral adapter works")
    return True


def main():
    print("="*80)
    print("Advanced SPEC-TTA Improvements Test Suite")
    print("="*80)
    
    tests = [
        ("B. Tail Damping", test_tail_damping),
        ("C. Local Spectral Adapter", test_local_spectral_adapter),
        ("D. Output-Only Mode", test_output_only_criterion),
        ("E. Adaptive Loss Schedule", test_adaptive_loss_schedule),
        ("F. Safe Update Manager", test_safe_update_manager),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append((name, False))
    
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(success for _, success in results)
    print("\n" + ("="*80))
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for integration")
    else:
        print("✗ SOME TESTS FAILED - Fix before integration")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
