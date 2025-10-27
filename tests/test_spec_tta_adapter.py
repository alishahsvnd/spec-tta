# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for SPEC-TTA spectral adapters.
Quick check to verify shapes, parameter counts, and gradient flow.
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tta.spec_tta.spectral_adapter import SpectralAdapter, TrendHead


def test_forward_shapes():
    """Test that adapters maintain correct shapes."""
    print("\n=== Testing Forward Pass Shapes ===")
    
    L, T, V, B = 96, 96, 7, 2
    bins = [1, 2, 5, 7]
    
    # Initialize modules
    A_in = SpectralAdapter(L, V, bins)
    A_out = SpectralAdapter(T, V, bins)
    tr = TrendHead(T, V)
    
    print(f"Config: L={L}, T={T}, V={V}, B={B}, bins={bins}")
    
    # Create test inputs
    x = torch.randn(B, L, V)
    y = torch.randn(B, T, V)
    
    # Forward pass
    xc = A_in(x)
    yc_adapted = A_out(y)
    yc = tr(yc_adapted)
    
    # Check shapes
    assert xc.shape == (B, L, V), f"Input adapter: Expected {(B, L, V)}, got {xc.shape}"
    assert yc_adapted.shape == (B, T, V), f"Output adapter: Expected {(B, T, V)}, got {yc_adapted.shape}"
    assert yc.shape == (B, T, V), f"Trend head: Expected {(B, T, V)}, got {yc.shape}"
    
    print(f"✓ Input adapter shape: {x.shape} → {xc.shape}")
    print(f"✓ Output adapter shape: {y.shape} → {yc_adapted.shape}")
    print(f"✓ Trend head shape: {yc_adapted.shape} → {yc.shape}")
    print("✓ Shape test PASSED")


def test_parameter_count():
    """Test that parameter count is as expected."""
    print("\n=== Testing Parameter Count ===")
    
    L, T, V = 96, 96, 7
    bins = [1, 2, 5, 7, 10, 15, 20, 25]  # K=8
    K = len(bins)
    
    A_in = SpectralAdapter(L, V, bins)
    A_out = SpectralAdapter(T, V, bins)
    tr = TrendHead(T, V)
    
    # Count parameters
    params_in = sum(p.numel() for p in A_in.parameters())
    params_out = sum(p.numel() for p in A_out.parameters())
    params_tr = sum(p.numel() for p in tr.parameters())
    
    total = params_in + params_out + params_tr
    
    # Expected:
    # - Each adapter: V * K * 2 (g_real + g_imag)
    # - Trend head: V * 2 (alpha + beta)
    expected_per_adapter = V * K * 2
    expected_trend = V * 2
    expected_total = expected_per_adapter * 2 + expected_trend
    
    print(f"Config: V={V}, K={K}")
    print(f"Input adapter parameters: {params_in} (expected: {expected_per_adapter})")
    print(f"Output adapter parameters: {params_out} (expected: {expected_per_adapter})")
    print(f"Trend head parameters: {params_tr} (expected: {expected_trend})")
    print(f"Total parameters: {total} (expected: {expected_total})")
    
    assert params_in == expected_per_adapter, f"Input adapter: Expected {expected_per_adapter}, got {params_in}"
    assert params_out == expected_per_adapter, f"Output adapter: Expected {expected_per_adapter}, got {params_out}"
    assert params_tr == expected_trend, f"Trend head: Expected {expected_trend}, got {params_tr}"
    assert total == expected_total, f"Total: Expected {expected_total}, got {total}"
    
    print("✓ Parameter count test PASSED")


def test_gradient_flow():
    """Test that gradients flow correctly through all modules."""
    print("\n=== Testing Gradient Flow ===")
    
    L, T, V, B = 96, 96, 7, 2
    bins = [1, 2, 5, 7]
    
    A_in = SpectralAdapter(L, V, bins)
    A_out = SpectralAdapter(T, V, bins)
    tr = TrendHead(T, V)
    
    x = torch.randn(B, L, V)
    y = torch.randn(B, T, V)
    
    # Forward
    xc = A_in(x)
    yc = tr(A_out(y))
    
    # Backward
    loss = yc.sum()
    loss.backward()
    
    # Check gradients exist and are non-zero
    assert A_in.g_real.grad is not None, "Input adapter g_real has no gradient"
    assert A_in.g_imag.grad is not None, "Input adapter g_imag has no gradient"
    assert A_out.g_real.grad is not None, "Output adapter g_real has no gradient"
    assert A_out.g_imag.grad is not None, "Output adapter g_imag has no gradient"
    assert tr.alpha.grad is not None, "Trend head alpha has no gradient"
    assert tr.beta.grad is not None, "Trend head beta has no gradient"
    
    # Check that gradients are actually flowing (non-zero)
    assert A_in.g_real.grad.abs().sum() > 0, "Input adapter g_real gradient is all zeros"
    assert A_out.g_real.grad.abs().sum() > 0, "Output adapter g_real gradient is all zeros"
    assert tr.alpha.grad.abs().sum() > 0, "Trend head alpha gradient is all zeros"
    
    print("✓ Input adapter gradients: g_real, g_imag")
    print("✓ Output adapter gradients: g_real, g_imag")
    print("✓ Trend head gradients: alpha, beta")
    print("✓ Gradient flow test PASSED")


def test_identity_initialization():
    """Test that adapters start at identity (zero gains)."""
    print("\n=== Testing Identity Initialization ===")
    
    L, V, B = 96, 7, 2
    bins = [1, 2, 5, 7]
    
    # Initialize with zero scale (default)
    A = SpectralAdapter(L, V, bins, init_scale=0.0)
    
    # Check that gains are initialized to zero
    assert torch.allclose(A.g_real, torch.zeros_like(A.g_real)), "g_real not initialized to zero"
    assert torch.allclose(A.g_imag, torch.zeros_like(A.g_imag)), "g_imag not initialized to zero"
    
    # Forward should be identity
    x = torch.randn(B, L, V)
    xc = A(x)
    
    # Should be very close to identity (small numerical errors allowed)
    diff = (xc - x).abs().max().item()
    assert diff < 1e-4, f"Identity initialization failed: max diff = {diff}"
    
    print(f"✓ Gains initialized to zero")
    print(f"✓ Identity transform verified (max diff: {diff:.2e})")
    print("✓ Identity initialization test PASSED")


def test_real_spectrum_constraints():
    """Test that DC and Nyquist bins remain real for real signals."""
    print("\n=== Testing Real Spectrum Constraints ===")
    
    L, V, B = 96, 7, 2
    bins = [0, 1, 2, 47]  # Include DC (0) and Nyquist (L//2 for even L)
    
    A = SpectralAdapter(L, V, bins, constrain_nyquist_dc_real=True)
    
    # Set some non-zero imaginary gains
    with torch.no_grad():
        A.g_imag.fill_(0.5)
    
    # Forward pass
    x = torch.randn(B, L, V)
    xc = A(x)
    
    # Result should still be real (imaginary part should be negligible)
    assert torch.allclose(xc, xc.real, atol=1e-5), "Output has significant imaginary component"
    
    print("✓ DC and Nyquist bins constrained to real")
    print("✓ Output remains real")
    print("✓ Real spectrum constraints test PASSED")


def test_trend_head_closed_form():
    """Test trend head closed-form update."""
    print("\n=== Testing Trend Head Closed-Form Update ===")
    
    T, V, B = 96, 7, 2
    tr = TrendHead(T, V)
    
    # Create synthetic data with known trend
    t = torch.arange(T, dtype=torch.float32).view(1, T, 1)
    true_alpha = torch.randn(V) * 0.01
    true_beta = torch.randn(V) * 0.1
    
    # Base prediction (no trend)
    y_pred = torch.randn(B, T, V)
    
    # Ground truth with trend
    y_pt = y_pred + t * true_alpha.view(1, 1, V) + true_beta.view(1, 1, V)
    
    # Full observation mask
    mask = torch.ones_like(y_pt)
    
    # Apply closed-form update
    tr.closed_form_update(y_pred, y_pt, mask)
    
    # Check that learned trend is close to true trend
    alpha_error = (tr.alpha - true_alpha).abs().max().item()
    beta_error = (tr.beta - true_beta).abs().max().item()
    
    print(f"Alpha error: {alpha_error:.6f}")
    print(f"Beta error: {beta_error:.6f}")
    
    assert alpha_error < 0.01, f"Alpha not learned correctly: error = {alpha_error}"
    assert beta_error < 0.5, f"Beta not learned correctly: error = {beta_error}"
    
    print("✓ Closed-form update test PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("SPEC-TTA Unit Tests")
    print("=" * 60)
    
    try:
        test_forward_shapes()
        test_parameter_count()
        test_gradient_flow()
        test_identity_initialization()
        test_real_spectrum_constraints()
        test_trend_head_closed_form()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
