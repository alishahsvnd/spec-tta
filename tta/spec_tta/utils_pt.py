# tta/spec_tta/utils_pt.py
"""
Helper utilities for PT (Partial Target) prefix computation.
Matches PETSA/TAFAS methodology where PT arrives as a contiguous prefix.
"""
import torch
from utils.fft_compat import rfft_1d


def pt_prefix_length(mask_pt: torch.Tensor) -> int:
    """
    Return M = min observed-prefix length across the batch.
    Assumes PT arrives as a contiguous prefix (as in TAFAS/PETSA).
    
    Args:
        mask_pt: [B, T, V] (bool or {0,1}) where 1 indicates observed position
        
    Returns:
        M: minimum contiguous prefix length across batch
    """
    # Check which timesteps have ANY observed variables
    tmask = (mask_pt.sum(dim=2) > 0)  # [B, T]
    
    # Count contiguous prefix length per batch element
    M_per_b = tmask.long().sum(dim=1)  # [B]
    
    # Return minimum to ensure all batch elements have data
    M = int(M_per_b.min().item())
    return M


def _norm_fft_mag(x: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized rFFT magnitude along time dimension (dim=1).
    
    Args:
        x: [B, T, V] time series tensor
        
    Returns:
        mag: [B, F, V] normalized FFT magnitudes where F = T//2 + 1
    """
    B, T, V = x.shape
    
    # Compute real FFT
    fr, fi = rfft_1d(x, n=T, dim=1)  # [B, F, V]
    
    # Compute magnitude
    mag = torch.sqrt(fr**2 + fi**2 + 1e-8)
    
    # Normalize across frequencies to get pseudo-PDF
    mag = mag / (mag.sum(dim=1, keepdim=True) + 1e-8)
    
    return mag
