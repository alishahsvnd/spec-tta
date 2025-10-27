"""
Multi-Scale Spectral Adapter for Maximum Accuracy
Combines multiple architectural enhancements to beat PETSA decisively.

Key innovations:
1. Parallel multi-scale spectral adapters (low/mid/high frequency)
2. Per-variable low-rank transformations (PETSA-style)
3. Learned gating between frequency scales
4. Ensemble of trend models

This sacrifices some efficiency for maximum accuracy.
Target: 10K-30K params to match/exceed PETSA's capacity strategically.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_adapter import SpectralAdapter, TrendHead
from utils.fft_compat import rfft_1d, irfft_1d


class MultiScaleSpectralAdapter(nn.Module):
    """
    Multi-scale spectral adaptation with parallel frequency bands.
    
    Architecture:
      1. Low-frequency adapter (trend, seasonality)
      2. Mid-frequency adapter (harmonics)
      3. High-frequency adapter (noise, fine details)
      4. Learned gating to combine outputs
    
    This increases capacity by having specialized adapters for each scale.
    """
    def __init__(
        self, 
        L: int, 
        V: int,
        k_low: int = 8,
        k_mid: int = 16, 
        k_high: int = 25,
        init_scale: float = 0.01,
        gating_dim: int = 32
    ):
        super().__init__()
        self.L = L
        self.V = V
        
        # FFT produces (L//2)+1 bins
        F = (L // 2) + 1
        
        # Define frequency ranges
        low_cutoff = max(1, int(F * 0.2))   # 0-20% of spectrum
        high_cutoff = int(F * 0.7)          # 70-100% of spectrum
        
        # Select top-k bins in each range (will be selected dynamically)
        # For now, use evenly spaced bins in each range
        low_bins = list(range(1, min(k_low + 1, low_cutoff)))
        mid_bins = list(range(low_cutoff, min(low_cutoff + k_mid, high_cutoff)))
        high_bins = list(range(high_cutoff, min(high_cutoff + k_high, F)))
        
        # Create specialized adapters for each scale
        self.low_adapter = SpectralAdapter(L, V, low_bins, init_scale, constrain_nyquist_dc_real=True)
        self.mid_adapter = SpectralAdapter(L, V, mid_bins, init_scale, constrain_nyquist_dc_real=True)
        self.high_adapter = SpectralAdapter(L, V, high_bins, init_scale, constrain_nyquist_dc_real=True)
        
        # Learned gating network to combine scales
        # Input: per-variable statistics, Output: 3 weights (low, mid, high)
        self.gating = nn.Sequential(
            nn.Linear(V * 3, gating_dim),  # 3 stats per variable: mean, std, energy
            nn.ReLU(),
            nn.Linear(gating_dim, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, V]
        Returns: [B, L, V] calibrated
        """
        B, L, V = x.shape
        
        # Apply each scale adapter
        x_low = self.low_adapter(x)
        x_mid = self.mid_adapter(x)
        x_high = self.high_adapter(x)
        
        # Compute input statistics for gating
        x_mean = x.mean(dim=1)  # [B, V]
        x_std = x.std(dim=1)    # [B, V]
        x_energy = (x ** 2).mean(dim=1)  # [B, V]
        stats = torch.cat([x_mean, x_std, x_energy], dim=-1)  # [B, V*3]
        
        # Compute gating weights
        weights = self.gating(stats)  # [B, 3]
        w_low = weights[:, 0:1].unsqueeze(-1)   # [B, 1, 1]
        w_mid = weights[:, 1:2].unsqueeze(-1)   # [B, 1, 1]
        w_high = weights[:, 2:3].unsqueeze(-1)  # [B, 1, 1]
        
        # Weighted combination
        x_cal = w_low * x_low + w_mid * x_mid + w_high * x_high
        
        return x_cal


class PerVariableLowRank(nn.Module):
    """
    Per-variable low-rank transformation (simpler than full PETSA).
    Each variable gets its own low-rank correction.
    """
    def __init__(self, T: int, V: int, rank: int, init_scale: float = 0.01):
        super().__init__()
        self.T = T
        self.V = V
        self.rank = rank
        
        # Per-variable low-rank matrices
        self.A = nn.Parameter(torch.randn(V, T, rank) * init_scale)
        self.B = nn.Parameter(torch.randn(V, rank, T) * init_scale)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, V]
        Returns: [B, T, V] with per-variable low-rank residual
        """
        B, T, V = x.shape
        
        # Apply per-variable transformation
        # For each variable v: x[:,:,v] + x[:,:,v] @ (A[v] @ B[v])
        residuals = []
        for v in range(V):
            x_v = x[:, :, v]  # [B, T]
            W_v = self.A[v] @ self.B[v]  # [T, T]
            res_v = x_v @ W_v  # [B, T]
            residuals.append(res_v.unsqueeze(-1))  # [B, T, 1]
        
        residual = torch.cat(residuals, dim=-1)  # [B, T, V]
        return x + residual


class EnsembleTrendHead(nn.Module):
    """
    Ensemble of multiple trend models:
    1. Linear: y += α*t + β
    2. Quadratic: y += α*t² + β*t + γ
    3. Exponential decay: y += α*exp(-β*t)
    
    Learns weights to combine them.
    """
    def __init__(self, T: int, V: int):
        super().__init__()
        self.T = T
        self.V = V
        
        # Linear trend
        self.alpha_lin = nn.Parameter(torch.zeros(V))
        self.beta_lin = nn.Parameter(torch.zeros(V))
        
        # Quadratic trend
        self.alpha_quad = nn.Parameter(torch.zeros(V))
        self.beta_quad = nn.Parameter(torch.zeros(V))
        self.gamma_quad = nn.Parameter(torch.zeros(V))
        
        # Exponential decay
        self.alpha_exp = nn.Parameter(torch.zeros(V))
        self.beta_exp = nn.Parameter(torch.ones(V) * 0.1)  # Small positive decay
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3, V) / 3.0)  # [3, V]
        
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B, T, V]
        """
        B, T, V = y.shape
        t = torch.arange(T, dtype=y.dtype, device=y.device).view(1, T, 1) / T  # Normalized time
        
        # Linear trend
        trend_lin = t * self.alpha_lin.view(1, 1, V) + self.beta_lin.view(1, 1, V)
        
        # Quadratic trend
        trend_quad = (
            (t ** 2) * self.alpha_quad.view(1, 1, V) +
            t * self.beta_quad.view(1, 1, V) +
            self.gamma_quad.view(1, 1, V)
        )
        
        # Exponential decay
        trend_exp = self.alpha_exp.view(1, 1, V) * torch.exp(-self.beta_exp.view(1, 1, V) * t)
        
        # Weighted combination (softmax over models)
        weights = F.softmax(self.ensemble_weights, dim=0)  # [3, V]
        w_lin = weights[0].view(1, 1, V)
        w_quad = weights[1].view(1, 1, V)
        w_exp = weights[2].view(1, 1, V)
        
        trend = w_lin * trend_lin + w_quad * trend_quad + w_exp * trend_exp
        
        return y + trend


class HighCapacitySpectralAdapter(nn.Module):
    """
    High-capacity spectral TTA combining all enhancements.
    
    Architecture:
      1. Multi-scale spectral adapter (low/mid/high frequencies)
      2. Per-variable low-rank transformations
      3. Ensemble trend model
      4. Residual connections throughout
    
    Target: 10K-30K params for maximum accuracy (competitive with PETSA).
    """
    def __init__(
        self,
        L: int,
        V: int,
        k_low: int = 8,
        k_mid: int = 16,
        k_high: int = 25,
        rank: int = 16,
        gating_dim: int = 64,
        init_scale: float = 0.01
    ):
        super().__init__()
        self.L = L
        self.V = V
        
        # Multi-scale spectral adapter
        self.spectral = MultiScaleSpectralAdapter(
            L, V, k_low, k_mid, k_high, init_scale, gating_dim
        )
        
        # Per-variable low-rank transformations (applied to spectral output)
        self.lowrank = PerVariableLowRank(L, V, rank, init_scale)
        
        # Ensemble trend model
        self.trend = EnsembleTrendHead(L, V)
        
    def forward(self, x: torch.Tensor, apply_trend: bool = True) -> torch.Tensor:
        """
        x: [B, L, V]
        Returns: [B, L, V] fully calibrated
        """
        # Multi-scale spectral adaptation
        x_spectral = self.spectral(x)
        
        # Per-variable low-rank correction
        x_lowrank = self.lowrank(x_spectral)
        
        # Trend correction
        if apply_trend:
            x_final = self.trend(x_lowrank)
        else:
            x_final = x_lowrank
            
        return x_final
    
    def count_parameters(self):
        """Count and display parameters by component."""
        total = 0
        print("High-Capacity Spectral Adapter Parameters:")
        print("=" * 60)
        
        # Spectral adapter
        spectral_params = sum(p.numel() for p in self.spectral.parameters())
        print(f"Multi-scale spectral adapter: {spectral_params:,}")
        total += spectral_params
        
        # Low-rank layers
        lr_params = sum(p.numel() for p in self.lowrank.parameters())
        print(f"Per-variable low-rank layer: {lr_params:,}")
        total += lr_params
        
        # Trend
        trend_params = sum(p.numel() for p in self.trend.parameters())
        print(f"Ensemble trend model: {trend_params:,}")
        total += trend_params
        
        print("=" * 60)
        print(f"TOTAL: {total:,}")
        print("=" * 60)
        
        return total
