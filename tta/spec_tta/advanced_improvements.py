# tta/spec_tta/advanced_improvements.py
"""
Advanced SPEC-TTA improvements for long-horizon stability.
Ideas B-F from user request.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List


class TailDampingHead(nn.Module):
    """
    B. Tail damping: Apply exponential decay to far-end predictions.
    Prevents over-correction at distant timesteps.
    """
    def __init__(self, T: int, V: int, damping_start: float = 0.6, damping_strength: float = 0.3):
        super().__init__()
        self.T, self.V = T, V
        
        # Create damping schedule: 1.0 (full strength) -> (1-damping_strength) at end
        t = torch.arange(T, dtype=torch.float32)
        start_idx = int(T * damping_start)
        damping = torch.ones(T)
        
        if start_idx < T:
            # Linear decay from start_idx to T
            decay_region = t[start_idx:]
            decay_norm = (decay_region - start_idx) / (T - start_idx)
            damping[start_idx:] = 1.0 - damping_strength * decay_norm
        
        self.register_buffer("damping_schedule", damping.view(1, T, 1))
    
    def forward(self, y_correction: torch.Tensor, y_base: torch.Tensor) -> torch.Tensor:
        """
        Apply damped correction: y_out = y_base + damping * (y_correction - y_base)
        
        Args:
            y_correction: Corrected prediction [B, T, V]
            y_base: Base prediction (before trend/shift) [B, T, V]
        """
        correction = y_correction - y_base
        damped_correction = correction * self.damping_schedule
        return y_base + damped_correction


def get_adaptive_loss_weights(horizon: int) -> Dict[str, float]:
    """
    E. Adaptive loss schedule based on horizon length.
    Observation from PETSA: frequency loss can hurt on long horizons.
    """
    if horizon <= 96:
        return {
            'beta_freq': 0.05,
            'lambda_pw': 1.0,
            'lambda_prox': 1e-4,
            'lr': 1e-3
        }
    elif horizon <= 192:
        return {
            'beta_freq': 0.02,  # Reduce frequency loss
            'lambda_pw': 1.0,
            'lambda_prox': 5e-4,  # Slightly stronger regularization
            'lr': 8e-4
        }
    elif horizon <= 336:
        return {
            'beta_freq': 0.005,  # Minimal frequency loss
            'lambda_pw': 0.5,    # Reduce structural loss weight
            'lambda_prox': 1e-3,  # Stronger regularization
            'lr': 5e-4
        }
    else:  # H >= 720
        return {
            'beta_freq': 0.001,  # Near-zero frequency loss
            'lambda_pw': 0.3,
            'lambda_prox': 5e-3,  # Much stronger regularization
            'lr': 2e-4  # Lower learning rate
        }


def should_use_output_only(horizon: int) -> bool:
    """
    D. Output-only adaptation criterion.
    For long horizons, freeze input adapter to reduce compounding instability.
    """
    return horizon >= 240


class SafeUpdateManager:
    """
    F. Safe update mechanism with rollback and parameter norm capping.
    """
    def __init__(self, max_param_norm: float = 5.0, patience: int = 5):
        self.max_param_norm = max_param_norm
        self.patience = patience
        
        self.best_loss = float('inf')
        self.best_state = {}
        self.no_improve_count = 0
        self.update_count = 0
        
    def before_update(self, modules: dict):
        """Save current state before applying update."""
        self.best_state = {
            name: {pname: p.data.clone() for pname, p in module.named_parameters()}
            for name, module in modules.items()
        }
    
    def after_update(self, current_loss: float, modules: dict) -> Dict[str, any]:
        """
        Check update safety and potentially rollback.
        Returns dict with metrics about the update.
        """
        self.update_count += 1
        metrics = {
            'safe_update_applied': True,
            'params_clipped': False,
            'rollback_occurred': False
        }
        
        # Check if loss improved BEFORE clipping params
        should_rollback = False
        if current_loss < self.best_loss * 0.95:  # Require 5% improvement
            self.best_loss = current_loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            
            # Rollback if no improvement for too long
            if self.no_improve_count >= self.patience and self.best_state:
                should_rollback = True
        
        # Rollback if needed
        if should_rollback:
            for name, module in modules.items():
                if name in self.best_state:
                    for pname, p in module.named_parameters():
                        if pname in self.best_state[name]:
                            p.data.copy_(self.best_state[name][pname])
            metrics['rollback_occurred'] = True
            metrics['safe_update_applied'] = False
            self.no_improve_count = 0  # Reset after rollback
        else:
            # Only clip if not rolling back
            for name, module in modules.items():
                for pname, p in module.named_parameters():
                    if p.requires_grad:
                        norm = p.data.norm().item()
                        if norm > self.max_param_norm:
                            p.data.mul_(self.max_param_norm / (norm + 1e-8))
                            metrics['params_clipped'] = True
        
        return metrics


class LocalSpectralAdapter(nn.Module):
    """
    C. Local two-segment spectral adapter with cross-fade.
    Handles non-stationarity by using different gains for early/late horizon.
    """
    def __init__(self, T: int, V: int, k_bins: list, split_ratio: float = 0.5):
        super().__init__()
        self.T = T
        self.V = V
        self.k_bins = torch.tensor(k_bins) if isinstance(k_bins, list) else k_bins
        K = len(k_bins)
        
        # Split point
        self.split_t = int(T * split_ratio)
        self.crossfade_width = max(4, T // 20)  # 5% of horizon
        
        # Early segment gains (first half)
        self.g_early_real = nn.Parameter(torch.zeros(V, K))
        self.g_early_imag = nn.Parameter(torch.zeros(V, K))
        
        # Late segment gains (second half)
        self.g_late_real = nn.Parameter(torch.zeros(V, K))
        self.g_late_imag = nn.Parameter(torch.zeros(V, K))
        
        # Create smooth crossfade weights
        t = torch.arange(T, dtype=torch.float32)
        fade_start = self.split_t - self.crossfade_width // 2
        fade_end = self.split_t + self.crossfade_width // 2
        
        early_weight = torch.ones(T)
        fade_mask = (t >= fade_start) & (t < fade_end)
        if fade_mask.any():
            t_fade = t[fade_mask]
            fade_progress = (t_fade - fade_start) / (fade_end - fade_start)
            early_weight[fade_mask] = 1.0 - fade_progress
        early_weight[t >= fade_end] = 0.0
        
        self.register_buffer("early_weight", early_weight.view(1, T, 1))
        self.register_buffer("late_weight", (1.0 - early_weight).view(1, T, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local spectral adaptation with crossfade."""
        from utils.fft_compat import rfft_1d, irfft_1d
        
        B, T, V = x.shape
        
        # Transform to frequency
        xr, xi = rfft_1d(x, n=T, dim=1)  # [B, F, V]
        
        # Apply early gains on selected bins
        xr_bins = xr[:, self.k_bins, :]  # [B, K, V]
        xi_bins = xi[:, self.k_bins, :]
        
        # Early gains
        gr_e = self.g_early_real.permute(1, 0).unsqueeze(0)  # [1, K, V]
        gi_e = self.g_early_imag.permute(1, 0).unsqueeze(0)
        yr_early_bins = xr_bins * (1 + gr_e) - xi_bins * gi_e
        yi_early_bins = xi_bins * (1 + gr_e) + xr_bins * gi_e
        
        # Reconstruct with early gains
        yr_early = xr.clone()
        yi_early = xi.clone()
        yr_early[:, self.k_bins, :] = yr_early_bins
        yi_early[:, self.k_bins, :] = yi_early_bins
        y_early = irfft_1d(yr_early, yi_early, n=T, dim=1)
        
        # Late gains
        gr_l = self.g_late_real.permute(1, 0).unsqueeze(0)
        gi_l = self.g_late_imag.permute(1, 0).unsqueeze(0)
        yr_late_bins = xr_bins * (1 + gr_l) - xi_bins * gi_l
        yi_late_bins = xi_bins * (1 + gr_l) + xr_bins * gi_l
        
        # Reconstruct with late gains
        yr_late = xr.clone()
        yi_late = xi.clone()
        yr_late[:, self.k_bins, :] = yr_late_bins
        yi_late[:, self.k_bins, :] = yi_late_bins
        y_late = irfft_1d(yr_late, yi_late, n=T, dim=1)
        
        # Crossfade
        y_out = self.early_weight * y_early + self.late_weight * y_late
        
        return y_out
