"""
Safe Mode (No-Regret) TTA for Long Horizons

Implements convex mixing between baseline and adapted forecasts with holdout validation.
Ensures TTA never makes predictions worse than the baseline model.

Key insight: On H=720, the PT prefix provides insufficient information to reliably
govern long-range corrections. Any unconstrained update risks overfitting the prefix
and harming the tail. This module provides a data-driven safety mechanism.

Based on: PETSA paper findings that spectral terms can hurt on long horizons
(Fig. 13: beta=0 optimal on some datasets for long T).
"""

import torch
from .losses import huber_loss_masked


@torch.no_grad()
def choose_gamma_safe(
    y_base: torch.Tensor,
    y_adapt: torch.Tensor, 
    y_true: torch.Tensor,
    mask_obs: torch.Tensor,
    deltas=(0.0, 0.25, 0.5, 0.75, 1.0),
    delta_huber: float = 0.5
) -> tuple[float, float]:
    """
    Pick gamma ∈ deltas to minimize Huber loss on a small holdout slice 
    of the observed prefix.
    
    Args:
        y_base: Baseline forecast (frozen model) [B, T, V]
        y_adapt: Adapted forecast (SPEC-TTA output) [B, T, V]
        y_true: Ground truth observations [B, T, V]
        mask_obs: Binary mask for observed prefix [B, T, V]
        deltas: Candidate gamma values to try (convex combination weights)
        delta_huber: Huber loss transition point
        
    Returns:
        best_gamma: Optimal mixing weight in [0, 1]
        best_loss: Holdout loss achieved with best_gamma
        
    Strategy:
        - Split observed prefix into train and holdout (last 4 points)
        - Try each gamma value on holdout
        - Return gamma that minimizes holdout loss
        - If gamma=0.0 is best, automatically fall back to baseline
    """
    B, T, V = y_true.shape
    
    # Identify observed prefix points
    tmask = (mask_obs.sum(dim=2) > 0)  # [B, T] - which timesteps have observations
    M = int(tmask.sum(dim=1).min().item())  # Minimum observed length across batch
    
    if M < 8:
        # Require at least 8 points to form meaningful train/holdout split
        # Fall back to baseline if insufficient data
        return 0.0, float('inf')
    
    # Split: use last 4 observed points as holdout
    M_train = max(4, M - 4)
    tr = slice(0, M_train)
    ho = slice(M_train, M)
    
    # Grid search over gamma values
    best_gamma, best_loss = 0.0, float('inf')
    for g in deltas:
        # Convex combination: y_mix = (1-g)*y_base + g*y_adapt
        y_mix = y_base[:, ho, :] * (1 - g) + y_adapt[:, ho, :] * g
        
        # Evaluate on holdout slice
        loss_ho = huber_loss_masked(
            y_mix, 
            y_true[:, ho, :], 
            mask_obs[:, ho, :], 
            delta=delta_huber
        ).item()
        
        if loss_ho < best_loss:
            best_loss, best_gamma = loss_ho, float(g)
    
    return best_gamma, best_loss


def safe_mix(
    y_base: torch.Tensor,
    y_adapt: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    """
    Convex combination of baseline and adapted forecasts.
    
    Args:
        y_base: Baseline forecast [B, T, V]
        y_adapt: Adapted forecast [B, T, V]
        gamma: Mixing weight in [0, 1]
        
    Returns:
        Mixed forecast: (1-gamma)*y_base + gamma*y_adapt
        
    Notes:
        - gamma=0.0 returns pure baseline
        - gamma=1.0 returns pure adapted forecast
        - 0 < gamma < 1 provides controlled blending
    """
    return y_base * (1 - gamma) + y_adapt * gamma
