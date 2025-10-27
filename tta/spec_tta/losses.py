# tta/spec_tta/losses.py
"""
Loss functions for SPEC-TTA test-time adaptation.
PT-prefix-aware versions that match PETSA/TAFAS methodology.
"""
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from utils.fft_compat import rfft_1d
from .utils_pt import pt_prefix_length


def huber_loss(pred, target, delta: float = 0.5, reduction: str = "mean"):
    """
    Standard Huber loss (for backward compatibility).
    """
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.clamp(abs_diff, max=delta)
    lin = abs_diff - quad
    loss = 0.5 * (quad ** 2) + delta * lin
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def huber_loss_masked(pred, target, mask, delta: float = 0.5):
    """
    Huber loss on OBSERVED entries only (PT-prefix-aware).
    
    Args:
        pred: [B, T, V] predictions
        target: [B, T, V] targets
        mask: [B, T, V] binary mask (1=observed, 0=unobserved)
        delta: Huber threshold
        
    Returns:
        loss: mean Huber loss over observed entries
    """
    diff = pred - target
    absd = diff.abs()
    quad = torch.clamp(absd, max=delta)
    lin = absd - quad
    loss = 0.5 * quad**2 + delta * lin
    
    # Weight by mask
    w = mask.float()
    return (loss * w).sum() / (w.sum() + 1e-8)

def frequency_l1_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Standard frequency L1 loss (for backward compatibility).
    L1 difference between rFFT magnitudes on full horizon.
    Args: pred, target: [B, T, V]
    """
    pr, pi = rfft_1d(pred, n=pred.size(1), dim=1)   # [B, F, V]
    tr, ti = rfft_1d(target, n=target.size(1), dim=1)
    pmag = torch.sqrt(pr ** 2 + pi ** 2 + 1e-8)
    tmag = torch.sqrt(tr ** 2 + ti ** 2 + 1e-8)
    return (pmag - tmag).abs().mean()


def frequency_l1_loss_pt_prefix(pred: torch.Tensor, target: torch.Tensor, mask_pt: torch.Tensor):
    """
    L1 difference between rFFT magnitudes on PT prefix [1:M] only.
    Avoids DC/zero-padding artifacts by computing on observed prefix.
    
    Args:
        pred: [B, T, V] predictions
        target: [B, T, V] targets
        mask_pt: [B, T, V] binary mask indicating observed PT prefix
        
    Returns:
        loss: mean L1 difference of FFT magnitudes on PT prefix
    """
    M = pt_prefix_length(mask_pt)
    
    if M < 4:  # too short for reliable FFT
        return pred.mean() * 0.0
    
    # Compute FFT on PT prefix only
    pr, pi = rfft_1d(pred[:, :M, :], n=M, dim=1)
    tr, ti = rfft_1d(target[:, :M, :], n=M, dim=1)
    
    pmag = torch.sqrt(pr**2 + pi**2 + 1e-8)
    tmag = torch.sqrt(tr**2 + ti**2 + 1e-8)
    
    return (pmag - tmag).abs().mean()

def _unfold_patches(x: torch.Tensor, patch_len: int) -> torch.Tensor:
    """
    x: [B, T, V]
    returns patches: [B, num_patches, patch_len, V]
    """
    B, T, V = x.shape
    if patch_len >= T:
        return x.unsqueeze(1)  # [B,1,T,V]
    num = T - patch_len + 1
    patches = []
    for s in range(num):
        patches.append(x[:, s:s+patch_len, :])
    return torch.stack(patches, dim=1)  # [B, num, patch_len, V]

def patchwise_structural_loss(pred: torch.Tensor, target: torch.Tensor, patch_len: int = 24):
    """
    Standard patchwise structural loss (for backward compatibility).
    Sum of losses on means, variances, and correlation.
    """
    Pp = _unfold_patches(pred, patch_len)
    Pt = _unfold_patches(target, patch_len)
    # means/vars over time dim (patch_len)
    mean_p = Pp.mean(dim=2)
    mean_t = Pt.mean(dim=2)
    var_p  = Pp.var(dim=2, unbiased=False)
    var_t  = Pt.var(dim=2, unbiased=False)

    # Pearson correlation per patch/variable
    Pp_z = Pp - mean_p.unsqueeze(2)
    Pt_z = Pt - mean_t.unsqueeze(2)
    cov = (Pp_z * Pt_z).mean(dim=2)
    std_p = torch.sqrt(var_p + 1e-8)
    std_t = torch.sqrt(var_t + 1e-8)
    corr = cov / (std_p * std_t + 1e-8)  # in [-1,1]

    l_mean = (mean_p - mean_t).abs().mean()
    l_var  = (var_p  - var_t).abs().mean()
    l_corr = (1.0 - corr).abs().mean()
    return l_mean + l_var + l_corr


def patchwise_structural_loss_pt_prefix(pred: torch.Tensor, target: torch.Tensor, 
                                         mask_pt: torch.Tensor, patch_len: int = 24):
    """
    Patchwise structural loss on PT prefix [1:M] only.
    Computes mean, variance, and correlation losses on observed prefix to avoid zero-padding artifacts.
    
    Args:
        pred: [B, T, V] predictions
        target: [B, T, V] targets
        mask_pt: [B, T, V] binary mask indicating observed PT prefix
        patch_len: length of patches for structural analysis
        
    Returns:
        loss: sum of mean, variance, and correlation losses on PT prefix
    """
    M = pt_prefix_length(mask_pt)
    
    if M < max(4, patch_len):  # too short for meaningful patches
        return pred.mean() * 0.0
    
    # Extract PT prefix and compute patches
    Pp = _unfold_patches(pred[:, :M, :], patch_len)
    Pt = _unfold_patches(target[:, :M, :], patch_len)
    
    # Compute statistics over patch dimension
    mean_p, mean_t = Pp.mean(dim=2), Pt.mean(dim=2)
    var_p, var_t = Pp.var(dim=2, unbiased=False), Pt.var(dim=2, unbiased=False)
    
    # Compute Pearson correlation
    Pp_z, Pt_z = Pp - mean_p.unsqueeze(2), Pt - mean_t.unsqueeze(2)
    cov = (Pp_z * Pt_z).mean(dim=2)
    std_p, std_t = torch.sqrt(var_p + 1e-8), torch.sqrt(var_t + 1e-8)
    corr = cov / (std_p * std_t + 1e-8)
    
    # Combine losses
    l_mean = (mean_p - mean_t).abs().mean()
    l_var = (var_p - var_t).abs().mean()
    l_corr = (1.0 - corr).abs().mean()
    
    return l_mean + l_var + l_corr

def horizon_consistency_loss(y_full: torch.Tensor, y_short: torch.Tensor):
    """
    Penalize disagreement between a dedicated short-horizon prediction and
    the first steps of a long-horizon prediction. No labels needed.
    Args:
        y_full:  [B, T_long, V]
        y_short: [B, T_short, V]
    """
    Tshort = y_short.size(1)
    return F.mse_loss(y_full[:, :Tshort, :], y_short)
