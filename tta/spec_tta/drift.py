# tta/spec_tta/drift.py
"""
Spectral drift detection for test-time adaptation.
PT-prefix-aware version that matches PETSA/TAFAS methodology.
"""
import torch
from .utils_pt import pt_prefix_length, _norm_fft_mag


@torch.no_grad()
def spectral_drift_pt_prefix(y_pred: torch.Tensor, y_true: torch.Tensor, mask_pt: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 distance of normalized FFT magnitudes on the CONTIGUOUS PT prefix.
    This matches PETSA's PT/T timing and avoids DC/zero-padding artifacts.
    
    **FIX for long horizons**: When full horizon is observed (M=T), use a sliding window
    to compute drift instead of averaging over the entire horizon, which dilutes the signal.
    
    Args:
        y_pred: [B, T, V] predicted time series
        y_true: [B, T, V] true/observed time series  
        mask_pt: [B, T, V] binary mask (1=observed, 0=unobserved)
        
    Returns:
        drift: scalar drift score (0 if prefix too short)
    """
    M = pt_prefix_length(mask_pt)
    T = y_pred.shape[1]
    
    if M < 4:  # too short for reliable FFT
        return torch.tensor(0.0, device=y_pred.device)
    
    # Adaptive drift calculation for long horizons (Improvement G: Horizon-Adaptive Drift)
    # When M >= 192 (long horizon or full observation), use a sliding window approach
    # to avoid diluting drift signal by averaging over too many timesteps
    if M >= 192:
        # Use a 96-timestep sliding window with stride 48
        # This captures local drift without averaging over the entire horizon
        window_size = 96
        stride = 48
        drifts = []
        
        for start in range(0, M - window_size + 1, stride):
            end = start + window_size
            pmag = _norm_fft_mag(y_pred[:, start:end, :])
            tmag = _norm_fft_mag(y_true[:, start:end, :])
            window_drift = (pmag - tmag).abs().mean()
            drifts.append(window_drift)
        
        # Return maximum drift across windows (most sensitive region)
        return torch.stack(drifts).max() if drifts else torch.tensor(0.0, device=y_pred.device)
    else:
        # Standard drift calculation for short horizons
        # Compute normalized FFT magnitudes on PT prefix only
        pmag = _norm_fft_mag(y_pred[:, :M, :])
        tmag = _norm_fft_mag(y_true[:, :M, :])
        
        # L1 distance averaged over frequencies and variables
        return (pmag - tmag).abs().mean()


# Keep old function for backward compatibility (fallback)
@torch.no_grad()
def spectral_drift_score(y_pred: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Legacy drift score function - now redirects to PT-prefix version.
    
    Args:
        y_pred, y_obs, mask: [B, T, V]
        
    Returns:
        drift: scalar drift score
    """
    return spectral_drift_pt_prefix(y_pred, y_obs, mask)
