# tta/spec_tta/drift_tuned.py
"""
Alternative drift detection with tunable window sizes for experimentation.
"""
import torch
from .utils_pt import pt_prefix_length, _norm_fft_mag


@torch.no_grad()
def spectral_drift_pt_prefix_tuned(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    mask_pt: torch.Tensor,
    window_size: int = 96,
    stride: int = 48
) -> torch.Tensor:
    """
    Tunable version of drift detection for experimentation.
    
    Args:
        y_pred: [B, T, V] predicted time series
        y_true: [B, T, V] true/observed time series  
        mask_pt: [B, T, V] binary mask (1=observed, 0=unobserved)
        window_size: Size of sliding window for long horizons
        stride: Stride for sliding window
        
    Returns:
        drift: scalar drift score (0 if prefix too short)
    """
    M = pt_prefix_length(mask_pt)
    T = y_pred.shape[1]
    
    if M < 4:  # too short for reliable FFT
        return torch.tensor(0.0, device=y_pred.device)
    
    # Adaptive drift calculation for long horizons
    if M >= 192:
        # Use sliding window approach
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
        pmag = _norm_fft_mag(y_pred[:, :M, :])
        tmag = _norm_fft_mag(y_true[:, :M, :])
        
        # L1 distance averaged over frequencies and variables
        return (pmag - tmag).abs().mean()
