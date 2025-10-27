# tta/spec_tta/spectral_adapter.py
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fft_compat import rfft_1d, irfft_1d

class SpectralAdapter(nn.Module):
    """
    Parameter-efficient frequency-selective adapter.
    - Keeps a sparse set of complex gains on selected Fourier bins per variable.
    - Multiplicative correction: X_f * (1 + ΔG), applied in frequency domain.

    Args:
        L: lookback length (time steps)
        V: number of variables
        k_bins: list of selected frequency bin indices (length K)
        init_scale: std for small random init of gains (0 -> identity)
        constrain_nyquist_dc_real: if True, force imag gains for DC/Nyquist to 0
    """
    def __init__(
        self, L: int, V: int, k_bins: List[int],
        init_scale: float = 0.0, constrain_nyquist_dc_real: bool = True
    ):
        super().__init__()
        self.L = L
        self.V = V
        self.register_buffer("k_bins", torch.tensor(k_bins, dtype=torch.long))
        K = len(k_bins)
        # Gains are real-valued parameters for real/imag parts per variable & selected bins
        g_init = torch.zeros(V, K)
        if init_scale > 0:
            g_init = torch.randn(V, K) * init_scale
        self.g_real = nn.Parameter(g_init.clone())  # [V, K]
        self.g_imag = nn.Parameter(g_init.clone())  # [V, K]
        self.constrain_nyquist_dc_real = constrain_nyquist_dc_real
        
        # Precompute which selected-bin indices correspond to DC/Nyquist
        # to enforce real-signal constraints (no imaginary component at DC/Nyquist)
        F = (L // 2) + 1  # Number of rFFT bins
        dc = 0
        nyq = (F - 1) if (L % 2 == 0) else -1  # Nyquist only exists if L is even
        special = []
        for j, kb in enumerate(k_bins):
            if kb == dc or (nyq >= 0 and kb == nyq):
                special.append(j)
        self.register_buffer("imag_freeze_idx", torch.tensor(special, dtype=torch.long))
        
        # Register gradient hook to zero gradients on imaginary gains for DC/Nyquist
        # This enforces strict real-signal constraints during backpropagation
        if len(special) > 0 and constrain_nyquist_dc_real:
            def _zero_grad_dc_nyq(grad):
                """Zero out gradients for imaginary gains at DC/Nyquist bins."""
                grad = grad.clone()
                grad[:, self.imag_freeze_idx] = 0.0
                return grad
            self.g_imag.register_hook(_zero_grad_dc_nyq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, V], real
        """
        B, L, V = x.shape
        assert L == self.L and V == self.V
        fr, fi = rfft_1d(x, n=L, dim=1)  # [B, F, V]
        Fbins = fr.size(1)
        K = self.k_bins.numel()

        # Build full real/imag gain tensors [V, F]
        g_r_full = x.new_zeros(V, Fbins)
        g_i_full = x.new_zeros(V, Fbins)
        g_r_full[:, self.k_bins] = self.g_real  # [V, F]
        g_i_full[:, self.k_bins] = self.g_imag  # [V, F]

        if self.constrain_nyquist_dc_real:
            # DC bin = 0; Nyquist = Fbins-1 (only if L is even)
            g_i_full[:, 0] = 0.0
            if (self.L % 2 == 0):
                g_i_full[:, -1] = 0.0

        # Apply multiplier (1 + g_r + j g_i)
        # (fr + j fi) * ((1+g_r) + j g_i) = [fr*(1+g_r) - fi*g_i] + j [fi*(1+g_r) + fr*g_i]
        # Broadcast V across batch and frequency dims
        fr2 = fr * (1.0 + g_r_full.T)[None, :, :] - fi * (g_i_full.T)[None, :, :]
        fi2 = fi * (1.0 + g_r_full.T)[None, :, :] + fr * (g_i_full.T)[None, :, :]

        x_cal = irfft_1d(fr2, fi2, n=L, dim=1)  # [B, L, V]
        return x_cal


class TrendHead(nn.Module):
    """
    Simple per-variable affine drift across the forecast horizon.
    y += alpha_v * t + beta_v
    alpha, beta learnable; can also be updated in closed-form on partial labels.
    """
    def __init__(self, T: int, V: int):
        super().__init__()
        self.T, self.V = T, V
        self.alpha = nn.Parameter(torch.zeros(V))  # slope per variable
        self.beta  = nn.Parameter(torch.zeros(V))  # offset per variable

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, T, V]
        B, T, V = y.shape
        assert T == self.T and V == self.V
        t = y.new_tensor(torch.arange(T, dtype=y.dtype)).view(1, T, 1)
        y = y + t * self.alpha.view(1, 1, V) + self.beta.view(1, 1, V)
        return y

    @torch.no_grad()
    def closed_form_update(self, y_pred: torch.Tensor, y_pt: torch.Tensor, mask: torch.Tensor):
        """
        Closed-form OLS on partially observed horizon (legacy method).
        Args:
            y_pred: [B, T, V] current calibrated prediction (before trend)
            y_pt:   [B, T, V] tensor with observed values where mask==1, arbitrary elsewhere
            mask:   [B, T, V] 1 for observed positions in the horizon; 0 otherwise
        """
        B, T, V = y_pred.shape
        t = y_pred.new_tensor(torch.arange(T, dtype=y_pred.dtype)).view(1, T, 1)
        # flatten batch/time, do per-variable independent 2x2 systems
        for v in range(V):
            m = mask[..., v].reshape(-1)  # [B*T]
            if m.sum() < 2:
                continue
            tp = t.expand(B, T, 1)[..., 0].reshape(-1)[m == 1]  # [M]
            yp = y_pt[..., v].reshape(-1)[m == 1] - y_pred[..., v].reshape(-1)[m == 1]
            # Solve [t, 1] θ ≈ yp in least squares
            A = torch.stack([tp, torch.ones_like(tp)], dim=1)  # [M, 2]
            # θ = (A^T A)^{-1} A^T y
            ATA = A.T @ A
            ATy = A.T @ yp
            try:
                theta = torch.linalg.solve(ATA, ATy)
            except Exception:
                theta = torch.pinverse(ATA) @ ATy
            self.alpha.data[v] = theta[0]
            self.beta.data[v]  = theta[1]
    
    @torch.no_grad()
    def closed_form_update_prefix(self, y_cal: torch.Tensor, y_true: torch.Tensor, mask_pt: torch.Tensor):
        """
        Update alpha/beta using OLS on PT prefix [1:M] only.
        Faster and more stable than full-horizon OLS by focusing on contiguous observed prefix.
        
        Solves for each variable v:
            min_{α_v, β_v} Σ_{t=1}^M (y_cal_{t,v} + α_v*t + β_v - y_{t,v})²
        
        Solution:
            [α_v, β_v]^T = (A^T A)^{-1} A^T (y_{1:M,v} - y_cal_{1:M,v})
        where A = [t, 1] is the design matrix.
        
        Args:
            y_cal: [B, T, V] calibrated prediction (pre-trend, output of adapter_out)
            y_true: [B, T, V] ground truth targets
            mask_pt: [B, T, V] binary mask indicating observed PT prefix
        """
        from .utils_pt import pt_prefix_length
        
        B, T, V = y_cal.shape
        M = pt_prefix_length(mask_pt)
        
        if M < 2:
            return  # Need at least 2 points for line fitting
        
        # Build design matrix A = [t, 1] for prefix [0, 1, ..., M-1]
        t = torch.arange(M, device=y_cal.device, dtype=y_cal.dtype).view(M, 1)  # [M, 1]
        A = torch.cat([t, torch.ones_like(t)], dim=1)  # [M, 2]
        
        # Precompute (A^T A)^{-1} once for all variables
        ATA = A.T @ A  # [2, 2]
        ATA_inv = torch.pinverse(ATA)  # [2, 2], numerically stable
        
        # Solve per-variable OLS
        for v in range(V):
            # Extract prefix predictions and targets
            y_cal_prefix = y_cal[:, :M, v]  # [B, M]
            y_true_prefix = y_true[:, :M, v]  # [B, M]
            
            # Compute residual: r = y_true - y_cal (what trend needs to explain)
            r = y_true_prefix - y_cal_prefix  # [B, M]
            
            # Average residual across batch (assumes batch samples share same trend pattern)
            r_mean = r.mean(dim=0, keepdim=True).T  # [M, 1]
            
            # Solve: θ = (A^T A)^{-1} A^T r
            theta = ATA_inv @ (A.T @ r_mean)  # [2, 1]
            
            # Update parameters
            self.alpha.data[v] = theta[0].item()
            self.beta.data[v] = theta[1].item()
