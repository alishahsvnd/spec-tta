import torch
import torch.nn as nn
from utils.fft_compat import rfft_1d, irfft_1d


class TimeShiftHead(nn.Module):
    """
    Per-variable fractional time shift tau_v (in 'steps'), applied in frequency domain
    via a linear phase ramp: e^{j 2π k tau / T}. Parameterized with tanh for bounded range.
    """
    def __init__(self, T: int, V: int, tau_max: float = 2.0):
        super().__init__()
        self.T, self.V, self.tau_max = T, V, tau_max
        self.tau_raw = nn.Parameter(torch.zeros(V))  # in R, squashed by tanh to [-tau_max, tau_max]

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, T, V]
        B, T, V = y.shape
        assert T == self.T and V == self.V
        fr, fi = rfft_1d(y, n=T, dim=1)  # [B, F, V]
        F = fr.size(1)  # F = T//2+1
        k = torch.arange(F, device=y.device, dtype=fr.dtype).view(F, 1)  # [F,1]
        tau = torch.tanh(self.tau_raw) * self.tau_max  # [V]
        # phase ramp: exp(j*2π k tau / T)
        phase = 2.0 * torch.pi * k * tau.view(1, V) / float(T)  # [F, V]
        cosP, sinP = torch.cos(phase), torch.sin(phase)         # [F, V]
        fr2 = fr * cosP[None, :, :] - fi * sinP[None, :, :]
        fi2 = fr * sinP[None, :, :] + fi * cosP[None, :, :]
        y_shift = irfft_1d(fr2, fi2, n=T, dim=1)                # [B, T, V]
        return y_shift


class PolyTrendHead(nn.Module):
    """
    Adaptive trend head:
    - Short horizons (T<=96): Linear trend (b*t + c) - prevents overfitting
    - Long horizons (T>96): Quadratic trend (a*t^2 + b*t + c) - captures curvature
    """
    def __init__(self, T: int, V: int):
        super().__init__()
        self.T, self.V = T, V
        self.use_quadratic = T > 96  # Only use quadratic for longer horizons
        
        if self.use_quadratic:
            self.a = nn.Parameter(torch.zeros(V))
        self.b = nn.Parameter(torch.zeros(V))
        self.c = nn.Parameter(torch.zeros(V))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, T, V = y.shape
        t = torch.arange(T, device=y.device, dtype=y.dtype).view(1, T, 1)
        
        if self.use_quadratic:
            # Quadratic trend for long horizons
            return y + (self.a.view(1,1,V) * t * t) + (self.b.view(1,1,V) * t) + self.c.view(1,1,V)
        else:
            # Linear trend for short horizons (prevents overfitting)
            return y + (self.b.view(1,1,V) * t) + self.c.view(1,1,V)

    @torch.no_grad()
    def closed_form_update_prefix(self, y_cal: torch.Tensor, y_true: torch.Tensor, mask_pt: torch.Tensor):
        """
        Solve for optimal trend parameters using PT prefix.
        - Quadratic (T>96): min_{a,b,c} \sum (y_cal + a t^2 + b t + c - y_true)^2
        - Linear (T<=96): min_{b,c} \sum (y_cal + b t + c - y_true)^2
        """
        from tta.spec_tta.utils_pt import pt_prefix_length
        B, T, V = y_cal.shape
        M = pt_prefix_length(mask_pt)
        
        if self.use_quadratic and M < 3:  # need >= 3 points for quadratic
            return
        elif not self.use_quadratic and M < 2:  # need >= 2 points for linear
            return
            
        t = torch.arange(M, device=y_cal.device, dtype=y_cal.dtype)
        
        if self.use_quadratic:
            # Quadratic fit: [t^2, t, 1]
            A = torch.stack([t*t, t, torch.ones_like(t)], dim=1)  # [M, 3]
        else:
            # Linear fit: [t, 1]
            A = torch.stack([t, torch.ones_like(t)], dim=1)  # [M, 2]
        
        ATA = A.T @ A
        ATA_inv = torch.pinverse(ATA)
        
        # Compute batch-averaged residual
        r = (y_true[:, :M, :] - y_cal[:, :M, :]).mean(dim=0)  # [M, V]
        
        # Solve per variable
        At = A.T
        rhs = At @ r  # [2 or 3, V]
        theta = ATA_inv @ rhs  # [2 or 3, V]
        
        # Assign parameters
        if self.use_quadratic:
            self.a.data.copy_(theta[0])
            self.b.data.copy_(theta[1])
            self.c.data.copy_(theta[2])
        else:
            self.b.data.copy_(theta[0])
            self.c.data.copy_(theta[1])
