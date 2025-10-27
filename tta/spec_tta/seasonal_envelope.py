"""
Seasonal Envelope Adapter for Long Horizons (H>=720)

A constrained spectral adapter that corrects only known seasonal frequencies
using amplitude and phase adjustments, modulated by a piecewise-linear envelope.

Key motivation:
- Free-form spectral gains (K~100+) are unidentifiable from short PT prefixes at H=720
- PETSA paper shows spectral L1 loss (beta) can hurt on long horizons (Fig. 13: beta=0 optimal)
- Solution: Restrict to a few seasonal lines (daily, weekly) with learnable envelope

This drastically reduces hypothesis space so PT prefix is sufficient for estimation.
"""

import torch
import torch.nn as nn
from utils.fft_compat import rfft_1d, irfft_1d


def _seasonal_bins_from_training(L: int, T: int, sampling: str = "hourly") -> list[int]:
    """
    Identify a few canonical seasonal frequency bins for the given horizon.
    
    Args:
        L: Input sequence length (lookback)
        T: Forecast horizon
        sampling: Data frequency ("hourly", "15min", etc.)
        
    Returns:
        List of rFFT bin indices corresponding to known seasonal periods
        
    Heuristic:
        - For hourly data: daily (24h), weekly (168h) cycles
        - For 15min data: daily (96 steps), weekly (672 steps)
        - Bin index k corresponds to period T/k
        
    Example:
        T=720, hourly → daily period=24 → k=720/24=30
        T=720, hourly → weekly period=168 → k=720/168≈4
    """
    F = T // 2 + 1  # rFFT output size
    bins = set()
    
    if sampling == "hourly":
        # Canonical periods: daily (24), weekly (24*7=168)
        periods = [24, 24 * 7]
    elif sampling == "15min":
        # 15-min data: daily (4*24=96), weekly (4*24*7=672)
        periods = [96, 672]
    else:
        # Default: assume hourly
        periods = [24, 24 * 7]
    
    for period in periods:
        k = round(T / period)
        if 0 < k < F:
            bins.add(k)
    
    # Can add more domain-specific periods here
    # (e.g., monthly, seasonal cycles for weather/energy)
    
    return sorted(list(bins))


class SeasonalEnvelopeAdapter(nn.Module):
    """
    Output-side adapter that edits only a small set of seasonal frequency lines.
    
    For each seasonal bin k and variable v:
        - Amplitude multiplier: (1 + a_v,k) where a_v,k is learned
        - Phase shift: phi_v,k ∈ [-phi_max, phi_max] radians
        
    Additionally, applies a piecewise-linear temporal envelope over [0, T-1]
    to modulate the correction strength (prevents abrupt changes at tail).
    
    Architecture:
        X_freq[k] → (1 + a) * exp(i*phi) * X_freq[k]  (for k in seasonal_bins)
        X_time → X_time * envelope(t)
        
    This is much more constrained than free-form spectral gains, making it
    identifiable from short PT prefixes even at H=720.
    """
    
    def __init__(
        self,
        T: int,
        V: int,
        seasonal_bins: list[int],
        phi_max: float = 0.25,
        n_knots: int = 4
    ):
        """
        Args:
            T: Forecast horizon
            V: Number of variables
            seasonal_bins: List of rFFT bin indices to adapt (e.g., [4, 30] for weekly, daily)
            phi_max: Maximum phase shift in radians (will be squashed via tanh)
            n_knots: Number of control points for piecewise-linear envelope
        """
        super().__init__()
        self.T = T
        self.V = V
        self.phi_max = phi_max
        self.n_knots = n_knots
        
        # Register seasonal bins as buffer (non-learnable)
        self.register_buffer("k_bins", torch.tensor(seasonal_bins, dtype=torch.long))
        K = len(seasonal_bins)
        
        # Learnable parameters: amplitude and phase for each (variable, seasonal_bin) pair
        self.a = nn.Parameter(torch.zeros(V, K))      # Relative amplitude adjustment
        self.phi = nn.Parameter(torch.zeros(V, K))    # Phase shift (radians)
        
        # Envelope knots: piecewise-linear modulation over time
        # Start at 1.0 (no damping initially)
        self.env_knots = nn.Parameter(torch.ones(V, n_knots))
    
    def _compute_envelope(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Compute piecewise-linear envelope over T timesteps.
        
        Returns:
            envelope: [V, T] - per-variable temporal modulation
        """
        # Normalized time axis [0, 1]
        t = torch.linspace(0, 1, T, device=device)  # [T]
        
        # Knot positions evenly spaced in [0, 1]
        knot_pos = torch.linspace(0, 1, self.n_knots, device=device)  # [n_knots]
        
        # For each timestep, find adjacent knots and interpolate
        idx = torch.clamp((t * (self.n_knots - 1)).long(), 0, self.n_knots - 2)  # [T]
        frac = t * (self.n_knots - 1) - idx.float()  # [T]
        
        # Barycentric weights for linear interpolation
        w0 = 1 - frac  # Weight for left knot
        w1 = frac      # Weight for right knot
        
        # Compute envelope for each variable
        env = torch.zeros(self.V, T, device=device)
        for v in range(self.V):
            # Clamp knot values to [0.5, 1.5] to avoid extreme gain/damping
            knots = torch.clamp(self.env_knots[v], 0.5, 1.5)  # [n_knots]
            env_v = knots[idx] * w0 + knots[idx + 1] * w1     # [T]
            env[v] = env_v
        
        return env
    
    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Apply seasonal envelope correction to forecast.
        
        Args:
            y_hat: Baseline forecast [B, T, V]
            
        Returns:
            y_corrected: Seasonally-adjusted forecast [B, T, V]
        """
        B, T, V = y_hat.shape
        
        # FFT to frequency domain
        fr, fi = rfft_1d(y_hat, n=T, dim=1)  # [B, F, V]
        F = fr.size(1)
        
        # Compute amplitude and phase adjustments
        A = 1.0 + self.a                             # [V, K] - amplitude multiplier
        phi = torch.tanh(self.phi) * self.phi_max    # [V, K] - phase shift (bounded)
        cosP = torch.cos(phi)
        sinP = torch.sin(phi)
        
        # Apply complex multiplication: (fr + j*fi) * (A*cos(phi) + j*A*sin(phi))
        fr2 = fr.clone()
        fi2 = fi.clone()
        
        for j, k in enumerate(self.k_bins.tolist()):
            # Complex multiplication at bin k for all variables
            # (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
            # where c = A*cos(phi), d = A*sin(phi)
            for v in range(V):
                a_v = A[v, j]
                cos_v = cosP[v, j]
                sin_v = sinP[v, j]
                
                # Real part: fr*A*cos - fi*A*sin
                fr2[:, k, v] = fr[:, k, v] * (a_v * cos_v) - fi[:, k, v] * (a_v * sin_v)
                # Imaginary part: fr*A*sin + fi*A*cos
                fi2[:, k, v] = fr[:, k, v] * (a_v * sin_v) + fi[:, k, v] * (a_v * cos_v)
        
        # Inverse FFT back to time domain
        y_corr = irfft_1d(fr2, fi2, n=T, dim=1)  # [B, T, V]
        
        # Apply piecewise-linear envelope
        env = self._compute_envelope(T, y_hat.device)  # [V, T]
        y_out = y_corr * env.t().unsqueeze(0)          # [B, T, V]
        
        return y_out
