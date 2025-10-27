# utils/fft_compat.py
import torch

def _has_new_fft():
    return hasattr(torch, "fft") and hasattr(torch.fft, "rfft")

def rfft_1d(x, n=None, dim=1):
    """
    x: real tensor [B, L, V] by default (dim=1 is time)
    Returns (fr, fi) each [B, F, V] where F = n//2+1.
    """
    assert x.dim() == 3, "Expected [B, L, V]"
    if _has_new_fft():
        fx = torch.fft.rfft(x, n=n, dim=dim)
        return fx.real, fx.imag
    else:
        if n is None:
            n = x.size(dim)
        # torch.rfft expects last 'signal_ndim' dims to be the signal; move time to last
        if dim != x.dim() - 2:  # want [B, L, V] with time at dim=1; rfft uses last dim
            # We will permute to [B, V, L]
            x_perm = x.permute(0, 2, 1).contiguous()  # [B, V, L]
            fx = torch.rfft(x_perm, signal_ndim=1, normalized=False, onesided=True)  # [B, V, F, 2]
            fr, fi = fx[..., 0], fx[..., 1]  # [B, V, F]
            return fr.permute(0, 2, 1), fi.permute(0, 2, 1)  # [B, F, V]
        else:
            fx = torch.rfft(x, signal_ndim=1, normalized=False, onesided=True)  # [B, F, V, 2]
            return fx[..., 0], fx[..., 1]

def irfft_1d(fr, fi, n, dim=1):
    """
    fr, fi: [B, F, V]
    Returns x: [B, L=n, V]
    """
    if _has_new_fft():
        fx = torch.complex(fr, fi)
        return torch.fft.irfft(fx, n=n, dim=dim)
    else:
        # Build shape [B, F, V, 2] for irfft
        fx = torch.stack([fr, fi], dim=-1)  # [B, F, V, 2]
        # torch.irfft expects last 'signal_ndim' dims are the frequency domain along time
        # We'll permute to [B, V, F, 2] and irfft with signal_sizes=(n,)
        fxv = fx.permute(0, 2, 1, 3).contiguous()  # [B, V, F, 2]
        x_perm = torch.irfft(fxv, signal_ndim=1, normalized=False, onesided=True, signal_sizes=(n,))
        # x_perm: [B, V, L]
        return x_perm.permute(0, 2, 1).contiguous()  # [B, L, V]
