# tta/spec_tta/improvements.py
"""
Enhanced features for SPEC-TTA
Implements top priority improvements from SPEC_TTA_IMPROVEMENTS.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from utils.fft_compat import rfft_1d


class AdaptiveDriftManager:
    """
    Dynamically adjusts drift threshold based on historical statistics.
    Prevents both over-adaptation and under-adaptation.
    """
    def __init__(
        self, 
        initial_threshold: float = 0.01,
        window_size: int = 100,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.0001,
        max_threshold: float = 0.1
    ):
        self.threshold = initial_threshold
        self.drift_history = deque(maxlen=window_size)
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Statistics
        self.total_samples = 0
        self.adaptations_triggered = 0
    
    def update(self, current_drift: float) -> bool:
        """
        Update drift history and determine if adaptation should occur.
        Returns True if adaptation should be triggered.
        """
        self.drift_history.append(current_drift)
        self.total_samples += 1
        
        # Compute adaptive threshold after warmup
        if len(self.drift_history) >= 10:
            recent_drifts = list(self.drift_history)
            mean_drift = np.mean(recent_drifts)
            std_drift = np.std(recent_drifts)
            
            # Exponential moving average update
            new_threshold = mean_drift + 0.5 * std_drift
            self.threshold = (1 - self.adaptation_rate) * self.threshold + \
                           self.adaptation_rate * new_threshold
            
            # Clip to valid range
            self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)
        
        # Decision: adapt if current drift exceeds threshold
        should_adapt = current_drift > self.threshold
        if should_adapt:
            self.adaptations_triggered += 1
        
        return should_adapt
    
    def get_stats(self) -> Dict[str, float]:
        """Return statistics for logging."""
        return {
            'threshold': self.threshold,
            'adaptation_rate': self.adaptations_triggered / max(self.total_samples, 1),
            'mean_drift': np.mean(list(self.drift_history)) if len(self.drift_history) > 0 else 0.0,
            'std_drift': np.std(list(self.drift_history)) if len(self.drift_history) > 0 else 0.0
        }


class MultiScaleFrequencySelector:
    """
    Selects frequency bins at multiple scales:
    - Low frequencies: long-term trends
    - Mid frequencies: daily/weekly patterns
    - High frequencies: short-term variations
    """
    def __init__(
        self,
        k_low: int = 4,
        k_mid: int = 8,
        k_high: int = 4,
        low_cutoff: float = 0.2,
        high_cutoff: float = 0.8
    ):
        self.k_low = k_low
        self.k_mid = k_mid
        self.k_high = k_high
        self.low_cutoff = low_cutoff  # 0-20% of spectrum
        self.high_cutoff = high_cutoff  # 80-100% of spectrum
    
    def select_bins(self, X: torch.Tensor) -> Dict[str, List[int]]:
        """
        X: [B, L, V] - input time series
        Returns dict with 'low', 'mid', 'high', 'all' frequency bins
        """
        B, L, V = X.shape
        
        # Compute FFT energy
        fr, fi = rfft_1d(X, n=L, dim=1)
        energy = (fr ** 2 + fi ** 2).sum(dim=(0, 2))  # [F]
        
        total_bins = len(energy)
        low_end = max(1, int(total_bins * self.low_cutoff))
        high_start = int(total_bins * self.high_cutoff)
        
        # Select top-K in each range
        bins_low = self._top_k_in_range(energy, 0, low_end, self.k_low)
        bins_mid = self._top_k_in_range(energy, low_end, high_start, self.k_mid)
        bins_high = self._top_k_in_range(energy, high_start, total_bins, self.k_high)
        
        all_bins = sorted(set(bins_low + bins_mid + bins_high))
        
        return {
            'low': bins_low,
            'mid': bins_mid,
            'high': bins_high,
            'all': all_bins,
            'energy_low': energy[:low_end].sum().item(),
            'energy_mid': energy[low_end:high_start].sum().item(),
            'energy_high': energy[high_start:].sum().item()
        }
    
    def _top_k_in_range(
        self, 
        energy: torch.Tensor, 
        start: int, 
        end: int, 
        k: int
    ) -> List[int]:
        """Select top-K energy bins in specified range."""
        if end <= start:
            return []
        
        range_energy = energy[start:end]
        k_actual = min(k, len(range_energy))
        
        if k_actual == 0:
            return []
        
        _, indices = torch.topk(range_energy, k_actual, largest=True)
        bins = (indices + start).cpu().tolist()
        return sorted(bins)


class OnlineLRScheduler:
    """
    Adaptive learning rate scheduling for online adaptation.
    Uses cosine annealing with warm restarts and loss-based resets.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_min: float = 1e-5,
        lr_max: float = 1e-3,
        cycle_length: int = 100,
        restart_factor: float = 1.5
    ):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.cycle_length = cycle_length
        self.restart_factor = restart_factor
        
        self.step_count = 0
        self.prev_loss = None
        self.loss_history = deque(maxlen=10)
    
    def step(self, loss: Optional[float] = None):
        """
        Update learning rate based on step count and loss.
        """
        # Cosine annealing within cycle
        cycle_pos = self.step_count % self.cycle_length
        progress = cycle_pos / self.cycle_length
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
             (1 + np.cos(np.pi * progress))
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Check for restart condition
        if loss is not None:
            self.loss_history.append(loss)
            
            # Restart if loss increased significantly
            if self.prev_loss is not None:
                if loss > self.restart_factor * self.prev_loss:
                    self.step_count = 0  # Restart cycle
                    # Optionally reduce max LR
                    self.lr_max *= 0.9
            
            self.prev_loss = loss
        
        self.step_count += 1
        
        return lr
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class ReplayBuffer:
    """
    Maintains a buffer of recent samples for pseudo-rehearsal.
    Helps prevent catastrophic forgetting in online adaptation.
    """
    def __init__(
        self,
        capacity: int = 500,
        batch_size: int = 32,
        device: str = 'cuda'
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.buffer = []
        self.position = 0
    
    def add(
        self, 
        X: torch.Tensor,
        Y_hat: torch.Tensor,
        Y_pt: torch.Tensor,
        mask_pt: torch.Tensor
    ):
        """Add a sample to the buffer (circular queue)."""
        sample = {
            'X': X.detach().cpu(),
            'Y_hat': Y_hat.detach().cpu(),
            'Y_pt': Y_pt.detach().cpu(),
            'mask_pt': mask_pt.detach().cpu()
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.position] = sample
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: Optional[int] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample a random batch from the buffer.
        Returns None if buffer doesn't have enough samples.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_samples = [self.buffer[i] for i in indices]
        
        # Stack into batch tensors
        return {
            'X': torch.cat([s['X'] for s in batch_samples]).to(self.device),
            'Y_hat': torch.cat([s['Y_hat'] for s in batch_samples]).to(self.device),
            'Y_pt': torch.cat([s['Y_pt'] for s in batch_samples]).to(self.device),
            'mask_pt': torch.cat([s['mask_pt'] for s in batch_samples]).to(self.device)
        }
    
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for replay."""
        return len(self.buffer) >= self.batch_size
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0


class UncertaintyEstimator:
    """
    Estimates prediction uncertainty using Monte Carlo Dropout.
    Can be used to weight adaptation loss and provide confidence intervals.
    """
    def __init__(
        self,
        n_samples: int = 10,
        dropout_p: float = 0.1
    ):
        self.n_samples = n_samples
        self.dropout_p = dropout_p
    
    def add_dropout(self, module: nn.Module):
        """Add dropout layers to a module for uncertainty estimation."""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Add dropout before linear layers
                setattr(module, f'{name}_dropout', nn.Dropout(self.dropout_p))
    
    def estimate_uncertainty(
        self,
        model: nn.Module,
        X: torch.Tensor,
        forward_fn: callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate prediction uncertainty using MC Dropout.
        
        Returns:
            mean_pred: [B, T, V] - mean prediction
            uncertainty: [B, T, V] - standard deviation (uncertainty)
        """
        model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                Y = forward_fn(X)
                predictions.append(Y)
        
        model.eval()  # Disable dropout
        
        predictions = torch.stack(predictions)  # [n_samples, B, T, V]
        mean_pred = predictions.mean(dim=0)  # [B, T, V]
        uncertainty = predictions.std(dim=0)  # [B, T, V]
        
        return mean_pred, uncertainty
    
    def uncertainty_weighted_loss(
        self,
        Y_pred: torch.Tensor,
        Y_true: torch.Tensor,
        uncertainty: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute loss weighted by inverse uncertainty.
        More certain predictions contribute more to loss.
        """
        # Inverse uncertainty weights
        weights = 1.0 / (uncertainty + 1e-6)
        
        # Normalize weights
        if reduction == 'mean':
            weights = weights / (weights.sum() + 1e-8)
        
        # Weighted MSE
        loss = weights * (Y_pred - Y_true) ** 2
        
        if reduction == 'mean':
            return loss.sum()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ChannelWiseGating(nn.Module):
    """
    Learn per-variable/channel gating for adaptive weighting.
    Some variables may need more/less adaptation than others.
    """
    def __init__(self, n_vars: int, learnable: bool = True):
        super().__init__()
        self.n_vars = n_vars
        
        if learnable:
            self.gates = nn.Parameter(torch.ones(n_vars))
        else:
            self.register_buffer('gates', torch.ones(n_vars))
    
    def forward(self, X: torch.Tensor, X_adapted: torch.Tensor) -> torch.Tensor:
        """
        X: [B, T, V] - original input
        X_adapted: [B, T, V] - adapted input
        
        Returns: [B, T, V] - gated mixture
        """
        gates = torch.sigmoid(self.gates)  # [V]
        gates = gates.view(1, 1, -1)  # [1, 1, V]
        
        # Linear interpolation: gate=1 => fully adapted, gate=0 => original
        output = gates * X_adapted + (1 - gates) * X
        
        return output
    
    def get_gate_values(self) -> torch.Tensor:
        """Get current gate values (after sigmoid)."""
        return torch.sigmoid(self.gates)


class EnhancedSpecTTAConfig:
    """
    Extended configuration with enhancement options.
    """
    def __init__(
        self,
        # Base config (same as SpecTTAConfig)
        L: int, T: int, V: int,
        k_bins: int = 16,
        patch_len: int = 24,
        huber_delta: float = 0.5,
        beta_freq: float = 0.05,
        lambda_pw: float = 1.0,
        lambda_prox: float = 1e-4,
        lambda_hc: float = 0.1,
        drift_threshold: float = 0.01,
        lr: float = 1e-3,
        grad_clip: float = 1.0,
        device: str = "cuda",
        reselection_every: int = 0,
        # Enhancement options
        use_adaptive_threshold: bool = True,
        use_multiscale_freq: bool = False,
        use_lr_scheduling: bool = True,
        use_replay_buffer: bool = False,
        use_uncertainty: bool = False,
        use_channel_gating: bool = False,
        replay_buffer_size: int = 500,
        lr_min: float = 1e-5,
        lr_max: float = 1e-3,
        lr_cycle_length: int = 100
    ):
        # Base parameters
        self.L, self.T, self.V = L, T, V
        self.k_bins = k_bins
        self.patch_len = patch_len
        self.huber_delta = huber_delta
        self.beta_freq = beta_freq
        self.lambda_pw = lambda_pw
        self.lambda_prox = lambda_prox
        self.lambda_hc = lambda_hc
        self.drift_threshold = drift_threshold
        self.lr = lr
        self.grad_clip = grad_clip
        self.device = device
        self.reselection_every = reselection_every
        
        # Enhancement flags
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_multiscale_freq = use_multiscale_freq
        self.use_lr_scheduling = use_lr_scheduling
        self.use_replay_buffer = use_replay_buffer
        self.use_uncertainty = use_uncertainty
        self.use_channel_gating = use_channel_gating
        
        # Enhancement parameters
        self.replay_buffer_size = replay_buffer_size
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_cycle_length = lr_cycle_length
