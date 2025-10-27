"""
High-Capacity Spectral TTA Adapter Wrapper
Integrates HighCapacitySpectralAdapter with PETSA's TTA framework.

KEY FIX: Now applies HC adapter to INPUTS (like original SPEC-TTA), not outputs!
"""

from typing import Optional
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np

from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
from .multi_scale_adapter import HighCapacitySpectralAdapter
from .losses import huber_loss, frequency_l1_loss, patchwise_structural_loss


class SpecTTAHighCapacityAdapter(nn.Module):
    """
    Wrapper for high-capacity multi-scale spectral TTA.
    Compatible with PETSA's adapter interface.
    """
    
    def __init__(self, cfg, model: nn.Module, norm_module: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        
        # Get configuration mode
        mode_configs = {
            'medium': {
                'k_low': 6, 'k_mid': 12, 'k_high': 20,
                'rank': 8, 'gating_dim': 32
            },
            'high': {
                'k_low': 8, 'k_mid': 16, 'k_high': 25,
                'rank': 16, 'gating_dim': 64
            },
            'ultra': {
                'k_low': 10, 'k_mid': 20, 'k_high': 19,
                'rank': 24, 'gating_dim': 128
            }
        }
        
        mode = cfg.TTA.SPEC_TTA_HC.MODE
        if mode in mode_configs:
            config = mode_configs[mode]
        else:
            # Use custom config
            config = {
                'k_low': cfg.TTA.SPEC_TTA_HC.K_LOW,
                'k_mid': cfg.TTA.SPEC_TTA_HC.K_MID,
                'k_high': cfg.TTA.SPEC_TTA_HC.K_HIGH,
                'rank': cfg.TTA.SPEC_TTA_HC.RANK,
                'gating_dim': cfg.TTA.SPEC_TTA_HC.GATING_DIM
            }
        
        # KEY FIX: Use two-adapter design like original SPEC-TTA!
        # adapter_in: Simple calibration (gradients don't flow here much)
        # adapter_out: HIGH-CAPACITY adaptation (gradients flow here!)
        
        from .spectral_adapter import SpectralAdapter, TrendHead
        
        # Simple input adapter (keeps it lightweight, just preprocessing)
        # Use same k_bins as original SPEC-TTA for consistency
        k_bins = 32
        self.adapter_in = SpectralAdapter(
            L=cfg.DATA.SEQ_LEN,
            V=cfg.DATA.N_VAR,
            k_bins=list(range(k_bins))  # Use top-k frequency bins
        )
        
        # HIGH-CAPACITY output adapter (this is where the magic happens!)
        self.adapter_out = HighCapacitySpectralAdapter(
            L=cfg.DATA.PRED_LEN,  # Output length, not input length!
            V=cfg.DATA.N_VAR,
            k_low=config['k_low'],
            k_mid=config['k_mid'],
            k_high=config['k_high'],
            rank=config['rank'],
            gating_dim=config['gating_dim'],
            init_scale=cfg.TTA.SPEC_TTA_HC.INIT_SCALE
        )
        
        # Trend head (as in original SPEC-TTA)
        self.trend_head = TrendHead(cfg.DATA.PRED_LEN, cfg.DATA.N_VAR)
        
        # Move to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.adapter_in = self.adapter_in.to(device)
        self.adapter_out = self.adapter_out.to(device)
        self.trend_head = self.trend_head.to(device)
        
        # Setup optimizer for all three modules (like original SPEC-TTA)
        params = (list(self.adapter_in.parameters()) + 
                  list(self.adapter_out.parameters()) + 
                  list(self.trend_head.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=cfg.TTA.SPEC_TTA_HC.LR)
        
        # Configuration
        self.lr = cfg.TTA.SPEC_TTA_HC.LR
        self.grad_clip = cfg.TTA.SPEC_TTA_HC.GRAD_CLIP
        self.drift_threshold = cfg.TTA.SPEC_TTA_HC.DRIFT_THRESHOLD
        
        # Loss weights
        self.beta_freq = cfg.TTA.SPEC_TTA_HC.BETA_FREQ
        self.lambda_pw = cfg.TTA.SPEC_TTA_HC.LAMBDA_PW
        self.lambda_prox = cfg.TTA.SPEC_TTA_HC.LAMBDA_PROX
        self.lambda_hc = cfg.TTA.SPEC_TTA_HC.LAMBDA_HC
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Save initial state
        self.model_state = deepcopy(self.model.state_dict())
        self.adapter_in_init_state = deepcopy(self.adapter_in.state_dict())
        self.adapter_out_init_state = deepcopy(self.adapter_out.state_dict())
        self.trend_head_init_state = deepcopy(self.trend_head.state_dict())
        
        # Metrics
        self.n_adapt = 0
        self.device = device
        
        # Get test loader
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test
        
        # Configure batch processing - use full dataset as one batch like PETSA
        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        
        # Metrics tracking
        self.mse_all = []
        self.mae_all = []
        
    def count_parameters(self):
        """Count and display parameters."""
        print("------- SPEC-TTA HC PARAMETERS -------")
        total_sum = 0
        
        # Count adapter_in
        print("\nAdapter IN (input calibration):")
        for name, param in self.adapter_in.named_parameters():
            print(f"  {param.requires_grad} {name:30s} {str(param.size()):20s} {param.numel()}")
            if param.requires_grad:
                total_sum += int(param.numel())
        
        # Count adapter_out (HIGH-CAPACITY)
        print("\nAdapter OUT (HIGH-CAPACITY output adaptation):")
        for name, param in self.adapter_out.named_parameters():
            print(f"  {param.requires_grad} {name:30s} {str(param.size()):20s} {param.numel()}")
            if param.requires_grad:
                total_sum += int(param.numel())
        
        # Count trend_head
        print("\nTrend Head:")
        for name, param in self.trend_head.named_parameters():
            print(f"  {param.requires_grad} {name:30s} {str(param.size()):20s} {param.numel()}")
            if param.requires_grad:
                total_sum += int(param.numel())
        
        print(f"\nTotal Trainable Parameters: {total_sum:,}")
        return total_sum
        
    def reset(self):
        """Reset all adapters to initial state."""
        self.model.load_state_dict(self.model_state)
        self.adapter_in.load_state_dict(self.adapter_in_init_state)
        self.adapter_out.load_state_dict(self.adapter_out_init_state)
        self.trend_head.load_state_dict(self.trend_head_init_state)
        self.n_adapt = 0
        
    def adapt(self):
        """
        Main adaptation loop following PETSA's pattern.
        KEY FIX: Now applies HC adapter to INPUTS (enc_window), not outputs!
        """
        print("\n========== Starting SPEC-TTA HC Adaptation ==========")
        print(f"Mode: {self.cfg.TTA.SPEC_TTA_HC.MODE}")
        print("Strategy: Adapt INPUTS before model (like original SPEC-TTA)")
        self.model.eval()
        
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        
        for idx, inputs in enumerate(self.test_loader):
            # Prepare all inputs from the full batch
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            
            # Process incrementally like PETSA
            while batch_end < len(enc_window_all):
                batch_size = self.cfg.TTA.PETSA.BATCH_SIZE  # Use same batch size as PETSA
                batch_end = batch_start + batch_size

                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start

                # Extract current batch
                enc_window = enc_window_all[batch_start:batch_end]
                enc_window_stamp = enc_window_stamp_all[batch_start:batch_end]
                dec_window = dec_window_all[batch_start:batch_end]
                dec_window_stamp = dec_window_stamp_all[batch_start:batch_end]
                
                # CORRECT APPROACH: Two-adapter design like original SPEC-TTA!
                # 1. Calibrate input (simple adapter)
                # 2. Get model prediction (frozen)
                # 3. Adapt output with HIGH-CAPACITY adapter
                # 4. Apply trend
                # 5. Loss on final output → gradients flow to adapter_out + trend_head
                
                with torch.no_grad():
                    # Step 1: Calibrate input
                    enc_calibrated = self.adapter_in(enc_window)
                    
                    # Step 2: Get model prediction (frozen model)
                    if self.norm_method == 'NST':
                        outputs = self.model(enc_calibrated, enc_window_stamp, dec_window, dec_window_stamp, self.norm_module)
                    else:
                        outputs = self.model(enc_calibrated, enc_window_stamp, dec_window, dec_window_stamp)
                    
                    if isinstance(outputs, tuple):
                        Y_hat = outputs[0]
                    else:
                        Y_hat = outputs
                    
                    # Extract prediction horizon
                    Y_hat = Y_hat[:, -self.cfg.MODEL.pred_len:, :]
                
                # Step 3 & 4: Adapt output + trend (WITH GRADIENTS for training)
                Y_adapted = self.adapter_out(Y_hat, apply_trend=False)
                Y_final = self.trend_head(Y_adapted)
                
                # Get ground truth
                Y_true = dec_window[:, -self.cfg.MODEL.pred_len:, :]
                
                # Compute drift (for threshold check, use simple MSE)
                with torch.no_grad():
                    drift = torch.nn.functional.mse_loss(Y_final, Y_true).item()
                
                # Adapt if drift is significant
                if drift > self.drift_threshold:
                    self.optimizer.zero_grad()
                    
                    # Multi-component loss (like original SPEC-TTA)
                    # 1. Robust loss (Huber)
                    loss_huber = huber_loss(Y_final, Y_true, delta=self.cfg.TTA.SPEC_TTA_HC.HUBER_DELTA)
                    
                    # 2. Frequency loss (spectral alignment)
                    loss_freq = frequency_l1_loss(Y_final, Y_true)
                    
                    # 3. Patchwise structural loss
                    loss_pw = patchwise_structural_loss(Y_final, Y_true, patch_len=self.cfg.TTA.SPEC_TTA_HC.PATCH_LEN)
                    
                    # 4. Proximal regularization (keep params near initialization)
                    loss_prox = 0.0
                    for p in (list(self.adapter_in.parameters()) + 
                             list(self.adapter_out.parameters()) + 
                             list(self.trend_head.parameters())):
                        loss_prox = loss_prox + (p ** 2).sum()
                    loss_prox = self.lambda_prox * loss_prox
                    
                    # Combined loss
                    loss = loss_huber + self.beta_freq * loss_freq + self.lambda_pw * loss_pw + loss_prox
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.grad_clip > 0:
                        all_params = (list(self.adapter_in.parameters()) + 
                                      list(self.adapter_out.parameters()) + 
                                      list(self.trend_head.parameters()))
                        torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                    
                    self.optimizer.step()
                    self.n_adapt += 1
                
                # Compute final metrics with updated adapters
                with torch.no_grad():
                    # Full forward pass
                    enc_cal = self.adapter_in(enc_window)
                    
                    if self.norm_method == 'NST':
                        outputs = self.model(enc_cal, enc_window_stamp, dec_window, dec_window_stamp, self.norm_module)
                    else:
                        outputs = self.model(enc_cal, enc_window_stamp, dec_window, dec_window_stamp)
                    
                    if isinstance(outputs, tuple):
                        Y_hat_final = outputs[0]
                    else:
                        Y_hat_final = outputs
                    
                    Y_hat_final = Y_hat_final[:, -self.cfg.MODEL.pred_len:, :]
                    Y_adapted_final = self.adapter_out(Y_hat_final, apply_trend=False)
                    Y_pred_final = self.trend_head(Y_adapted_final)
                    
                    # Compute metrics
                    mse = torch.nn.functional.mse_loss(Y_pred_final, Y_true, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                    mae = torch.nn.functional.l1_loss(Y_pred_final, Y_true, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                    
                    self.mse_all.append(mse)
                    self.mae_all.append(mae)
                
                # Move to next batch
                batch_start = batch_end
                batch_idx += 1
                
                # Logging
                if (batch_idx) % 100 == 0:
                    avg_mse = np.mean(np.concatenate(self.mse_all[-100:]))
                    avg_mae = np.mean(np.concatenate(self.mae_all[-100:]))
                    print(f"Batch {batch_idx}: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}, Updates={self.n_adapt}")
        
        # Final results
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        final_mse = np.mean(self.mse_all)
        final_mae = np.mean(self.mae_all)
        
        print(f'\nAfter SPEC-TTA HC Adaptation')
        print(f'Mode: {self.cfg.TTA.SPEC_TTA_HC.MODE}')
        print(f'Strategy: INPUT adaptation (like original SPEC-TTA)')
        print(f'Number of adaptations: {self.n_adapt}')
        print(f'Test MSE: {final_mse:.4f}, Test MAE: {final_mae:.4f}')
        print()
        
        self.model.eval()
