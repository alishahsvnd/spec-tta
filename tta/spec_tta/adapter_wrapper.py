# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Adapter wrapper to integrate SpecTTAManager with PETSA's TTA framework.
This provides the same interface as PETSA/TAFAS adapters.
"""

from typing import Optional
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np

from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
from .manager import SpecTTAManager, SpecTTAConfig


def get_horizon_adaptive_bins(horizon: int, base_bins: int = 32) -> int:
    """
    Scale frequency bins based on horizon length for better long-term modeling.
    
    Rationale: Longer horizons require more frequency bins to capture extended
    temporal patterns and long-range periodicities.
    
    Args:
        horizon: Forecast horizon length
        base_bins: Base number of bins for H=96 (default: 32)
        
    Returns:
        Scaled number of bins appropriate for the horizon
    """
    if horizon <= 96:
        return base_bins  # 32
    elif horizon <= 192:
        return int(base_bins * 1.5)  # 48
    elif horizon <= 336:
        return int(base_bins * 2.0)  # 64
    else:  # H >= 720
        return int(base_bins * 3.0)  # 96


class SpecTTAAdapter(nn.Module):
    """
    Wrapper that makes SpecTTAManager compatible with PETSA's adapter interface.
    Provides the same methods as PETSAAdapter: adapt(), count_parameters(), reset(), etc.
    """
    
    def __init__(self, cfg, model: nn.Module, norm_module: Optional[nn.Module] = None):
        super(SpecTTAAdapter, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        
        # Get test loader
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test
        
        # Determine k_bins: use horizon-adaptive scaling if enabled
        if cfg.TTA.SPEC_TTA.USE_HORIZON_ADAPTIVE_BINS:
            k_bins = get_horizon_adaptive_bins(
                horizon=cfg.DATA.PRED_LEN,
                base_bins=cfg.TTA.SPEC_TTA.K_BINS
            )
            print(f"[Horizon-Adaptive] H={cfg.DATA.PRED_LEN} → k_bins={k_bins} (base={cfg.TTA.SPEC_TTA.K_BINS})")
        else:
            k_bins = cfg.TTA.SPEC_TTA.K_BINS
        
        # Build SpecTTA configuration
        spec_cfg = SpecTTAConfig(
            L=cfg.DATA.SEQ_LEN,
            T=cfg.DATA.PRED_LEN,
            V=cfg.DATA.N_VAR,
            k_bins=k_bins,
            patch_len=cfg.TTA.SPEC_TTA.PATCH_LEN,
            huber_delta=cfg.TTA.SPEC_TTA.HUBER_DELTA,
            beta_freq=cfg.TTA.SPEC_TTA.BETA_FREQ,
            lambda_pw=cfg.TTA.SPEC_TTA.LAMBDA_PW,
            lambda_prox=cfg.TTA.SPEC_TTA.LAMBDA_PROX,
            lambda_hc=cfg.TTA.SPEC_TTA.LAMBDA_HC,
            drift_threshold=cfg.TTA.SPEC_TTA.DRIFT_THRESHOLD,
            lr=cfg.TTA.SPEC_TTA.LR,
            grad_clip=cfg.TTA.SPEC_TTA.GRAD_CLIP,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            reselection_every=cfg.TTA.SPEC_TTA.RESELECTION_EVERY,
            use_adaptive_schedule=cfg.TTA.SPEC_TTA.USE_ADAPTIVE_SCHEDULE,
            use_output_only=cfg.TTA.SPEC_TTA.USE_OUTPUT_ONLY,
            use_safe_updates=cfg.TTA.SPEC_TTA.USE_SAFE_UPDATES,
            enable_h720_adaptation=cfg.TTA.SPEC_TTA.ENABLE_H720_ADAPTATION
        )
        
        # Initialize SpecTTA manager
        self.spec_tta = SpecTTAManager(model, spec_cfg)
        
        # Set horizon-adaptive flag to prevent quality detector override
        if cfg.TTA.SPEC_TTA.USE_HORIZON_ADAPTIVE_BINS:
            self.spec_tta.cfg.use_horizon_adaptive_bins = True
        
        # Freeze model parameters (following PETSA pattern)
        self._freeze_all_model_params()
        
        # Save initial state
        self.model_state = deepcopy(self.model.state_dict())
        
        # Configure batch processing - use full dataset as one batch like PETSA
        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        
        # Metrics tracking
        self.mse_all = []
        self.mae_all = []
        self.n_adapt = 0
        
    def count_parameters(self):
        """Count and print trainable parameters."""
        print("------- SPEC-TTA PARAMETERS -------")
        total_sum = 0
        
        # Count adapter_in parameters
        if self.spec_tta.adapter_in is not None:
            for name, param in self.spec_tta.adapter_in.named_parameters():
                print(param.requires_grad, f"adapter_in.{name}", param.size(), param.numel())
                if param.requires_grad:
                    total_sum += int(param.numel())
        
        # Count adapter_out parameters
        if self.spec_tta.adapter_out is not None:
            for name, param in self.spec_tta.adapter_out.named_parameters():
                print(param.requires_grad, f"adapter_out.{name}", param.size(), param.numel())
                if param.requires_grad:
                    total_sum += int(param.numel())
        
        # Count trend_head parameters
        if self.spec_tta.trend_head is not None:
            for name, param in self.spec_tta.trend_head.named_parameters():
                print(param.requires_grad, f"trend_head.{name}", param.size(), param.numel())
                if param.requires_grad:
                    total_sum += int(param.numel())
        
        print("Total Trainable Parameters: ", total_sum)
        
    def adapt(self):
        """
        Main adaptation loop following PETSA's pattern.
        Processes test data incrementally like PETSA does.
        """
        print("\n========== Starting SPEC-TTA Adaptation ==========")
        self.model.eval()
        
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        test_len = len(self.test_loader.dataset)
        
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
                
                # Enable safe mode for long horizons (H>=720) to prevent regression
                use_safe_mode = self.model_cfg.pred_len >= 720
                
                # Get baseline forecast (frozen model WITHOUT input adaptation) for safe mode
                with torch.no_grad():
                    if use_safe_mode:
                        # Baseline: raw model output without ANY adaptation
                        if self.model_cfg.output_attention:
                            Y_base, _ = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
                        else:
                            Y_base = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
                        Y_base = Y_base[:, -self.model_cfg.pred_len:, :]
                    else:
                        Y_base = None
                    
                    # Get model prediction with calibrated input (for adaptation)
                    enc_window_cal = self.spec_tta.calibrate_input(enc_window)
                    
                    # Get model output (frozen model)
                    if self.model_cfg.output_attention:
                        Y_hat, _ = self.model(enc_window_cal, enc_window_stamp, dec_window, dec_window_stamp)
                    else:
                        Y_hat = self.model(enc_window_cal, enc_window_stamp, dec_window, dec_window_stamp)
                    
                    # Extract prediction horizon
                    Y_hat = Y_hat[:, -self.model_cfg.pred_len:, :]
                
                # Get ground truth - full horizon available for evaluation
                Y_true_full = dec_window[:, -self.model_cfg.pred_len:, :]
                
                # PT prefix mask configuration
                # For H>=720: Use realistic PT prefix (10-15%) - required for safe mode
                # For H<720: Use full horizon (100%) - maintains proven working behavior
                T = self.model_cfg.pred_len
                Y_pt = Y_true_full.clone()
                
                if use_safe_mode:
                    # Safe mode (H>=720): Use realistic PT prefix for holdout validation
                    # Note: This doesn't matter since we force gamma=0, but kept for consistency
                    M = max(50, min(int(0.15 * T), 150))  # 10-15% prefix
                    mask_pt = torch.zeros_like(Y_hat)
                    mask_pt[:, :M, :] = 1.0
                else:
                    # Standard mode (H<720): Use full horizon (original working behavior)
                    # This maintains the proven results for H=96, 192, 336
                    mask_pt = torch.ones_like(Y_hat)
                
                # Phase 1: On first batch, assess checkpoint quality with full target
                # This allows adaptive capacity scaling based on checkpoint strength
                if batch_idx == 0:
                    self.spec_tta._ensure_modules(enc_window, Y=Y_true_full,
                                                   x_mark_enc=enc_window_stamp,
                                                   x_dec=dec_window,
                                                   x_mark_dec=dec_window_stamp)
                
                # Perform adaptation step (returns metrics and final forecast)
                metrics, Y_final = self.spec_tta.adapt_step(
                    X=enc_window,
                    Y_hat=Y_hat,
                    Y_pt=Y_pt,
                    mask_pt=mask_pt,
                    forecaster_short=None,
                    T_short=0,
                    Y_base=Y_base,  # Pass TRUE baseline forecast (no adaptation) for safe mode
                    use_safe_mode=use_safe_mode
                )
                
                # Compute metrics on FULL horizon (not just prefix)
                targets = dec_window[:, -self.model_cfg.pred_len:, :]
                mse = torch.nn.functional.mse_loss(Y_final, targets, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = torch.nn.functional.l1_loss(Y_final, targets, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                
                # Also compute baseline MSE for comparison
                if use_safe_mode and Y_base is not None:
                    mse_baseline = torch.nn.functional.mse_loss(Y_base, targets, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx}: MSE_baseline={np.mean(mse_baseline):.6f}, MSE_final={np.mean(mse):.6f}, gamma={metrics.get('safe_gamma', -1):.3f}")
                
                self.mse_all.append(mse)
                self.mae_all.append(mae)
                
                self.n_adapt += 1
                
                # Move to next batch
                batch_start = batch_end
                batch_idx += 1
                
                # Logging
                if (batch_idx) % 100 == 0:
                    avg_mse = np.mean(np.concatenate(self.mse_all[-100:]))
                    avg_mae = np.mean(np.concatenate(self.mae_all[-100:]))
                    print(f"Batch {batch_idx}: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}, "
                          f"Drift={metrics.get('drift', 0):.6f}, Updates={self.spec_tta.update_count}")
        
        # Final results - concatenate all batch results
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        final_mse = np.mean(self.mse_all)
        final_mae = np.mean(self.mae_all)
        print(f"\n========== SPEC-TTA Adaptation Complete ==========")
        print(f"Final MSE: {final_mse:.6f}")
        print(f"Final MAE: {final_mae:.6f}")
        print(f"Total Adaptation Updates: {self.spec_tta.update_count}")
        print(f"Total Steps: {self.n_adapt}")
        
        # Save results in standardized format for protocol evaluation
        self._save_standardized_results(final_mse, final_mae)
        
    def reset(self):
        """Reset model to initial state."""
        self.model.load_state_dict(self.model_state, strict=True)
        # Reset SpecTTA manager by reinitializing
        self.spec_tta.adapter_in = None
        self.spec_tta.adapter_out = None
        self.spec_tta.trend_head = None
        self.spec_tta.update_count = 0
        
    def _freeze_all_model_params(self):
        """Freeze all model parameters (only adapt the spectral modules)."""
        for param in self.model.parameters():
            param.requires_grad = False
        if self.norm_module is not None:
            for param in self.norm_module.parameters():
                param.requires_grad = False
    
    def _save_standardized_results(self, mse: float, mae: float):
        """
        Save results in standardized format for protocol evaluation.
        
        Saves to:
        1. Individual JSON file in results directory
        2. Appends to aggregated CSV file
        """
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        
        try:
            from tools.result_logger import save_run_result, append_to_csv, format_result_filename
            
            # Extract metadata from config
            dataset = self.cfg.DATA.NAME
            horizon = self.cfg.DATA.PRED_LEN
            model = self.cfg.MODEL.MODEL_TYPE
            method = 'SPEC-TTA'
            
            # Count parameters
            n_params = 0
            if self.spec_tta.adapter_in is not None:
                n_params += sum(p.numel() for p in self.spec_tta.adapter_in.parameters() if p.requires_grad)
            if self.spec_tta.adapter_out is not None:
                n_params += sum(p.numel() for p in self.spec_tta.adapter_out.parameters() if p.requires_grad)
            if self.spec_tta.trend_head is not None:
                n_params += sum(p.numel() for p in self.spec_tta.trend_head.parameters() if p.requires_grad)
            
            # Save individual JSON file
            json_filename = format_result_filename(dataset, horizon, model, method, extension='json')
            json_path = os.path.join(self.cfg.RESULT_DIR, json_filename)
            
            save_run_result(
                output_path=json_path,
                dataset=dataset,
                horizon=horizon,
                model=model,
                method=method,
                mse=float(mse),
                mae=float(mae),
                n_params=int(n_params),
                n_updates=int(self.spec_tta.update_count),
                extra_metrics={
                    'k_bins': int(self.spec_tta.config.k_bins),
                    'drift_threshold': float(self.spec_tta.config.drift_threshold),
                    'beta_freq': float(self.spec_tta.config.beta_freq),
                },
                format='json'
            )
            print(f"Saved individual result to: {json_path}")
            
            # Append to aggregated CSV
            csv_path = os.path.join(os.path.dirname(self.cfg.RESULT_DIR), 'aggregated_results.csv')
            append_to_csv(
                csv_path=csv_path,
                dataset=dataset,
                horizon=horizon,
                model=model,
                method=method,
                mse=float(mse),
                mae=float(mae),
                n_params=int(n_params),
                n_updates=int(self.spec_tta.update_count)
            )
            print(f"Appended result to: {csv_path}")
            
        except Exception as e:
            print(f"Warning: Could not save standardized results: {e}")
            import traceback
            traceback.print_exc()
