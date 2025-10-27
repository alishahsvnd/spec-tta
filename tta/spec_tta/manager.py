# tta/spec_tta/manager.py
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from .spectral_adapter import SpectralAdapter
from .temporal_heads import TimeShiftHead, PolyTrendHead
from .losses import (
    huber_loss, frequency_l1_loss, patchwise_structural_loss, horizon_consistency_loss,
    huber_loss_masked, frequency_l1_loss_pt_prefix, patchwise_structural_loss_pt_prefix
)
from .drift import spectral_drift_score, spectral_drift_pt_prefix
from .utils_pt import pt_prefix_length
from utils.fft_compat import rfft_1d
from .checkpoint_quality import CheckpointQualityDetector, print_quality_report
from .lora_time import LowRankTimeAdaptation, HybridAdaptationGate, print_lora_summary

class SpecTTAConfig:
    def __init__(
        self,
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
        reselection_every: int = 0,  # 0 => fixed bins; >0 => recompute every N updates
        use_adaptive_schedule: bool = False,  # E: Enable adaptive loss weights based on horizon
        use_output_only: bool = False,  # D: Freeze input adapter for long horizons (H>=240)
        use_safe_updates: bool = False,  # F: Enable safe update manager with rollback
        enable_h720_adaptation: bool = True  # Allow adaptation at H>=720 (remove forced baseline)
    ):
        self.L, self.T, self.V = L, T, V
        self.k_bins = k_bins
        self.patch_len = patch_len
        self.huber_delta = huber_delta
        self.lambda_pw = lambda_pw
        self.lambda_hc = lambda_hc
        self.drift_threshold = drift_threshold
        self.grad_clip = grad_clip
        self.device = device
        self.reselection_every = reselection_every
        self.use_adaptive_schedule = use_adaptive_schedule
        self.use_output_only = use_output_only
        self.use_safe_updates = use_safe_updates
        self.enable_h720_adaptation = enable_h720_adaptation
        # Note: use_horizon_adaptive_bins will be set by adapter_wrapper based on config flag
        self.use_horizon_adaptive_bins = False  # Will be updated if horizon-adaptive mode is active
        
        # Apply adaptive schedule if enabled (Improvement E)
        if use_adaptive_schedule:
            from .advanced_improvements import get_adaptive_loss_weights
            adaptive_weights = get_adaptive_loss_weights(T)
            self.beta_freq = adaptive_weights['beta_freq']
            self.lambda_pw = adaptive_weights['lambda_pw']
            self.lambda_prox = adaptive_weights['lambda_prox']
            self.lr = adaptive_weights['lr']
        else:
            self.beta_freq = beta_freq
            self.lambda_prox = lambda_prox
            self.lr = lr

class SpecTTAManager(nn.Module):
    def __init__(self, forecaster: nn.Module, cfg: SpecTTAConfig):
        super().__init__()
        self.forecaster = forecaster
        self.cfg = cfg

        # Initialize selected frequency bins (top-K energy from lookback spectrum when first seen)
        self.register_buffer("selected_bins", torch.arange(cfg.k_bins))

        # Placeholders until first call
        self.adapter_in: Optional[SpectralAdapter] = None
        self.adapter_out: Optional[SpectralAdapter] = None
        self.time_shift: Optional[TimeShiftHead] = None
        self.trend_head: Optional[PolyTrendHead] = None

        self.update_count = 0
        self._high_drift_streak = 0  # Track consecutive high-drift updates for bin reselection
        self._last_X = None  # Cache last input for fallback bin selection
        
        # Phase 1: Checkpoint quality detection for adaptive capacity
        self.quality_detector = CheckpointQualityDetector()
        self.quality_assessed = False
        self.checkpoint_quality = None
        
        # Phase 2: LoRA time-domain adaptation for hybrid mode
        self.lora_time: Optional[LowRankTimeAdaptation] = None
        self.hybrid_gate: Optional[HybridAdaptationGate] = None
        self.use_hybrid_mode = False  # Set to True when checkpoint quality is POOR/FAIR
        
        # F: Initialize safe update manager if enabled
        if cfg.use_safe_updates:
            from .advanced_improvements import SafeUpdateManager
            self.safe_update_manager = SafeUpdateManager(max_param_norm=5.0, patience=5)
        else:
            self.safe_update_manager = None

    def _select_bins(self, X: torch.Tensor) -> List[int]:
        # X: [B, L, V]
        fr, fi = rfft_1d(X, n=self.cfg.L, dim=1)
        energy = (fr ** 2 + fi ** 2).sum(dim=(0, 2))  # [F]
        K = min(self.cfg.k_bins, energy.numel())
        k_bins = torch.topk(energy, K, largest=True).indices
        return k_bins.sort().values.tolist()

    def _assess_checkpoint_quality(self, X: torch.Tensor, Y: torch.Tensor,
                                    x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Phase 1: Assess checkpoint quality on first batch and auto-tune capacity.
        This allows SPEC-TTA to adapt its capacity based on checkpoint strength.
        NOTE: Baseline MSE is computed on model's internal normalized space, which may differ
        from final denormalized MSE. Thresholds are calibrated for this space.
        """
        if self.quality_assessed:
            return
        
        # Evaluate baseline performance (no adaptation)
        self.forecaster.eval()
        with torch.no_grad():
            if x_mark_enc is not None:
                Y_pred = self.forecaster(X, x_mark_enc, x_dec, x_mark_dec)
            else:
                Y_pred = self.forecaster(X)
            
            # Handle attention output if needed
            if isinstance(Y_pred, tuple):
                Y_pred = Y_pred[0]
            
            # Extract prediction horizon
            Y_pred = Y_pred[:, -Y.size(1):, :]
            
            # Compute MSE (Y_pred is in model's output space, Y is in raw/dataset space)
            # This may be comparing different spaces, but thresholds are calibrated for this
            baseline_mse = torch.nn.functional.mse_loss(Y_pred, Y).item()
        
        # Determine quality level - ADJUSTED thresholds for mixed-space comparison
        # Based on empirical observations:
        # - Good checkpoints (final MSE ~0.18): baseline MSE ~3.5
        # - Poor checkpoints: baseline MSE >10
        if baseline_mse < 5.0:
            quality = "excellent"
        elif baseline_mse < 8.0:
            quality = "good"
        elif baseline_mse < 12.0:
            quality = "fair"
        else:
            quality = "poor"
        
        # Get adaptive configuration
        adaptive_cfg = self.quality_detector.get_adaptive_config(quality)
        
        # Auto-tune hyperparameters based on checkpoint quality
        # IMPORTANT: Preserve horizon-adaptive k_bins if enabled (don't override!)
        original_k_bins = self.cfg.k_bins
        
        # Only override k_bins if NOT using horizon-adaptive mode
        if not self.cfg.use_horizon_adaptive_bins:
            self.cfg.k_bins = adaptive_cfg['k_bins']
        else:
            # Keep horizon-adaptive bins, but adjust other hyperparameters
            print(f"🔒 Preserving horizon-adaptive k_bins={self.cfg.k_bins} (not auto-tuning)")
        
        self.cfg.beta_freq = adaptive_cfg['beta_freq']
        self.cfg.lr = adaptive_cfg['lr']
        self.cfg.lambda_pw = adaptive_cfg['lambda_pw']
        self.cfg.drift_threshold = adaptive_cfg['drift_threshold']
        
        # Print diagnostic report
        print_quality_report(quality, baseline_mse, adaptive_cfg)
        
        if not self.cfg.use_horizon_adaptive_bins and original_k_bins != self.cfg.k_bins:
            print(f"📊 Auto-tuned k_bins: {original_k_bins} → {self.cfg.k_bins}")
            print(f"   Expected parameters: ~{adaptive_cfg['expected_params']}")
            print(f"   (vs PETSA: ~25,934 params = {25934 / adaptive_cfg['expected_params']:.1f}x fewer)")
        
        self.checkpoint_quality = quality
        self.quality_assessed = True

    def _ensure_modules(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None,
                       x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Phase 1: Assess checkpoint quality on first batch BEFORE creating modules
        # This updates cfg.k_bins based on checkpoint quality
        if not self.quality_assessed and Y is not None:
            self._assess_checkpoint_quality(X, Y, x_mark_enc, x_dec, x_mark_dec)
        
        # Now create modules with potentially updated k_bins from quality assessment
        if self.adapter_in is None:
            bins = self._select_bins(X)
            self.selected_bins = torch.tensor(bins, device=X.device)
            self.adapter_in  = SpectralAdapter(self.cfg.L, self.cfg.V, bins).to(self.cfg.device)
            
            # Standard spectral adapter for all horizons
            # Note: For H>=720, safe mode will force gamma=0 (baseline only)
            self.adapter_out = SpectralAdapter(self.cfg.T, self.cfg.V, bins).to(self.cfg.device)
            
            # instantiate new temporal heads: fractional time-shift then quadratic trend
            self.time_shift = TimeShiftHead(self.cfg.T, self.cfg.V).to(self.cfg.device)
            self.trend_head = PolyTrendHead(self.cfg.T, self.cfg.V).to(self.cfg.device)
            
            # Phase 2: Create LoRA time-domain adapter for POOR/FAIR checkpoints
            if self.checkpoint_quality in ['poor', 'fair']:
                print("\n🚀 Enabling HYBRID mode: Frequency + Time domain adaptation")
                self.use_hybrid_mode = True
                
                # Create LoRA adapter (rank 4, ~1.7K params for iTransformer)
                self.lora_time = LowRankTimeAdaptation(
                    model=self.forecaster,
                    rank=4,
                    alpha=1.0,
                    target_modules=['query', 'key', 'value', 'out_proj']
                ).to(self.cfg.device)
                
                # Create adaptive blending gate (start with 70% frequency, 30% time)
                self.hybrid_gate = HybridAdaptationGate(
                    initial_alpha=0.7,
                    learnable=True
                ).to(self.cfg.device)
                
                print_lora_summary(self.lora_time)
                print(f"🎚️  Hybrid Gate: α={self.hybrid_gate.alpha:.2f} (freq weight)")
            
            self._build_optimizer()
            
            # Print confirmation of actual parameters created
            if self.checkpoint_quality is not None:
                actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
                print(f"✅ Created SPEC-TTA modules with {actual_params} parameters (Quality: {self.checkpoint_quality.upper()})")

    def _build_optimizer(self):
        # D: Output-only mode - freeze input adapter for long horizons to reduce compounding
        if self.cfg.use_output_only:
            # Only adapt output adapter, time shift, and trend head
            params = list(self.adapter_out.parameters()) + list(self.time_shift.parameters()) + list(self.trend_head.parameters())
            # Freeze input adapter
            for p in self.adapter_in.parameters():
                p.requires_grad = False
        else:
            # Standard mode - adapt all components
            params = list(self.adapter_in.parameters()) + list(self.adapter_out.parameters()) + list(self.time_shift.parameters()) + list(self.trend_head.parameters())
        
        # Phase 2: Add LoRA parameters if in hybrid mode
        if self.use_hybrid_mode and self.lora_time is not None:
            params += self.lora_time.get_parameters()
            # Add hybrid gate parameter
            if self.hybrid_gate is not None:
                params += list(self.hybrid_gate.parameters())
        
        self.opt = optim.Adam(params, lr=self.cfg.lr)

    def calibrate_input(self, X: torch.Tensor) -> torch.Tensor:
        self._ensure_modules(X)
        return self.adapter_in(X)

    def predict_with_calibration(self, X: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None, output_attention=False) -> torch.Tensor:
        # Phase 2: Support hybrid mode (frequency + time domain)
        if self.use_hybrid_mode and self.lora_time is not None:
            return self._predict_hybrid(X, x_mark_enc, x_dec, x_mark_dec, output_attention)
        
        # Standard frequency-domain only mode
        Xc = self.calibrate_input(X)
        # forecaster is frozen, as in PETSA
        if output_attention and x_mark_enc is not None:
            Y_hat, _ = self.forecaster(Xc, x_mark_enc, x_dec, x_mark_dec)
        elif x_mark_enc is not None:
            Y_hat = self.forecaster(Xc, x_mark_enc, x_dec, x_mark_dec)
        else:
            Y_hat = self.forecaster(X)
        Y_co = self.adapter_out(Y_hat)
        Y_co = self.time_shift(Y_co)
        Y_out = self.trend_head(Y_co)
        return Y_out
    
    def _predict_hybrid(self, X: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None, output_attention=False) -> torch.Tensor:
        """
        Hybrid prediction combining frequency-domain (SPEC-TTA) and time-domain (LoRA) adaptations.
        
        Computes two prediction paths:
        1. Frequency path: Input adapter → Forecaster → Output adapter → Temporal heads
        2. Time path: Forecaster with LoRA hooks → Output adapter → Temporal heads
        
        Then blends them using adaptive gate: α * freq + (1-α) * time
        """
        # Path 1: Frequency-domain adaptation (standard SPEC-TTA)
        Xc = self.calibrate_input(X)  # Apply input spectral adapter
        if x_mark_enc is not None:
            Y_freq = self.forecaster(Xc, x_mark_enc, x_dec, x_mark_dec)
        else:
            Y_freq = self.forecaster(Xc)
        if isinstance(Y_freq, tuple):
            Y_freq = Y_freq[0]
        Y_freq = self.adapter_out(Y_freq)
        Y_freq = self.time_shift(Y_freq)
        Y_freq = self.trend_head(Y_freq)
        
        # Path 2: Time-domain adaptation (LoRA on original input)
        # LoRA hooks are already registered, so just run forecaster
        if x_mark_enc is not None:
            Y_time = self.forecaster(X, x_mark_enc, x_dec, x_mark_dec)
        else:
            Y_time = self.forecaster(X)
        if isinstance(Y_time, tuple):
            Y_time = Y_time[0]
        Y_time = self.adapter_out(Y_time)
        Y_time = self.time_shift(Y_time)
        Y_time = self.trend_head(Y_time)
        
        # Blend predictions using adaptive gate
        Y_out = self.hybrid_gate(Y_freq, Y_time)
        
        return Y_out

    def _proximal_penalty(self) -> torch.Tensor:
        # Trust-region L2 on parameter deltas (they're initialized at 0)
        reg = 0.0
        params_list = list(self.adapter_in.parameters()) + list(self.adapter_out.parameters()) + list(self.time_shift.parameters()) + list(self.trend_head.parameters())
        
        # Phase 2: Add LoRA parameters to proximal penalty if in hybrid mode
        if self.use_hybrid_mode and self.lora_time is not None:
            params_list += self.lora_time.get_parameters()
        
        for p in params_list:
            reg = reg + (p ** 2).sum()
        return self.cfg.lambda_prox * reg

    def _bins_from_residual(self, Y_hat: torch.Tensor, Y_pt: torch.Tensor, mask_pt: torch.Tensor, K: int) -> List[int]:
        """
        Select top-K frequency bins based on spectral residual energy on PT prefix.
        Falls back to lookback energy if PT prefix is too short or unavailable.
        
        Args:
            Y_hat: [B, T, V] predictions
            Y_pt: [B, T, V] partial targets
            mask_pt: [B, T, V] observation mask
            K: number of bins to select
            
        Returns:
            List of K selected bin indices (sorted)
        """
        M = pt_prefix_length(mask_pt)
        
        if M >= 4 and mask_pt.sum() > 0:
            # Use PT prefix residual energy
            resid = (Y_hat[:, :M, :] - Y_pt[:, :M, :])  # [B, M, V]
            rr, ri = rfft_1d(resid, n=M, dim=1)  # [B, F, V]
            score = (rr.abs() + ri.abs()).mean(dim=(0, 2))  # [F] - mean across batch and variables
        else:
            # Fallback to lookback energy if PT prefix too short
            if self._last_X is not None:
                fr, fi = rfft_1d(self._last_X, n=self.cfg.L, dim=1)
                score = (fr ** 2 + fi ** 2).sum(dim=(0, 2))  # [F]
            else:
                # Ultimate fallback: uniform distribution
                F = (self.cfg.L // 2) + 1
                score = torch.ones(F, device=Y_hat.device)
        
        K = min(K, score.numel())
        return torch.topk(score, K, largest=True).indices.sort().values.tolist()
    
    def _maybe_reselect_bins(self, new_bins: List[int]):
        """
        Reselect frequency bins and warm-start overlapping gains.
        
        Args:
            new_bins: List of new bin indices to use
        """
        if new_bins == self.selected_bins.tolist():
            return  # No change needed
        
        # Skip bin reselection for H>=720 (using seasonal envelope adapter with fixed bins)
        if self.cfg.T >= 720:
            return
        
        # Save old adapters for warm-starting
        in_old, out_old = self.adapter_in, self.adapter_out
        
        # Create new adapters with updated bins
        self.adapter_in = SpectralAdapter(self.cfg.L, self.cfg.V, new_bins).to(self.cfg.device)
        self.adapter_out = SpectralAdapter(self.cfg.T, self.cfg.V, new_bins).to(self.cfg.device)
        
        # Warm-start: copy overlapping bin gains from old to new
        old_bin_map = {int(k): j for j, k in enumerate(in_old.k_bins.tolist())}
        for new_j, k in enumerate(new_bins):
            if k in old_bin_map:
                old_j = old_bin_map[k]
                # Copy gains from old adapter at position old_j to new adapter at position new_j
                self.adapter_in.g_real.data[:, new_j] = in_old.g_real.data[:, old_j]
                self.adapter_in.g_imag.data[:, new_j] = in_old.g_imag.data[:, old_j]
                self.adapter_out.g_real.data[:, new_j] = out_old.g_real.data[:, old_j]
                self.adapter_out.g_imag.data[:, new_j] = out_old.g_imag.data[:, old_j]
        
        # Update selected bins buffer
        self.selected_bins = torch.tensor(new_bins, device=self.cfg.device)
        
        # Rebuild optimizer with new parameters
        self._build_optimizer()

    def adapt_step(
        self,
        X: torch.Tensor,              # [B, L, V]
        Y_hat: torch.Tensor,          # [B, T, V] pre-calibrated model output (forecaster(Xc) if already computed)
        Y_pt: torch.Tensor,           # [B, T, V] with observed PT positions filled; arbitrary elsewhere
        mask_pt: torch.Tensor,        # [B, T, V] 1 at observed PT indices, 0 otherwise
        forecaster_short=None,        # optional callable for short-horizon for horizon-consistency
        T_short: int = 0,
        Y_base: torch.Tensor = None,  # [B, T, V] baseline forecast (frozen model) for safe mode
        use_safe_mode: bool = False   # Enable safe mode mixing
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        One PT update with robust + structural + (small) frequency + proximal + (optional) horizon-consistency.
        Supports adaptive bin reselection under sustained drift.
        
        With use_safe_mode=True, returns convex mix of baseline and adapted forecasts
        using holdout-validated gamma to ensure no regression.
        
        Returns:
            metrics: Dict of loss/drift metrics
            Y_final: Final forecast (adapted or mixed with baseline if safe mode enabled)
        """
        self._ensure_modules(X)
        
        # Cache last input for fallback bin selection
        self._last_X = X.detach()

        # Compute current calibrated prediction (apply time-shift then trend)
        Y_cal = self.adapter_out(Y_hat)
        Y_cal = self.time_shift(Y_cal)
        Y_cal = self.trend_head(Y_cal)

        # Drift trigger: skip if drift small (use PT-prefix-aware drift)
        with torch.no_grad():
            drift = spectral_drift_pt_prefix(Y_cal, Y_pt, mask_pt)
        metrics = {"drift": float(drift)}
        
        # Track consecutive high-drift updates for adaptive bin reselection
        if float(drift) > self.cfg.drift_threshold:
            self._high_drift_streak += 1
        else:
            self._high_drift_streak = 0
        
        # Adaptive bin reselection: if drift sustained for patience updates, refocus on shifted frequencies
        if (self.cfg.reselection_every > 0) and (self._high_drift_streak >= self.cfg.reselection_every):
            new_bins = self._bins_from_residual(Y_hat, Y_pt, mask_pt, self.cfg.k_bins)
            self._maybe_reselect_bins(new_bins)
            self._high_drift_streak = 0  # Reset streak after reselection
            metrics["reselected_bins"] = True
        else:
            metrics["reselected_bins"] = False

        if drift < self.cfg.drift_threshold:
            # No update needed, return baseline if available, else current calibrated forecast
            Y_final = Y_base if (use_safe_mode and Y_base is not None) else Y_cal
            return metrics, Y_final  # no update

        self.opt.zero_grad()

        # Loss on observed PT prefix positions (avoids zero-padding artifacts)
        if mask_pt.sum() > 0:
            # Use PT-prefix-aware loss functions
            loss_huber = huber_loss_masked(Y_cal, Y_pt, mask_pt, delta=self.cfg.huber_delta)
            loss_pw = patchwise_structural_loss_pt_prefix(Y_cal, Y_pt, mask_pt, patch_len=self.cfg.patch_len)
            loss_freq = frequency_l1_loss_pt_prefix(Y_cal, Y_pt, mask_pt)
        else:
            loss_huber = Y_cal.mean() * 0.0
            loss_pw    = Y_cal.mean() * 0.0
            loss_freq  = Y_cal.mean() * 0.0

        # Optional horizon-consistency (label-free)
        loss_hc = Y_cal.mean() * 0.0
        if (forecaster_short is not None) and (T_short > 0):
            with torch.no_grad():
                Xc = self.adapter_in(X)
                Y_short_hat = forecaster_short(Xc, T_short)  # user-provided wrapper if backbone allows variable T
            Y_short_cal = self.trend_head(self.time_shift(self.adapter_out(Y_short_hat)))[:, :T_short, :]
            loss_hc = horizon_consistency_loss(Y_cal, Y_short_cal)

        loss = loss_huber + self.cfg.lambda_pw * loss_pw + self.cfg.beta_freq * loss_freq \
               + self.cfg.lambda_hc * loss_hc + self._proximal_penalty()

        # F: Save state before update if safe update manager enabled
        if self.safe_update_manager is not None:
            modules = {
                'adapter_in': self.adapter_in,
                'adapter_out': self.adapter_out,
                'time_shift': self.time_shift,
                'trend_head': self.trend_head
            }
            self.safe_update_manager.before_update(modules)

        loss.backward()
        
        # Phase 2: Gradient clipping for all parameters including LoRA
        grad_params = list(self.adapter_in.parameters()) + list(self.adapter_out.parameters()) + list(self.time_shift.parameters()) + list(self.trend_head.parameters())
        if self.use_hybrid_mode and self.lora_time is not None:
            grad_params += self.lora_time.get_parameters()
            if self.hybrid_gate is not None:
                grad_params += list(self.hybrid_gate.parameters())
        
        torch.nn.utils.clip_grad_norm_(grad_params, self.cfg.grad_clip)
        self.opt.step()
        self.update_count += 1

        # F: Check safety and potentially rollback after update
        if self.safe_update_manager is not None:
            safe_metrics = self.safe_update_manager.after_update(float(loss.detach()), modules)
            metrics.update(safe_metrics)

        metrics.update({
            "loss": float(loss.detach()),
            "huber": float(loss_huber.detach()),
            "pw": float(loss_pw.detach()),
            "freq": float(loss_freq.detach()),
            "hc": float(loss_hc.detach())
        })

        # Closed-form trend update using PT prefix (faster and more stable)
        self.trend_head.closed_form_update_prefix(
            self.time_shift(self.adapter_out(Y_hat).detach()),  # Pre-trend shifted calibrated prediction
            Y_pt,                                                # Ground truth targets
            mask_pt                                              # Observation mask
        )

        # Safe mode: Mix baseline and adapted forecasts using holdout validation
        if use_safe_mode and Y_base is not None:
            from .safe_mode import choose_gamma_safe, safe_mix
            
            # For H>=720: Check if adaptation should be allowed
            # Legacy behavior (enable_h720_adaptation=False): Force gamma=0 (baseline only)
            # New behavior (enable_h720_adaptation=True): Allow safe mode to choose gamma adaptively
            if self.cfg.T >= 720 and not self.cfg.enable_h720_adaptation:
                gamma = 0.0
                ho_loss = 0.0
                Y_final = Y_base  # Always use baseline for H>=720 (legacy mode)
                metrics["safe_gamma"] = 0.0
                metrics["safe_holdout_huber"] = 0.0
                metrics["safe_mode_rejected"] = True
                metrics["safe_mode_forced_baseline"] = True
                if self.update_count == 1:
                    print(f"[Safe Mode] H={self.cfg.T}>=720: Forcing gamma=0.0 (baseline only, no adaptation)")
            else:
                # Get adapted forecast after closed-form trend update
                Y_adapt = self.trend_head(self.time_shift(self.adapter_out(Y_hat)))
                
                # Find optimal mixing weight via holdout validation
                gamma, ho_loss = choose_gamma_safe(
                    y_base=Y_base,
                    y_adapt=Y_adapt,
                    y_true=Y_pt,
                    mask_obs=mask_pt
                )
                
                # Apply convex mixing
                Y_final = safe_mix(Y_base, Y_adapt, gamma)
                
                # Track safe mode metrics
                metrics["safe_gamma"] = float(gamma)
                metrics["safe_holdout_huber"] = float(ho_loss)
                
                # If gamma=0, this update provided no benefit (baseline better)
                if gamma == 0.0:
                    metrics["safe_mode_rejected"] = True
                    if self.update_count % 10 == 0:
                        print(f"[Safe Mode] Update {self.update_count}: gamma=0.0 (baseline better, rejected adaptation)")
                else:
                    metrics["safe_mode_rejected"] = False
                    if self.update_count % 10 == 0:
                        print(f"[Safe Mode] Update {self.update_count}: gamma={gamma:.3f} (controlled mixing applied)")
        else:
            # Standard mode: return adapted forecast
            Y_final = self.trend_head(self.time_shift(self.adapter_out(Y_hat)))

        return metrics, Y_final
