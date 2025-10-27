# tta/spec_tta/checkpoint_quality.py
"""
Checkpoint Quality Detection for Adaptive SPEC-TTA Capacity
"""
import torch
import torch.nn.functional as F
from typing import Literal

CheckpointQuality = Literal["excellent", "good", "fair", "poor"]

class CheckpointQualityDetector:
    """
    Detect checkpoint quality on first batch to automatically tune SPEC-TTA capacity.
    
    Design Philosophy:
    - Excellent checkpoint (MSE < 0.4): Use minimal SPEC-TTA (k_bins=16, 448 params)
    - Good checkpoint (MSE < 0.6): Use standard SPEC-TTA (k_bins=32, 910 params)
    - Fair checkpoint (MSE < 0.8): Use high-capacity SPEC-TTA (k_bins=64, 1824 params)
    - Poor checkpoint (MSE >= 0.8): Use hybrid SPEC-TTA + LoRA (k_bins=64, 2624 params)
    """
    
    def __init__(self, 
                 excellent_threshold: float = 0.4,
                 good_threshold: float = 0.6,
                 fair_threshold: float = 0.8):
        self.excellent_threshold = excellent_threshold
        self.good_threshold = good_threshold
        self.fair_threshold = fair_threshold
        
    def assess_quality(self, forecaster: torch.nn.Module, 
                      batch: dict) -> tuple[CheckpointQuality, float]:
        """
        Evaluate baseline performance without adaptation.
        
        Args:
            forecaster: Pre-trained forecasting model
            batch: First test batch with keys 'x' (input) and 'y' (target)
            
        Returns:
            quality: One of "excellent", "good", "fair", "poor"
            baseline_mse: MSE without any adaptation
        """
        forecaster.eval()
        with torch.no_grad():
            # Forward pass without adaptation
            pred = forecaster(batch['x'])
            baseline_mse = F.mse_loss(pred, batch['y']).item()
        
        # Classify quality
        if baseline_mse < self.excellent_threshold:
            quality = "excellent"
        elif baseline_mse < self.good_threshold:
            quality = "good"
        elif baseline_mse < self.fair_threshold:
            quality = "fair"
        else:
            quality = "poor"
            
        return quality, baseline_mse
    
    def get_adaptive_config(self, quality: CheckpointQuality) -> dict:
        """
        Get SPEC-TTA configuration tuned for checkpoint quality.
        
        Args:
            quality: Checkpoint quality level
            
        Returns:
            config: Dictionary with optimal hyperparameters
        """
        if quality == "excellent":
            return {
                "k_bins": 32,           # Standard bins (empirically optimal for excellent checkpoints)
                "beta_freq": 0.1,       # Standard frequency mixing (matches GOOD mode)
                "lr": 0.001,            # Standard learning rate
                "lambda_pw": 1.0,       # Standard patchwise loss (matches GOOD mode)
                "drift_threshold": 0.005,  # Standard threshold (matches GOOD mode)
                "use_lora": False,      # No LoRA compensation
                "expected_params": 910,
                "mode": "standard"      # Changed from "minimal" to "standard"
            }
        elif quality == "good":
            return {
                "k_bins": 32,           # Standard frequency bins
                "beta_freq": 0.1,       # Standard frequency mixing
                "lr": 0.001,            # Standard learning rate
                "lambda_pw": 1.0,       # Standard patchwise loss
                "drift_threshold": 0.005,
                "use_lora": False,      # No LoRA compensation
                "expected_params": 910,
                "mode": "standard"
            }
        elif quality == "fair":
            return {
                "k_bins": 64,           # High frequency bins
                "beta_freq": 0.15,      # Stronger frequency mixing
                "lr": 0.002,            # Higher learning rate
                "lambda_pw": 1.5,       # Stronger patchwise loss
                "drift_threshold": 0.003,
                "use_lora": False,      # Still frequency-only
                "expected_params": 1824,
                "mode": "high_capacity"
            }
        else:  # poor
            return {
                "k_bins": 64,           # High frequency bins
                "beta_freq": 0.2,       # Strong frequency mixing
                "lr": 0.005,            # Aggressive learning rate
                "lambda_pw": 2.0,       # Strong patchwise loss
                "drift_threshold": 0.001,
                "use_lora": True,       # Add LoRA compensation!
                "lora_rank": 4,         # Low-rank time-domain adaptation
                "lora_alpha": 0.3,      # 30% time-domain, 70% frequency
                "expected_params": 2624,
                "mode": "hybrid"
            }

def print_quality_report(quality: CheckpointQuality, baseline_mse: float, config: dict):
    """Print diagnostic report about checkpoint quality and chosen configuration."""
    
    mode_emoji = {
        "minimal": "⚡",
        "standard": "✅", 
        "high_capacity": "🔥",
        "hybrid": "🚀"
    }
    
    quality_emoji = {
        "excellent": "🌟",
        "good": "✅",
        "fair": "⚠️",
        "poor": "❌"
    }
    
    print("\n" + "="*70)
    print(f"{quality_emoji[quality]} CHECKPOINT QUALITY ASSESSMENT")
    print("="*70)
    print(f"Quality Level:      {quality.upper()}")
    print(f"Baseline MSE:       {baseline_mse:.4f}")
    print(f"Selected Mode:      {config['mode'].upper()} {mode_emoji[config['mode']]}")
    print(f"Frequency Bins:     {config['k_bins']}")
    print(f"Learning Rate:      {config['lr']:.4f}")
    print(f"Beta Frequency:     {config['beta_freq']:.2f}")
    print(f"Use LoRA:           {config['use_lora']}")
    print(f"Expected Params:    {config['expected_params']}")
    print("="*70)
    
    # Provide context
    if quality == "excellent":
        print("✨ Excellent checkpoint! Using minimal SPEC-TTA for efficiency.")
    elif quality == "good":
        print("✅ Good checkpoint! Using standard SPEC-TTA configuration.")
    elif quality == "fair":
        print("⚠️  Fair checkpoint. Increasing capacity for robustness.")
    else:
        print("❌ Poor checkpoint detected! Enabling hybrid mode with LoRA.")
        print("   This will use frequency + time domain adaptation for recovery.")
    
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Example: Detect and configure
    detector = CheckpointQualityDetector()
    
    # Simulate different checkpoint qualities
    for baseline_mse in [0.3, 0.5, 0.7, 0.9]:
        # Mock quality detection
        if baseline_mse < 0.4:
            quality = "excellent"
        elif baseline_mse < 0.6:
            quality = "good"
        elif baseline_mse < 0.8:
            quality = "fair"
        else:
            quality = "poor"
        
        config = detector.get_adaptive_config(quality)
        print_quality_report(quality, baseline_mse, config)
