"""
LoRA-inspired Time-Domain Adaptation for SPEC-TTA

This module provides low-rank time-domain adaptation to complement SPEC-TTA's
frequency-domain adaptation. When checkpoint quality is poor, this enables 
hybrid frequency+time adaptation for improved robustness.

Key differences from PETSA:
- PETSA: Full LoRA on all linear layers (rank 16, ~25K params)
- This: Selective LoRA on key layers (rank 4, ~1.7K params)
- Focus on attention QKV and output projection for maximum impact
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for a single linear transformation.
    
    Implements: W' = W + BA where B ∈ R^(d_out × r), A ∈ R^(r × d_in)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        rank: Rank of low-rank decomposition (default: 4)
        alpha: Scaling factor for LoRA update (default: 1.0)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices: B (down-projection), A (up-projection)
        # Initialize A with small random values, B with zeros
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, original_output: torch.Tensor, original_input: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation to the original linear layer output.
        
        Args:
            original_output: Output from original W @ x
            original_input: Input x to the layer
            
        Returns:
            Adapted output: W @ x + scaling * B @ A @ x
        """
        # Compute low-rank update: BA @ x
        lora_output = (original_input @ self.lora_A.T) @ self.lora_B.T
        
        # Scale and add to original output
        return original_output + self.scaling * lora_output
    
    def parameters_count(self) -> int:
        """Return number of trainable parameters."""
        return self.lora_A.numel() + self.lora_B.numel()


class LowRankTimeAdaptation(nn.Module):
    """
    Time-domain adaptation using LoRA for SPEC-TTA hybrid mode.
    
    Applies selective low-rank updates to attention layers for improved
    adaptation when checkpoint quality is poor. Designed to work alongside
    frequency-domain adaptation.
    
    Target layers (per attention block):
    - query, key, value projections (3 × rank 4)
    - output projection (1 × rank 4)
    
    For iTransformer (4 encoder layers, d_model=512):
    - 4 layers × 4 LoRA modules × ~110 params = ~1760 params
    
    Args:
        model: The forecasting model to adapt
        rank: Rank for LoRA decomposition (default: 4)
        alpha: LoRA scaling factor (default: 1.0)
        target_modules: List of module name patterns to adapt
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha
        
        # Default: adapt attention QKV and output projections
        if target_modules is None:
            target_modules = ['query', 'key', 'value', 'out_proj']
        self.target_modules = target_modules
        
        # Storage for LoRA layers and hooks
        self.lora_layers: Dict[str, LoRALayer] = {}
        self.hooks = []
        self.activations = {}  # Store inputs for LoRA computation
        
        # Initialize LoRA layers for target modules
        self._initialize_lora_layers()
        
    def _initialize_lora_layers(self):
        """Scan model and create LoRA layers for target modules."""
        for name, module in self.model.named_modules():
            # Check if this module matches any target pattern
            if isinstance(module, nn.Linear) and any(target in name for target in self.target_modules):
                in_features = module.in_features
                out_features = module.out_features
                
                # Create LoRA layer
                lora_layer = LoRALayer(in_features, out_features, self.rank, self.alpha)
                self.lora_layers[name] = lora_layer
                
                # Register forward hook to apply LoRA
                hook = module.register_forward_hook(self._make_forward_hook(name, lora_layer))
                self.hooks.append(hook)
                
        # Convert to ModuleDict for proper parameter registration
        self.lora_layers = nn.ModuleDict(self.lora_layers)
        
    def _make_forward_hook(self, name: str, lora_layer: LoRALayer):
        """Create forward hook that applies LoRA adaptation."""
        def hook(module, input, output):
            # Store input for LoRA computation
            original_input = input[0]
            
            # Apply LoRA: output = W @ x + BA @ x
            adapted_output = lora_layer(output, original_input)
            
            return adapted_output
        return hook
    
    def enable(self):
        """Enable LoRA adaptation (hooks active)."""
        if not self.hooks:
            self._initialize_lora_layers()
    
    def disable(self):
        """Disable LoRA adaptation (remove hooks)."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def parameters_count(self) -> int:
        """Return total number of trainable LoRA parameters."""
        return sum(lora.parameters_count() for lora in self.lora_layers.values())
    
    def get_parameters(self) -> List[nn.Parameter]:
        """Return list of all LoRA parameters for optimization."""
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return params
    
    def state_dict_custom(self) -> Dict:
        """Return state dict with only LoRA parameters."""
        state = {}
        for name, lora_layer in self.lora_layers.items():
            state[f"{name}.lora_A"] = lora_layer.lora_A.data.clone()
            state[f"{name}.lora_B"] = lora_layer.lora_B.data.clone()
        return state
    
    def load_state_dict_custom(self, state: Dict):
        """Load LoRA parameters from state dict."""
        for name, lora_layer in self.lora_layers.items():
            if f"{name}.lora_A" in state:
                lora_layer.lora_A.data.copy_(state[f"{name}.lora_A"])
            if f"{name}.lora_B" in state:
                lora_layer.lora_B.data.copy_(state[f"{name}.lora_B"])


class HybridAdaptationGate(nn.Module):
    """
    Adaptive gating mechanism to blend frequency and time-domain adaptations.
    
    Learns to weight frequency vs time-domain updates based on their relative
    effectiveness. Starts with frequency-dominant (alpha=0.7) and adapts online.
    
    Output = alpha * freq_adaptation + (1 - alpha) * time_adaptation
    
    Args:
        initial_alpha: Initial weight for frequency domain (default: 0.7)
        learnable: Whether alpha is learnable vs fixed (default: True)
    """
    
    def __init__(self, initial_alpha: float = 0.7, learnable: bool = True):
        super().__init__()
        
        if learnable:
            # Alpha as learnable parameter (constrained to [0,1] via sigmoid)
            self.alpha_logit = nn.Parameter(
                torch.logit(torch.tensor(initial_alpha))
            )
        else:
            # Fixed alpha
            self.register_buffer('alpha_logit', torch.logit(torch.tensor(initial_alpha)))
        
        self.learnable = learnable
        
    @property
    def alpha(self) -> float:
        """Get current alpha value (frequency domain weight)."""
        return torch.sigmoid(self.alpha_logit).item()
    
    def forward(self, freq_output: torch.Tensor, time_output: torch.Tensor) -> torch.Tensor:
        """
        Blend frequency and time-domain model outputs.
        
        Args:
            freq_output: Output with frequency-domain adaptation applied
            time_output: Output with time-domain adaptation applied
            
        Returns:
            Blended output
        """
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * freq_output + (1 - alpha) * time_output
    
    def get_weights(self) -> Tuple[float, float]:
        """Return current (freq_weight, time_weight) for logging."""
        alpha = self.alpha
        return alpha, 1 - alpha


def print_lora_summary(lora_adapter: LowRankTimeAdaptation):
    """Print summary of LoRA adaptation setup."""
    print("\n" + "="*70)
    print("⏰ TIME-DOMAIN LoRA ADAPTATION SUMMARY")
    print("="*70)
    print(f"Rank:                {lora_adapter.rank}")
    print(f"Alpha:               {lora_adapter.alpha:.2f}")
    print(f"Target Modules:      {', '.join(lora_adapter.target_modules)}")
    print(f"LoRA Layers:         {len(lora_adapter.lora_layers)}")
    print(f"Total Parameters:    {lora_adapter.parameters_count():,}")
    print("="*70)
    print("\n📋 Adapted Layers:")
    for name in lora_adapter.lora_layers.keys():
        print(f"  • {name}")
    print("="*70 + "\n")
