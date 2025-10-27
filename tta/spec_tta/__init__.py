# tta/spec_tta/__init__.py
from .manager import SpecTTAManager, SpecTTAConfig

def build_adapter(cfg, model, norm_module=None):
    """
    Build a SpecTTA adapter compatible with PETSA's TTA framework.
    
    Args:
        cfg: Configuration object
        model: Forecasting model (will be frozen)
        norm_module: Optional normalization module
    
    Returns:
        SpecTTAAdapter instance
    """
    from .adapter_wrapper import SpecTTAAdapter
    return SpecTTAAdapter(cfg, model, norm_module=norm_module)
