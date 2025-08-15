# src/open_r1/utils/dream_compat.py
import torch
from typing import Any, Optional
import torch.nn as nn
import types

class ModelCompatWrapper(nn.Module):
    """
    Wrap a DreamModel (or other model that exposes dual_cache_generate/diffusion_generate)
    and provide a minimal compatibility layer expected by the trainer codebase.
    
    This wrapper follows the same pattern as the Dream evaluation code.
    """

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        # Expose generation_config if model has it
        self.generation_config = getattr(model, "generation_config", None)
        
        # Expose config if model has it, otherwise create a minimal one
        if hasattr(model, "config"):
            self.config = model.config
        else:
            # Create a minimal config object for compatibility
            class MinimalConfig:
                def __init__(self):
                    self._name_or_path = getattr(model, '_name_or_path', 'unknown_model')
                    self.model_type = getattr(model, 'model_type', 'custom')
            self.config = MinimalConfig()
        
        # Initial binding of generation methods (following Dream pattern)
        self._bind_dream_methods()

    def _bind_dream_methods(self, use_cache=False):
        """
        Bind Dream generation methods to the model.
        This follows the exact pattern from Dream evaluation code.
        """
        try:
            if use_cache:
                from model.generation_utils_block import DreamGenerationMixin
            else:
                from model.generation_utils import DreamGenerationMixin
            
            self.model.diffusion_generate = types.MethodType(
                DreamGenerationMixin.diffusion_generate, self.model
            )
            self.model._sample = types.MethodType(
                DreamGenerationMixin._sample, self.model
            )
        except ImportError:
            # If Dream modules are not available, that's ok - the model might already have these methods
            pass

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def save_pretrained(self, save_directory: str):
        # Delegate if the underlying model implements it
        if hasattr(self.model, "save_pretrained"):
            return self.model.save_pretrained(save_directory)
        raise RuntimeError("Underlying model has no save_pretrained()")

    def resize_token_embeddings(self, *args, **kwargs):
        if hasattr(self.model, "resize_token_embeddings"):
            return self.model.resize_token_embeddings(*args, **kwargs)
        raise RuntimeError("Underlying model has no resize_token_embeddings()")

    def rebind_generation_methods(self, use_cache=False):
        """
        Rebind generation methods based on use_cache parameter.
        This should be called by the trainer code when needed, following Dream pattern.
        """
        self._bind_dream_methods(use_cache=use_cache)

    # Provide attribute passthrough for convenience (so external code can still access model.*)
    def __getattr__(self, name: str):
        # Use __dict__ to avoid recursion
        if 'model' not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (model not initialized)")
        
        model_instance = self.__dict__['model']
        
        # Check if the internal model has this attribute
        if hasattr(model_instance, name):
            return getattr(model_instance, name)
        
        # If the internal model doesn't have this attribute either, raise error
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
