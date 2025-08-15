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
        # Set a flag to indicate we're in initialization
        self._initializing = True
        
        # Must call super().__init__() first for nn.Module
        super().__init__()
        
        # Now we can safely assign the model and other attributes
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
        
        # Clear the initialization flag
        self._initializing = False
        
        # Initial binding of generation methods (following Dream pattern)
        try:
            self._bind_dream_methods()
        except Exception as e:
            # If binding fails, that's ok - methods might already exist or be bound later
            pass

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
        
    def gradient_checkpointing_enable(self):
        """
        代理 gradient_checkpointing_enable 到底层模型，如果没有则抛出友好异常。
        """
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable()
        raise AttributeError(f"Underlying model does not support gradient_checkpointing_enable")
    
    # Provide attribute passthrough for convenience (so external code can still access model.*)
    def __getattr__(self, name: str):
        # During initialization, just raise AttributeError for unknown attributes
        if self.__dict__.get('_initializing', False):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # After initialization, check if we have the model using __dict__ to avoid recursion
        model_instance = self.__dict__.get('model', None)
        if model_instance is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (model not initialized)")
        
        # Check if the internal model has this attribute
        if hasattr(model_instance, name):
            return getattr(model_instance, name)
        
        # If the internal model doesn't have this attribute either, raise error
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
