# src/open_r1/utils/dream_compat.py
import torch
from typing import Any, Optional
import torch.nn as nn
import types
from transformers import PretrainedConfig
from model.modeling_dream import DreamModel 

class ModelCompatWrapper(nn.Module):
    """
    Wrap a DreamModel (or other model that exposes dual_cache_generate/diffusion_generate)
    and

    This wrapper follows the same pattern as the Dream evaluation code.
    """
    def __init__(self, model_or_config: Any, tokenizer: Optional[Any] = None):
        # 必须先调用父类 nn.Module 的 __init__ 方法
        super().__init__()
        self._init_flag = True
        
        # 处理从配置对象进行的初始化（为了兼容 trl 的 create_reference_model）
        if isinstance(model_or_config, PretrainedConfig):
            # 如果传入的是配置对象，则用它来创建一个新的 DreamModel 实例
            model = DreamModel(model_or_config)
        else:
            # 否则，直接使用传入的模型实例
            model = model_or_config
        self.add_module("model", model)
        # 现在可以安全地为 model 和其他属性赋值
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
                    self._name_or_path = getattr(
                        model, "_name_or_path", "unknown_model"
                    )
                    self.model_type = getattr(model, "model_type", "custom")

            self.config = MinimalConfig()

        # Clear the initialization flag
        self._initializing = False

        # track requested gradient checkpointing state if called before model set
        self._gc_enabled_requested = False

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

    def __setattr__(self, name, value):
        """
        Intercept setting of the wrapped `model` so that if gradient_checkpointing
        was requested before the inner model existed, we can enable it once the
        model is assigned.
        """
        # For the very common case of setting attributes during initialization
        # or other simple attributes, delegate to the base implementation.
        # Handle 'model' specially so we can react when it's assigned.
        if name == "model":
            super().__setattr__(name, value)
            # If gradient checkpointing was requested earlier, apply it now
            try:
                if (
                    self.__dict__.get("_gc_enabled_requested", False)
                    and value is not None
                ):
                    if hasattr(value, "gradient_checkpointing_enable"):
                        value.gradient_checkpointing_enable()
                    # Clear the requested flag regardless so we don't re-run
                    self.__dict__["_gc_enabled_requested"] = False
            except Exception:
                # Be defensive: don't raise from __setattr__ -- just continue
                pass
            return

        super().__setattr__(name, value)

    def gradient_checkpointing_enable(self):
        """
        代理 gradient_checkpointing_enable 到底层模型，如果没有则延迟应用（不抛出异常）。
        如果在模型尚未初始化时被调用，则记录请求，并在模型赋值时执行。
        """
        # If still initializing or model not set, defer the request instead of raising
        if (
            self.__dict__.get("_initializing", False)
            or self.__dict__.get("model", None) is None
        ):
            # Record the request so __setattr__ can apply it later
            self.__dict__["_gc_enabled_requested"] = True
            return
        # If underlying model supports it, call through
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable()
        # Otherwise raise a clear error
        raise AttributeError(
            f"Underlying model does not support gradient_checkpointing_enable"
        )

    def get_parameter(self, target: str):
        """
        Delegate get_parameter calls to the inner model.
        """
        model = self.__dict__.get("model", None)
        if model is not None:
            try:
                # First, try to resolve the parameter from the wrapped model.
                if hasattr(model, "get_parameter"):
                    return model.get_parameter(target)
                else:
                    # Fallback for models without get_parameter: manually walk attributes
                    obj = model
                    for name in target.split("."):
                        obj = getattr(obj, name)
                    return obj
            except AttributeError:
                # If not found in the wrapped model, it might be on the wrapper itself.
                pass

        # Fallback to the default implementation for the wrapper's own parameters
        # or when the model is not yet set.
        return super().get_parameter(target)

    def get_submodule(self, target: str):
        """
        Delegate get_submodule lookups to the wrapped model.
        """
        model = self.__dict__.get("model", None)
        if model is not None:
            try:
                # First, try to resolve the submodule from the wrapped model.
                if hasattr(model, "get_submodule"):
                    return model.get_submodule(target)
                else:
                    # Fallback for models without get_submodule: manually walk attributes
                    mod = model
                    for name in target.split("."):
                        mod = getattr(mod, name)
                    return mod
            except AttributeError:
                # If not found in the wrapped model, it might be on the wrapper itself.
                pass
        
        # Fallback to the default implementation for the wrapper's own attributes
        # or when the model is not yet set.
        return super().get_submodule(target)

    # Provide attribute passthrough for convenience (so external code can still access model.*)
    def __getattr__(self, name: str):
        # During initialization, just raise AttributeError for unknown attributes
        if self.__dict__.get("_initializing", False):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # After initialization, check if we have the model using __dict__ to avoid recursion
        model_instance = self.__dict__.get("model", None)
        if model_instance is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' (model not initialized)"
            )

        # Check if the internal model has this attribute
        if hasattr(model_instance, name):
            return getattr(model_instance, name)

        # If the internal model doesn't have this attribute either, raise error
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
