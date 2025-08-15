# src/open_r1/utils/dream_compat.py
"""
Compatibility wrapper for Dream-style models so they behave like the "old" Dream models
expected by other code (e.g. TRL create_reference_model). This wrapper ensures that the
underlying `model` attribute is visible and, crucially, registered as an nn.Module submodule
so parameters are discoverable by optimizers / DeepSpeed / accelerate.
"""

from __future__ import annotations

import types
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MinimalConfig:
    use_return_dict: bool = True
    model_type: str = "dream_compat_minimal"


class ModelCompatWrapper(nn.Module):
    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        super().__init__()

        # IMPORTANT: register as a submodule so that nn.Module machinery (parameters(),
        # named_parameters(), etc.) discovers the wrapped model's parameters.
        # Using super().__setattr__ ensures PyTorch registers the child module.
        try:
            super().__setattr__("model", model)
        except Exception:
            # fallback (shouldn't normally happen)
            object.__setattr__(self, "model", model)

        # register tokenizer if provided (tokenizer is usually not an nn.Module,
        # so regular setattr is fine)
        if tokenizer is not None:
            try:
                super().__setattr__("tokenizer", tokenizer)
            except Exception:
                object.__setattr__(self, "tokenizer", tokenizer)

        # Expose config/generation_config if present; fallback to MinimalConfig
        try:
            if hasattr(model, "generation_config"):
                super().__setattr__("generation_config", getattr(model, "generation_config"))
        except Exception:
            pass

        try:
            if hasattr(model, "config"):
                super().__setattr__("config", getattr(model, "config"))
            else:
                super().__setattr__("config", MinimalConfig())
        except Exception:
            super().__setattr__("config", MinimalConfig())

        # mark binding flag
        super().__setattr__("_dream_methods_bound", False)

        # Try to bind generation helpers
        try:
            self._bind_dream_methods()
        except Exception:
            logger.debug("Initial _bind_dream_methods failed; will try again when model set.", exc_info=True)

    # -------------------------
    # Helper binding / rebind
    # -------------------------
    def _bind_dream_methods(self, use_cache: bool = False):
        model = getattr(self, "model", None)
        if model is None:
            return

        for name in ("generate", "prepare_inputs_for_generation", "update_model_kwargs_for_generation"):
            if hasattr(model, name):
                try:
                    fn = getattr(model, name)
                    # bind so method calls behave like model's method (self=model)
                    object.__setattr__(self, name, types.MethodType(fn.__func__ if hasattr(fn, "__func__") else fn, model))
                except Exception:
                    logger.debug("Could not bind %s from underlying model", name, exc_info=True)

        object.__setattr__(self, "_dream_methods_bound", True)

    def rebind_generation_methods(self, use_cache: bool = False):
        self._bind_dream_methods(use_cache=use_cache)

    # -------------------------
    # Delegation helpers
    # -------------------------
    def get_parameter(self, target: str) -> torch.nn.Parameter:
        model = getattr(self, "model", None)

        if model is not None:
            if hasattr(model, "get_parameter"):
                try:
                    return model.get_parameter(target)
                except Exception:
                    logger.debug("Delegated model.get_parameter failed for %s", target, exc_info=True)

            # Manual walk
            parts = target.split(".")
            obj = model
            try:
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                last = parts[-1]
                return getattr(obj, last)
            except Exception:
                logger.debug("Manual get_parameter walk failed for %s", target, exc_info=True)

        return super().get_parameter(target)

    def get_submodule(self, target: str):
        model = getattr(self, "model", None)

        if target == "model" or target.startswith("model."):
            sub_target = target if target == "model" else target.split(".", 1)[1]
            if model is not None:
                if hasattr(model, "get_submodule"):
                    try:
                        if sub_target == "model" or sub_target == "":
                            return model
                        return model.get_submodule(sub_target)
                    except Exception:
                        logger.debug("model.get_submodule failed for %s, trying manual walk", sub_target, exc_info=True)
                try:
                    if sub_target == "model" or sub_target == "":
                        return model
                    cur = model
                    for part in sub_target.split("."):
                        cur = getattr(cur, part)
                    return cur
                except Exception:
                    logger.debug("Manual walk on wrapped model failed for %s", sub_target, exc_info=True)

        try:
            return super().get_submodule(target)
        except Exception:
            try:
                cur = self
                for part in target.split("."):
                    cur = getattr(cur, part)
                return cur
            except Exception as e:
                raise AttributeError(f"{self.__class__.__name__} has no submodule {target}") from e

    # -------------------------
    # Attribute access / setting
    # -------------------------
    def __setattr__(self, name: str, value: Any):
        try:
            if name == "model":
                # Use super().__setattr__ to register submodule properly
                super().__setattr__(name, value)

                # update config/generation_config if available
                try:
                    if hasattr(value, "config"):
                        super().__setattr__("config", getattr(value, "config"))
                    else:
                        super().__setattr__("config", MinimalConfig())
                except Exception:
                    pass

                try:
                    if hasattr(value, "generation_config"):
                        super().__setattr__("generation_config", getattr(value, "generation_config"))
                except Exception:
                    pass

                # rebind helper methods
                try:
                    self._bind_dream_methods()
                except Exception:
                    logger.debug("Rebinding dream methods after model set failed.", exc_info=True)
                return

            # default behaviour keeps nn.Module registration for other submodules
            super().__setattr__(name, value)
        except Exception:
            try:
                object.__setattr__(self, name, value)
            except Exception:
                logger.exception("Failed to set attribute %s on ModelCompatWrapper", name)

    def __getattr__(self, name: str) -> Any:
        model = getattr(self, "model", None)
        if model is not None:
            try:
                return getattr(model, name)
            except AttributeError:
                pass
            except Exception:
                logger.debug("Delegated getattr raised for %s", name, exc_info=True)

        raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")

    # -------------------------
    # nn.Module convenience overrides (delegation)
    # -------------------------
    @property
    def device(self):
        model = getattr(self, "model", None)
        if model is not None:
            try:
                for p in model.parameters():
                    return p.device
            except Exception:
                pass
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        model = getattr(self, "model", None)
        if model is not None:
            try:
                model = model.to(*args, **kwargs)
                # ensure the registered submodule references updated model instance
                super().__setattr__("model", model)
                return self
            except Exception:
                logger.debug("Delegated model.to failed; falling back to wrapper.to", exc_info=True)
        return super().to(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        model = getattr(self, "model", None)
        if model is not None:
            try:
                return model.state_dict(*args, **kwargs)
            except Exception:
                logger.debug("model.state_dict failed - falling back to super().state_dict", exc_info=True)
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        model = getattr(self, "model", None)
        if model is not None:
            try:
                return model.load_state_dict(state_dict, *args, **kwargs)
            except Exception:
                logger.debug("model.load_state_dict failed - falling back to super().load_state_dict", exc_info=True)
        return super().load_state_dict(state_dict, *args, **kwargs)
