# src/open_r1/utils/dream_compat.py
"""
Compatibility wrapper for Dream-style models so they behave like the "old" Dream models
expected by other code (e.g. TRL create_reference_model). This wrapper ensures that the
underlying `model` attribute is visible in the instance __dict__ and that attribute / submodule
lookup delegates to the wrapped model when appropriate.

Primary fixes:
- always place the underlying model into instance __dict__ using object.__setattr__
  to avoid custom __setattr__ interception causing 'no attribute model' errors.
- make get_submodule / get_parameter / __getattr__ robust and delegate to wrapped model.
- make __setattr__ defensive so setting 'model' rebinds generation methods safely.
"""

from __future__ import annotations

import types
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MinimalConfig:
    """
    Minimal config object used when wrapped model doesn't expose a config attribute.
    Feel free to extend fields if other parts of code expect more attributes.
    """
    use_return_dict: bool = True
    model_type: str = "dream_compat_minimal"
    # Add fields later as needed by callers.


class ModelCompatWrapper(nn.Module):
    """
    A wrapper around an nn.Module implementing a Dream-compatible surface:
    - exposes 'model' in instance __dict__ immediately
    - delegates get_submodule / get_parameter / attribute access to wrapped model if present
    - safe __setattr__ handling for 'model' assignment to rebind generation methods
    """

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        # ensure nn.Module init runs
        super().__init__()

        # Put the underlying model directly into the instance __dict__ to make it
        # visible to code that uses getattr(self, "model") or expects `model` attribute.
        object.__setattr__(self, "model", model)

        # Optionally attach tokenizer in the same safe way
        if tokenizer is not None:
            object.__setattr__(self, "tokenizer", tokenizer)

        # Expose config / generation_config as convenience attributes if available.
        # Use minimal fallbacks to avoid AttributeError when other code expects these.
        try:
            if hasattr(model, "generation_config"):
                object.__setattr__(self, "generation_config", getattr(model, "generation_config"))
        except Exception:
            pass

        try:
            if hasattr(model, "config"):
                object.__setattr__(self, "config", getattr(model, "config"))
            else:
                object.__setattr__(self, "config", MinimalConfig())
        except Exception:
            object.__setattr__(self, "config", MinimalConfig())

        # Track if we attempted to bind Dream-generation mixin methods already.
        object.__setattr__(self, "_dream_methods_bound", False)

        # Try to bind generation related helpers if present on the wrapped model
        try:
            self._bind_dream_methods()
        except Exception:
            # Don't fail initialization if binding fails; it may be bound later when model is set.
            logger.debug("Initial _bind_dream_methods failed; will try again when model set.", exc_info=True)

    # -------------------------
    # Helper binding / rebind
    # -------------------------
    def _bind_dream_methods(self, use_cache: bool = False):
        """
        Attempt to rebind or expose generation helper methods from underlying model, if they exist.
        This method should keep failing non-fatally if wrapped model doesn't have expected APIs.
        """
        model = self.__dict__.get("model", None)
        if model is None:
            return

        # Example: if the underlying model has DreamGenerationMixin methods, expose them at wrapper-level.
        # We don't know all methods; safely copy a couple of commonly used ones if present.
        for name in ("generate", "prepare_inputs_for_generation", "update_model_kwargs_for_generation"):
            if hasattr(model, name):
                try:
                    # bind the function so calls like wrapper.generate(...) work and self points to model
                    fn = getattr(model, name)
                    # If it's a function defined on the model instance, just assign a wrapper-bound method
                    object.__setattr__(self, name, types.MethodType(fn.__func__ if hasattr(fn, "__func__") else fn, model))
                except Exception:
                    # best-effort; do not raise
                    logger.debug("Could not bind %s from underlying model", name, exc_info=True)

        # Mark as attempted
        object.__setattr__(self, "_dream_methods_bound", True)

    def rebind_generation_methods(self, use_cache: bool = False):
        """Public wrapper to rebind generation-related methods after model replacement."""
        self._bind_dream_methods(use_cache=use_cache)

    # -------------------------
    # Delegation helpers
    # -------------------------
    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """
        Try to return a parameter by dotted path. Prefer delegating to the wrapped model
        when possible; fall back to wrapper's own get_parameter behavior.
        """
        model = self.__dict__.get("model", None)

        if model is not None:
            # If the wrapped model itself implements get_parameter, prefer delegating
            if hasattr(model, "get_parameter"):
                try:
                    return model.get_parameter(target)
                except Exception:
                    # fall back to manual walk below
                    logger.debug("Delegated model.get_parameter failed for %s, falling back", target, exc_info=True)

            # Manual dotted-path walk on wrapped model: e.g. "decoder.layer.0.attn.q_proj.weight"
            parts = target.split(".")
            obj = model
            try:
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                # last part should be a parameter attribute
                last = parts[-1]
                param = getattr(obj, last)
                return param
            except Exception as e:
                logger.debug("Manual get_parameter walk failed for %s: %s", target, e, exc_info=True)

        # Fallback: try wrapper's own get_parameter (nn.Module API) to raise the usual errors
        return super().get_parameter(target)

    def get_submodule(self, target: str):
        """
        Resolve a submodule name. Prefer resolving on the wrapped model, but be robust:
        - If target starts with 'model' or 'model.' try to delegate to wrapped model.
        - If wrapped model can't resolve, fall back to wrapper's own submodules or attribute-walk.
        """
        model = self.__dict__.get("model", None)

        # If the request explicitly targets 'model' or starts with 'model.', delegate to model
        if target == "model" or target.startswith("model."):
            # remove leading 'model.' if present
            sub_target = target if target == "model" else target.split(".", 1)[1]
            if model is not None:
                # If model exposes get_submodule, use it
                if hasattr(model, "get_submodule"):
                    try:
                        # If target was exactly "model", return the wrapped model
                        if sub_target == "model" or sub_target == "":
                            return model
                        return model.get_submodule(sub_target)
                    except Exception:
                        logger.debug("model.get_submodule failed for %s, trying manual walk", sub_target, exc_info=True)

                # Manual attribute walk
                try:
                    if sub_target == "model" or sub_target == "":
                        return model
                    cur = model
                    for part in sub_target.split("."):
                        cur = getattr(cur, part)
                    return cur
                except Exception:
                    logger.debug("Manual walk on wrapped model failed for %s", sub_target, exc_info=True)

            # If model not set, fall through to wrapper-based resolution

        # Try default nn.Module resolution (this will find wrapper's own submodules)
        try:
            return super().get_submodule(target)
        except Exception:
            # As last resort, do manual attribute walk on wrapper
            try:
                cur = self
                for part in target.split("."):
                    cur = getattr(cur, part)
                return cur
            except Exception as e:
                # Re-raise AttributeError consistent with nn.Module.get_submodule behaviour
                raise AttributeError(f"{self.__class__.__name__} has no submodule {target}") from e

    # -------------------------
    # Attribute access / setting
    # -------------------------
    def __setattr__(self, name: str, value: Any):
        """
        Special-case setting 'model' to ensure the underlying model is placed into __dict__
        directly and that we attempt to rebind generation functions. Defensive: never raise
        from __setattr__ so that external code setting attributes doesn't crash this object.
        """
        try:
            if name == "model":
                # Place model directly into instance __dict__ to ensure visibility
                object.__setattr__(self, "model", value)
                # Update related attributes (config/generation_config) if present on new model
                try:
                    if hasattr(value, "config"):
                        object.__setattr__(self, "config", getattr(value, "config"))
                    else:
                        object.__setattr__(self, "config", MinimalConfig())
                except Exception:
                    pass

                try:
                    if hasattr(value, "generation_config"):
                        object.__setattr__(self, "generation_config", getattr(value, "generation_config"))
                except Exception:
                    pass

                # Try to rebind generation methods from the new model
                try:
                    self._bind_dream_methods()
                except Exception:
                    logger.debug("Rebinding dream methods after model set failed.", exc_info=True)
                return

            # For other attributes, use default behaviour (keeps nn.Module behaviour intact)
            super().__setattr__(name, value)
        except Exception:
            # Last-resort: write into __dict__ directly (defensive, avoid crashing)
            try:
                object.__setattr__(self, name, value)
            except Exception:
                # swallow to avoid breaking callers; but log for diagnostics
                logger.exception("Failed to set attribute %s on ModelCompatWrapper", name)

    def __getattr__(self, name: str) -> Any:
        """
        When attribute lookup fails on wrapper, try delegated lookup on the wrapped model.
        This allows calls like wrapper.some_model_method(...) to work transparently.
        """
        # Standard behaviour: if attribute isn't found on wrapper, try wrapped model
        model = self.__dict__.get("model", None)
        if model is not None:
            try:
                return getattr(model, name)
            except AttributeError:
                pass
            except Exception:
                # Other exceptions from the underlying getattr should not crash here
                logger.debug("Delegated getattr raised for %s", name, exc_info=True)

        # If still not found, raise AttributeError (consistent with normal behaviour)
        raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")

    # -------------------------
    # nn.Module convenience overrides (delegation)
    # -------------------------
    @property
    def device(self):
        # Prefer to report the wrapped model's device if available
        model = self.__dict__.get("model", None)
        if model is not None:
            try:
                # Many models expose parameters; get device from first parameter
                for p in model.parameters():
                    return p.device
            except Exception:
                pass
        # Fall back to wrapper params (if any)
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        # Delegate 'to' to wrapped model if present, else to super
        model = self.__dict__.get("model", None)
        if model is not None:
            try:
                model = model.to(*args, **kwargs)
                # ensure __dict__ keeps the (possibly new) model object
                object.__setattr__(self, "model", model)
                return self
            except Exception:
                logger.debug("Delegated model.to failed; falling back to wrapper.to", exc_info=True)
        return super().to(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        model = self.__dict__.get("model", None)
        if model is not None:
            try:
                return model.state_dict(*args, **kwargs)
            except Exception:
                logger.debug("model.state_dict failed - falling back to super().state_dict", exc_info=True)
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        model = self.__dict__.get("model", None)
        if model is not None:
            try:
                return model.load_state_dict(state_dict, *args, **kwargs)
            except Exception:
                logger.debug("model.load_state_dict failed - falling back to super().load_state_dict", exc_info=True)
        return super().load_state_dict(state_dict, *args, **kwargs)

    # Add any additional delegations expected by the rest of the codebase as needed.
