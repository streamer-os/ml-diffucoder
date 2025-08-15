# src/open_r1/utils/dream_compat.py
import torch
from typing import Any, Optional
import torch.nn as nn

class ModelCompatWrapper(nn.Module):
    """
    Wrap a DreamModel (or other model that exposes dual_cache_generate/diffusion_generate)
    and provide a minimal compatibility layer expected by the trainer codebase.
    """

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        # Expose generation_config if model has it
        self.generation_config = getattr(model, "generation_config", None)

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

    def diffusion_generate(self, input_ids, attention_mask=None, **kwargs):
        """
        Compatibility entrypoint used by trainer generation code.

        It:
        - prefers dream-style dual_cache_generate if available,
        - falls back to diffusion_generate if present,
        - otherwise error.
        kwargs will be forwarded as-is (so pass steps, block_length, dual_cache, replace_position, top_p, top_k, temperature, etc).
        """
        # Normalize inputs
        attn = attention_mask

        # If Dream API exists, call its dual_cache_generate (or block generate)
        if hasattr(self.model, "dual_cache_generate"):
            return self.model.dual_cache_generate(
                input_ids,
                attention_mask=attn,
                **kwargs,
            )

        # Fallback: some models already implement diffusion_generate
        if hasattr(self.model, "diffusion_generate"):
            return self.model.diffusion_generate(
                input_ids,
                attention_mask=attn,
                **kwargs,
            )

        # Last resort: try generic .generate if it implements some behavior
        if hasattr(self.model, "generate"):
            # We try to map plausible kwargs for generate
            gen_kwargs = {}
            # common mapping
            for k in ("max_length", "temperature", "top_p", "top_k", "do_sample"):
                if k in kwargs:
                    gen_kwargs[k] = kwargs[k]
            return self.model.generate(input_ids, attention_mask=attn, **gen_kwargs)

        raise RuntimeError(
            "Wrapped model has no dual_cache_generate/diffusion_generate/generate method. "
            "Make sure you imported the Dream model files and wrapped them with ModelCompatWrapper."
        )

    # Provide attribute passthrough for convenience (so external code can still access model.*)
    def __getattr__(self, name: str):
        # ensure Python doesn't recurse
        if name in ("model", "tokenizer", "generation_config"):
            return super().__getattribute__(name)
        return getattr(self.model, name)
