# src/open_r1/utils/model_utils.py
import logging
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Optional

logger = logging.getLogger(__name__)

# keep existing helper functions above (if any) unchanged
# Replace or add this get_model function
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig

def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer
    
def get_model(model_args, training_args):
    """
    Try to load DreamModel (from model/). If succeeds, wrap with ModelCompatWrapper.
    Otherwise fall back to AutoModel.from_pretrained (original behavior).
    """
    # choose dtype
    torch_dtype = None
    if getattr(model_args, "torch_dtype", None):
        # model_args might contain a string like "bfloat16"
        td = model_args.torch_dtype
        if isinstance(td, str):
            torch_dtype = getattr(torch, td, None)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if getattr(training_args, "bf16", False) else None

    # Try DreamModel import (model/ must be in PYTHONPATH or installed)
    try:
        from .....model.modeling_dream import DreamModel
        from .....model.configuration_dream import DreamConfig
        # compat wrapper
        from .dream_compat import ModelCompatWrapper

        logger.info("Loading DreamModel from model/ directory (compat wrapper).")
        # load config then model
        try:
            config = DreamConfig.from_pretrained(model_args.model_name_or_path)
        except Exception:
            # if path is not a pretrained dir, try default config
            config = DreamConfig()

        # prepare kwargs for from_pretrained
        fp_kwargs = dict(
            config=config,
            trust_remote_code=True,
        )
        if torch_dtype is not None:
            fp_kwargs["torch_dtype"] = torch_dtype

        model = DreamModel.from_pretrained(model_args.model_name_or_path, **fp_kwargs)
        # wrap
        wrapped = ModelCompatWrapper(model)
        return wrapped

    except Exception as e:
        # fallback to standard AutoModel
        logger.warning("Could not load DreamModel (will fallback to AutoModel). Exception: %s", e)
        # original logic (simplified)
        auto_kwargs = {}
        if torch_dtype is not None:
            auto_kwargs["torch_dtype"] = torch_dtype
        auto_kwargs["trust_remote_code"] = getattr(model_args, "trust_remote_code", True)
        model = AutoModel.from_pretrained(model_args.model_name_or_path, **auto_kwargs)
        return model
