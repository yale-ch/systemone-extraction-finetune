"""A tiny random Gemma 4 (same architecture, layer types and tokenizer as gemma-4-12B-it) for CPU tests.

The sliding window is shrunk to 16 tokens so that padded batches cross it, which the real model (window 1024)
never does on ~270-token name requests: if padding is handled correctly here, it is there too.
"""
from __future__ import annotations

import copy
import json
import os

import torch

GEMMA_12B_TEXT = {
    "attention_bias": False, "attention_dropout": 0.0, "attention_k_eq_v": True, "bos_token_id": 2,
    "enable_moe_block": False, "eos_token_id": 1, "final_logit_softcapping": 30.0, "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size_per_layer_input": 0, "initializer_range": 0.02, "max_position_embeddings": 262144,
    "model_type": "gemma4_unified_text", "num_kv_shared_layers": 0, "pad_token_id": 0, "rms_norm_eps": 1e-06,
    "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "tie_word_embeddings": True, "use_bidirectional_attention": "vision", "use_double_wide_mlp": False,
    "vocab_size": 262144, "vocab_size_per_layer_input": 262144,
}


def make_tiny_gemma(path: str, seed: int = 0, tokenizer_from: str | None = None) -> str:
    """tokenizer_from: a directory whose tokenizer files to copy in (the released model), so the tiny model works
    as a `--base` on its own."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if os.path.exists(os.path.join(path, "config.json")):
        return path
    text = dict(copy.deepcopy(GEMMA_12B_TEXT), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                layer_types=["sliding_attention"] * 5 + ["full_attention"], num_attention_heads=2,
                num_key_value_heads=1, num_global_key_value_heads=1, head_dim=32, global_head_dim=64, sliding_window=16)
    cfg = {"architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified",
           "text_config": text, "tie_word_embeddings": True, "eos_token_id": [1, 106],
           "vision_config": {"model_type": "gemma4_unified_vision", "mm_embed_dim": 64, "output_proj_dims": 64},
           "audio_config": {"model_type": "gemma4_unified_audio", "hidden_size": 32, "audio_embed_dim": 32,
                            "output_proj_dims": 32}}
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.save_pretrained(path)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if tokenizer_from and os.path.exists(os.path.join(tokenizer_from, f)):
            with open(os.path.join(tokenizer_from, f), "rb") as src, open(os.path.join(path, f), "wb") as dst:
                dst.write(src.read())
    return path
