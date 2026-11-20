"""A tiny random Qwen3.5 (same hybrid architecture and tokenizer as Qwen/Qwen3.5-4B) for CPU tests.

Qwen3.5 alternates three linear-attention layers (Gated DeltaNet: a causal conv and a recurrent state instead of
keys and values) with one full-attention layer. The tiny model keeps that pattern: 3 linear + 1 full, twice.
The tokenizer files (~20 MB) come from the Hub on first use and are copied into the model directory, so the
directory works as a `--base` on its own.
"""
from __future__ import annotations

import copy
import json
import os

import torch

QWEN35_4B_TEXT = {
    "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True, "eos_token_id": 248044,
    "full_attention_interval": 4, "hidden_act": "silu", "initializer_range": 0.02, "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 262144, "mlp_only_layers": [], "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 0,
    "rms_norm_eps": 1e-06, "tie_word_embeddings": True, "vocab_size": 248320, "mamba_ssm_dtype": "float32",
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default",
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25},
}
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"]


def make_tiny_qwen(path: str, seed: int = 0) -> str:
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM

    if os.path.exists(os.path.join(path, "config.json")):
        return path
    os.makedirs(path, exist_ok=True)
    for f in TOKENIZER_FILES:
        with open(hf_hub_download("Qwen/Qwen3.5-4B", f), "rb") as src, open(os.path.join(path, f), "wb") as dst:
            dst.write(src.read())
    text = dict(copy.deepcopy(QWEN35_4B_TEXT), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2, num_attention_heads=2,
                num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16)
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": text,
           "tie_word_embeddings": True,
           "vision_config": {"model_type": "qwen3_5", "depth": 1, "hidden_size": 32, "intermediate_size": 64,
                             "num_heads": 2, "out_hidden_size": 64}}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), dtype=torch.bfloat16)
    model.save_pretrained(path)
    return path
