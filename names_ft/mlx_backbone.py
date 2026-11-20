"""Run a fine-tuned model's backbone (Gemma 4 or Qwen3.5) with MLX on Apple silicon (`uv sync --group mlx`).

PyTorch on the Mac GPU (MPS) reaches about a third of the GPU's matmul speed on this model; MLX is Apple's own
array framework. Only the backbone moves to MLX: mlx-lm's model code reads the base weights straight from the
Hugging Face cache (the same files PyTorch uses), and the fine-tuned LoRA is folded into them at load time
(W + alpha/r * B @ A, as PEFT's merge does), so there is no converted copy on disk. The extraction heads and
decoding stay in PyTorch, on the CPU: they take a few percent of the time.

The instructions and field list before the name are run once and shared, as in core.SharedPrefix: the prefix's
keys and values are recorded once, and each batch attends to them without storing anything. Qwen3.5's
linear-attention layers (Gated DeltaNet) keep a conv state and a recurrent state instead of keys and values;
mlx-lm continues a multi-token batch from them, so each batch starts from the prefix's final states.

mlx-lm's Gemma 4 and Qwen3.5 modules use transformers' names and weight layouts for every linear layer a LoRA
targets (q/k/v/o_proj, in_proj_qkv/z/a/b, out_proj, gate/up/down_proj), so the LoRA folds in unchanged. What
mlx-lm lays out differently for Qwen3.5 on load (conv1d axes, norm weights stored minus one) holds no LoRA.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import ArraysCache
from mlx_lm.utils import _get_classes, load_model
from safetensors.torch import load_file

from turion_extract.layout import Layout
from turion_extract.model import ExtractHeads, load_tokenizer

from .core import Item, SharedPrefix, predict_with


class _Record:
    """Cache stand-in for the prefix pass: keeps each layer's keys and values."""
    offset = 0

    def update_and_fetch(self, keys, values):
        self.keys, self.values = keys, values
        return keys, values

    def make_mask(self, n, return_array=False, window_size=None):
        return create_causal_mask(n, 0, window_size=window_size)


class _Prefix:
    """Cache stand-in for a batch: positions start after the prefix, and attention sees the prefix's keys and
    values (broadcast over the batch) followed by the batch's own. Nothing is stored, so one serves every batch."""

    def __init__(self, keys, values):
        self.keys, self.values, self.offset = keys, values, keys.shape[2]

    def update_and_fetch(self, keys, values):
        b = keys.shape[0]
        return (mx.concatenate([mx.broadcast_to(self.keys, (b, *self.keys.shape[1:])), keys], axis=2),
                mx.concatenate([mx.broadcast_to(self.values, (b, *self.values.shape[1:])), values], axis=2))

    def make_mask(self, n, return_array=False, window_size=None):
        return create_causal_mask(n, self.offset, window_size=window_size)

    def batch(self, b: int) -> "_Prefix":
        return self


class _LinearPrefix:
    """Cache stand-in for a batch at a linear-attention layer, in place of mlx-lm's ArraysCache(size=2): [0] is
    the conv state (the prefix's last kernel-1 conv inputs, [1, kernel-1, channels]) and [1] the recurrent state
    ([1, value heads, value dim, key dim], fp32) after the prefix. The layer concatenates the conv state in front of
    the batch's conv inputs and starts the recurrence from the state; what it writes back is dropped.

    `batch(b)` gives a view broadcast to the batch. The recurrence then makes one state per item for the layer
    (Qwen3.5-4B: 32 x 128 x 128 fp32 = 2 MB per item, in and out), freed before the next layer runs: about 150 MB
    for a batch of 37 (4096 tokens), not 24 layers' worth. No padding mask is needed (and none is given, which
    keeps mlx-lm's faster kernel): pads come after the real tokens and both conv and recurrence are causal.
    """
    lengths = None

    def __init__(self, conv, state):
        self.arrays = [conv, state]

    def batch(self, b: int) -> "_LinearPrefix":
        return _LinearPrefix(*[mx.broadcast_to(a, (b, *a.shape[1:])) for a in self.arrays])

    def __getitem__(self, i):
        return self.arrays[i]

    def __setitem__(self, i, value):
        pass

    def advance(self, n):
        pass

    def make_mask(self, n):
        return None


def record_prefix(lm, fresh_cache: list, ids: list[int]):
    """Run the prefix once. Returns its hidden states [n, d] and one cache stand-in per layer; pass
    `[c.batch(b) for c in them]` as the cache of any batch of b requests' remaining tokens.
    `fresh_cache`: the model's make_cache(), read only for which layers are linear attention."""
    rec = [ArraysCache(size=2) if isinstance(c, ArraysCache) else _Record() for c in fresh_cache]
    h = lm(mx.array([ids]), cache=rec)
    shared = [_LinearPrefix(c[0], c[1]) if isinstance(c, ArraysCache) else _Prefix(c.keys, c.values) for c in rec]
    mx.eval(h, *[c.arrays if isinstance(c, _LinearPrefix) else (c.keys, c.values) for c in shared])
    return h[0], shared


def _lora_target(key: str) -> str:
    """PEFT key -> mlx-lm parameter name. The decoder is `model.language_model` in transformers' Gemma 4 (and
    Qwen3.5's image-text class) and `model` in Qwen3_5ForCausalLM; mlx-lm calls it `language_model.model` in both:
    base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight          (Gemma 4)
    base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight               (Qwen3.5)
      -> language_model.model.layers.0.<module>.weight"""
    k = re.sub(r"\.lora_[AB]\.weight$", "", key).removeprefix("base_model.model.").removeprefix("model.")
    return "language_model.model." + k.removeprefix("language_model.") + ".weight"


def _model_config(path: Path) -> dict | None:
    """A text-only Qwen3.5 config (as Qwen3_5ForCausalLM saves it) names a model type mlx-lm has no module for;
    its qwen3_5 module reads such a config."""
    with open(path / "config.json") as f:
        model_type = json.load(f).get("model_type")
    return {"model_type": "qwen3_5"} if model_type == "qwen3_5_text" else None


def _model_classes(config: dict):
    """mlx-lm's classes for the config. For Qwen3.5 with tied embeddings, a checkpoint that still stores
    lm_head.weight (a fine-tune saved untied) would fail mlx-lm's strict load: that weight is dropped here,
    since only the decoder is used. Every other weight is still checked strictly."""
    model_class, args_class = _get_classes(config)
    if config.get("model_type") != "qwen3_5":
        return model_class, args_class

    class Decoder(model_class):
        def sanitize(self, weights):
            weights = super().sanitize(weights)
            if self.language_model.args.tie_word_embeddings:
                weights.pop("language_model.lm_head.weight", None)
            return weights

    return Decoder, args_class


class MLXExtractor:
    """A names_ft model directory (best/) with the backbone in MLX: .layout, .tok, .heads and .predict(items)."""

    def __init__(self, model_dir: str, max_len: int | None = None, dtype=None):
        """dtype: an mx dtype to cast the base weights to before the LoRA is folded in (mx.float32, to compare
        with PyTorch closely); default: as stored (bf16)."""
        with open(os.path.join(model_dir, "turion_extract.json")) as f:
            cfg = json.load(f)
        self.tok = load_tokenizer(model_dir)
        self.layout = Layout(self.tok, max_len=max_len or cfg.get("max_len", 16384))
        self.heads = ExtractHeads(**cfg["heads"])
        self.heads.load_state_dict(load_file(os.path.join(model_dir, "heads.safetensors")))
        self.heads.eval()

        base = cfg["base_model"]
        path = Path(base) if os.path.isdir(base) else Path(snapshot_download(base, allow_patterns=["*.json", "*.safetensors"]))
        model, _ = load_model(path, lazy=True, model_config=_model_config(path), get_model_classes=_model_classes)
        if dtype is not None:
            model.update(tree_map(lambda p: p.astype(dtype) if mx.issubdtype(p.dtype, mx.floating) else p,
                                  model.parameters()))
        if cfg.get("adapter"):
            self._merge_lora(model, model_dir)
        mx.eval(model.parameters())
        self.lm = model.language_model.model             # the decoder: returns the normed last hidden states
        # one entry per layer that keeps state: keys/values (not Gemma's KV-shared layers) or linear-attention state
        self.make_cache = model.make_cache

    @staticmethod
    def _merge_lora(model, model_dir: str) -> None:
        with open(os.path.join(model_dir, "adapter_config.json")) as f:
            ac = json.load(f)
        if ac.get("use_dora") or ac.get("use_rslora") or ac.get("rank_pattern") or ac.get("alpha_pattern"):
            raise ValueError("only plain LoRA adapters can be merged here")
        scale = ac["lora_alpha"] / ac["r"]
        lora = load_file(os.path.join(model_dir, "adapter_model.safetensors"))
        params = dict(tree_flatten(model.parameters()))
        merged = {}
        for key in lora:
            if not key.endswith(".lora_A.weight"):
                continue
            name = _lora_target(key)
            if name not in params:
                raise KeyError(f"adapter weight {key} has no MLX parameter {name}")
            a, b = lora[key].float(), lora[key.replace(".lora_A.", ".lora_B.")].float()
            w = params[name]
            if w.shape != (b.shape[0], a.shape[1]):       # mlx-lm fused, split or re-laid out this weight
                raise ValueError(f"adapter weight {key} is for a {(b.shape[0], a.shape[1])} matrix; {name} is {w.shape}")
            # B @ A in PyTorch on the CPU: MLX's GPU matmul keeps only about 3 digits (mlx 0.32.4)
            merged[name] = (w.astype(mx.float32) + mx.array((scale * (b @ a)).numpy())).astype(w.dtype)
        if len(merged) * 2 != len(lora):
            raise ValueError(f"merged {len(merged)} LoRA pairs from {len(lora)} adapter tensors")
        model.load_weights(list(merged.items()), strict=False)

    def _hidden(self, items: list[Item], cache, skip: int, pad_id: int) -> torch.Tensor:
        n = max(it.n_tokens for it in items) - skip
        ids = np.full((len(items), n), pad_id, dtype=np.int32)
        for b, it in enumerate(items):                    # right-padded: causal attention keeps pads out of real tokens
            ids[b, :it.n_tokens - skip] = it.enc.input_ids[skip:]
        h = self.lm(mx.array(ids), cache=cache)
        return torch.from_numpy(np.array(h.astype(mx.float32)))

    @torch.no_grad()
    def predict(self, items: list[Item], micro_tokens: int = 32768, share_prefix: bool = True):
        """As core.predict: answers per row id, and the mean field loss over items with targets."""
        pad = self.tok.pad_token_id
        ids = SharedPrefix.common(items) if share_prefix else None
        if not ids:
            return predict_with(lambda mb: self._hidden(mb, None, 0, pad), self.heads, items, micro_tokens)
        prefix_h, shared = record_prefix(self.lm, self.make_cache(), ids)
        prefix_h = torch.from_numpy(np.array(prefix_h.astype(mx.float32)))

        def hidden(mb):
            h = self._hidden(mb, [c.batch(len(mb)) for c in shared], len(ids), pad)
            return torch.cat([prefix_h.unsqueeze(0).expand(len(mb), -1, -1), h], dim=1)

        return predict_with(hidden, self.heads, items, micro_tokens, skip=len(ids))
