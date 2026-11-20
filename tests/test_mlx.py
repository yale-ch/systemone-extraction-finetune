"""The MLX backend's shared prefix gives the same hidden states as a full pass, on a tiny random Gemma 4 shaped like
the 12B (k=v full-attention layers, no KV sharing), with a sliding window shorter than the prefix. Mac only."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_map  # noqa: E402
from mlx_lm.models import gemma4_text  # noqa: E402

from names_ft.mlx_backbone import _lora_target, _Prefix, _Record  # noqa: E402


def test_lora_key_mapping():
    assert _lora_target("base_model.model.model.language_model.layers.7.self_attn.q_proj.lora_A.weight") == \
        "language_model.model.layers.7.self_attn.q_proj.weight"


def test_shared_prefix_matches_full_pass():
    mx.random.seed(0)
    args = gemma4_text.ModelArgs(hidden_size=64, num_hidden_layers=6, intermediate_size=128, num_attention_heads=4,
                                 head_dim=16, global_head_dim=32, num_key_value_heads=2, num_global_key_value_heads=1,
                                 vocab_size=512, vocab_size_per_layer_input=512, num_kv_shared_layers=0,
                                 hidden_size_per_layer_input=0, sliding_window=8, attention_k_eq_v=True,
                                 use_double_wide_mlp=False)
    model = gemma4_text.Model(args)
    model.update(tree_map(lambda p: p * 4, model.parameters()))      # larger weights: differences would show
    lm = model.model
    rng = np.random.default_rng(0)
    prefix = rng.integers(1, 512, 20).tolist()                        # longer than the window
    rests = [rng.integers(1, 512, n).tolist() for n in (3, 12, 7)]

    rec = [_Record() for _ in range(len(model.make_cache()))]
    ph = lm(mx.array([prefix]), cache=rec)
    cache = [_Prefix(c.keys, c.values) for c in rec]
    T = max(map(len, rests))
    ids = np.zeros((len(rests), T), dtype=np.int32)                   # right-padded with token 0
    for b, r in enumerate(rests):
        ids[b, :len(r)] = r
    hb = lm(mx.array(ids), cache=cache)
    hb2 = lm(mx.array(ids[:1]), cache=cache)                          # the cache is unchanged by use

    for b, r in enumerate(rests):
        full = lm(mx.array([prefix + r]))[0]
        assert np.allclose(np.array(ph[0]), np.array(full[:len(prefix)]), atol=1e-4)
        assert np.allclose(np.array(hb[b, :len(r)]), np.array(full[len(prefix):]), atol=1e-4)
    assert np.allclose(np.array(hb2[0, :len(rests[0])]), np.array(hb[0, :len(rests[0])]), atol=1e-5)
