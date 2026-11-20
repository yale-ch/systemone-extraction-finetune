"""Prefix sharing on a tiny random Qwen3.5 (three linear-attention layers, then one full-attention layer, twice):
each batch starts from the prefix's conv and recurrent states and gives the same hidden states as whole requests."""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
DATA = os.path.join(ROOT, "data", "names")
pytestmark = pytest.mark.skipif(not os.path.exists(DATA), reason="needs data/names")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from tiny_qwen import make_tiny_qwen

    from names_ft.core import encode_rows, read_jsonl
    from turion_extract.layout import Layout
    from turion_extract.model import get_backbone, load_model, load_tokenizer

    path = make_tiny_qwen(str(tmp_path_factory.mktemp("tiny-qwen35")))
    model = load_model(path, device="cpu", dtype=torch.float32).eval()
    with torch.no_grad():                       # the random init is small: larger weights make differences show
        for p in model.parameters():
            if p.ndim >= 2:
                p.mul_(3)
    layout = Layout(load_tokenizer(path), max_len=1024)
    items, _ = encode_rows(layout, read_jsonl(os.path.join(DATA, "dev.jsonl"), 40))
    return model, get_backbone(model), layout, items


def test_shared_prefix_matches_full_forward(setup):
    from transformers.cache_utils import LinearAttentionLayer

    from names_ft.core import SharedPrefix, hidden_states, micro_batches

    model, bb, layout, items = setup
    pad = layout.tok.pad_token_id
    ids = SharedPrefix.common(items)
    assert ids and len(ids) > 64                                        # more than one chunk of the delta rule
    with torch.no_grad():
        prefix = SharedPrefix(bb, ids, "cpu")
        linear = [l for l in prefix.cache.layers if isinstance(l, LinearAttentionLayer)]
        assert len(linear) == 6 and len(prefix.cache.layers) == 8
        before = [(l.conv_states[0].clone(), l.recurrent_states[0].clone()) for l in linear]
        batches = micro_batches(sorted(items, key=lambda it: it.n_tokens), 1500, skip=prefix.n)
        assert len(batches) > 2 and len({max(it.n_tokens for it in mb) for mb in batches}) > 2
        assert any(len({it.n_tokens for it in mb}) > 1 for mb in batches)      # real padding
        worst = 0.0
        for mb in batches + batches[:1]:                                # several batches reuse one prefix
            full = hidden_states(bb, mb, "cpu", pad)
            shared = hidden_states(bb, mb, "cpu", pad, prefix)
            for b, it in enumerate(mb):
                torch.testing.assert_close(shared[b, :it.n_tokens], full[b, :it.n_tokens], atol=1e-4, rtol=1e-4)
                worst = max(worst, float((full[b, prefix.n:it.n_tokens] - full[0, prefix.n - 1]).abs().max()))
        assert worst > 0.1                                              # the compared states are not degenerate
        for l, (c, r) in zip(linear, before):                           # the prefix's states are never written
            assert torch.equal(l.conv_states[0], c) and torch.equal(l.recurrent_states[0], r)
        # one-token batches too (the module's single-step path would write the conv state in place)
        one = hidden_states(bb, items[:3], "cpu", pad)[:, :prefix.n + 1]
        out = bb(input_ids=torch.tensor([[it.enc.input_ids[prefix.n]] for it in items[:3]]),
                 past_key_values=prefix.cache, use_cache=True).last_hidden_state
        torch.testing.assert_close(out[:, 0], one[:, prefix.n], atol=1e-4, rtol=1e-4)
        assert all(torch.equal(l.conv_states[0], c) for l, (c, r) in zip(linear, before))


def test_predict_same_answers(setup):
    from names_ft import core
    from turion_extract.model import ExtractHeads, hidden_size

    model, bb, layout, items = setup
    pad = layout.tok.pad_token_id
    torch.manual_seed(0)
    heads = ExtractHeads(hidden_size(model)).eval()
    made = []
    real = core.SharedPrefix.__init__
    try:                                                                # predict really shares (no silent fallback)
        core.SharedPrefix.__init__ = lambda self, *a, **k: (real(self, *a, **k), made.append(self))[0]
        a, la = core.predict(bb, heads, items, "cpu", pad, 1500, share_prefix=True)
    finally:
        core.SharedPrefix.__init__ = real
    assert len(made) == 1
    b, lb = core.predict(bb, heads, items, "cpu", pad, 1500, share_prefix=False)
    values = lambda ans: {i: {f: x["value"] for f, x in v["answers"].items()} for i, v in ans.items()}
    assert len(a) == len(items) and values(a) == values(b) and abs(la - lb) < 1e-4
