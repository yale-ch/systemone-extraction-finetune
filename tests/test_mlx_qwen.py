"""The MLX backend on Qwen3.5 (three linear-attention layers, then one full-attention layer, repeated): the shared
prefix gives the same hidden states as a full pass, and a model directory with a PEFT LoRA over every linear layer
gives the same hidden states and answers as PyTorch with the LoRA merged. Mac only.

MLX's fp32 matmul on the GPU is only accurate to about 3 digits here (mlx 0.32.4: relative error up to 5e-3 against
numpy, 1e-7 on the CPU), so the exact comparisons run MLX on the CPU and the GPU ones (which also cover mlx-lm's
Metal kernel for the recurrence) use a tolerance well above that and far below what a wrong state gives."""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten, tree_map  # noqa: E402
from mlx_lm.models import qwen3_5  # noqa: E402

from names_ft.mlx_backbone import MLXExtractor, _LinearPrefix, _lora_target, record_prefix  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
DATA = os.path.join(ROOT, "data", "names")


@pytest.fixture
def on(request):
    """Run MLX on the device named by the test's `on` parameter, then go back to the default."""
    default = mx.default_device()
    mx.set_default_device(getattr(mx, request.param))
    yield request.param
    mx.set_default_device(default)


def test_lora_key_mapping():
    assert _lora_target("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight") == \
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
    assert _lora_target("base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_B.weight") == \
        "language_model.model.layers.3.self_attn.q_proj.weight"


# cpu: mlx-lm's array-op recurrence; gpu with key dim 32: its Metal kernel for the recurrence. The random model
# amplifies rounding (cpu: up to 7e-5 between the two passes; gpu: 3e-3); starting a batch from a zero conv or
# recurrent state instead of the prefix's gives differences above 1.
@pytest.mark.parametrize("on, key_dim, atol", [("cpu", 16, 1e-3), ("cpu", 32, 1e-3), ("gpu", 32, 0.03)], indirect=["on"])
def test_shared_prefix_matches_full_pass(on, key_dim, atol):
    mx.random.seed(0)
    args = qwen3_5.TextModelArgs(model_type="qwen3_5", hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=32, vocab_size=512,
                                 linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=key_dim,
                                 linear_value_head_dim=16, tie_word_embeddings=True)
    model = qwen3_5.TextModel(args)
    lm = model.model
    rng = np.random.default_rng(0)
    prefix = rng.integers(1, 512, 20).tolist()
    h_prefix, shared = record_prefix(lm, model.make_cache(), prefix)
    linear = [c for c in shared if isinstance(c, _LinearPrefix)]
    assert len(shared) == 8 and len(linear) == 6
    assert linear[0].arrays[0].shape == (1, 3, 2 * 2 * key_dim + 4 * 16)       # conv state: kernel-1 inputs
    assert linear[0].arrays[1].shape == (1, 4, 16, key_dim)                    # recurrent state
    before = [[np.array(a) for a in c.arrays] for c in linear]

    for lengths in [(3, 12, 7), (9,), (1, 1), (30, 2, 30, 17)]:               # several batches reuse one prefix
        rests = [rng.integers(1, 512, n).tolist() for n in lengths]
        ids = np.zeros((len(rests), max(lengths)), dtype=np.int32)            # right-padded with token 0
        for b, r in enumerate(rests):
            ids[b, :len(r)] = r
        hb = lm(mx.array(ids), cache=[c.batch(len(rests)) for c in shared])
        for b, r in enumerate(rests):
            full = np.array(lm(mx.array([prefix + r]))[0])
            assert np.abs(full[len(prefix):] - full[len(prefix) - 1]).max() > 1            # not degenerate
            assert np.allclose(np.array(h_prefix), full[:len(prefix)], atol=atol)
            assert np.allclose(np.array(hb[b, :len(r)]), full[len(prefix):], atol=atol)
    for c, old in zip(linear, before):                                         # the prefix's states are not written
        assert all(np.array_equal(np.array(a), o) for a, o in zip(c.arrays, old))


@pytest.mark.skipif(not os.path.exists(DATA), reason="needs data/names")
@pytest.mark.parametrize("on", ["cpu"], indirect=True)
def test_model_dir_matches_pytorch(on, tmp_path):
    """A model directory as names_ft.train writes it (PEFT adapter on every decoder linear layer, heads, config,
    tokenizer) over a tiny Qwen3.5 base: MLX with the LoRA folded in == PyTorch with the LoRA merged."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file, save_file
    from pathlib import Path

    from mlx_lm.utils import load_model as mlx_load_model
    from tiny_qwen import make_tiny_qwen

    from names_ft.core import encode_rows, hidden_states, predict, read_jsonl
    from names_ft.fields import FIELDS
    from turion_extract.model import ExtractHeads, get_backbone, hidden_size, load_model

    base = make_tiny_qwen(str(tmp_path / "tiny-qwen35"))
    torch.manual_seed(0)
    model = get_peft_model(load_model(base, device="cpu", dtype=torch.float32),
                           LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    with torch.no_grad():
        for k, p in model.named_parameters():
            if "lora_B" in k:
                p.normal_(std=0.05)              # B @ A as large as the tiny model's own weights (std 0.02)
    heads = ExtractHeads(hidden_size(model)).eval()
    out = str(tmp_path / "best")
    os.makedirs(out)
    model.save_pretrained(out)
    save_file({k: v.detach().contiguous() for k, v in heads.state_dict().items()}, os.path.join(out, "heads.safetensors"))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(os.path.join(base, f), out)
    with open(os.path.join(out, "turion_extract.json"), "w") as f:
        json.dump({"name": "turion-extract-names", "base_model": base, "adapter": True, "heads": heads.config(),
                   "max_len": 1024, "fields": FIELDS}, f)
    with open(os.path.join(out, "fields.json"), "w") as f:
        json.dump(FIELDS, f)

    lora = load_file(os.path.join(out, "adapter_model.safetensors"))
    kinds = {k.split(".layers.")[1].split(".", 1)[1].split(".lora_")[0] for k in lora}
    assert kinds == {f"linear_attn.{n}" for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")} | \
        {f"self_attn.{n}_proj" for n in "qkvo"} | {f"mlp.{n}_proj" for n in ("gate", "up", "down")}
    assert all(k.startswith("base_model.model.model.layers.") for k in lora)

    ext = MLXExtractor(out, dtype=mx.float32)
    base_mlx, _ = mlx_load_model(Path(base), model_config={"model_type": "qwen3_5"})        # no LoRA
    base_mlx.update(tree_map(lambda p: p.astype(mx.float32), base_mlx.parameters()))
    w, w0 = dict(tree_flatten(ext.lm.parameters())), dict(tree_flatten(base_mlx.language_model.model.parameters()))
    changed = {k for k in w if not np.array_equal(np.array(w[k]), np.array(w0[k]))}
    assert len(changed) == len(lora) // 2 == 62 and all(k.endswith("_proj.weight") or ".in_proj_" in k for k in changed)

    merged = PeftModel.from_pretrained(load_model(base, device="cpu", dtype=torch.float32), out).merge_and_unload().eval()
    bb = get_backbone(merged)
    items, _ = encode_rows(ext.layout, read_jsonl(os.path.join(DATA, "dev.jsonl"), 24))
    pad = ext.tok.pad_token_id
    with torch.no_grad():
        ref = hidden_states(bb, items, "cpu", pad)
        no_lora = hidden_states(get_backbone(load_model(base, device="cpu", dtype=torch.float32).eval()), items, "cpu", pad)
    assert float((no_lora - ref).abs().max()) > 0.5                    # the LoRA matters: without it, far off
    got = ext._hidden(items, None, 0, pad)
    for b, it in enumerate(items):                                     # MLX on the CPU, fp32: the same numbers
        torch.testing.assert_close(got[b, :it.n_tokens], ref[b, :it.n_tokens], atol=1e-4, rtol=1e-4)

    values = lambda ans: {i: {f: x["value"] for f, x in v["answers"].items()} for i, v in ans.items()}
    want, _ = predict(bb, heads, items, "cpu", pad, 1500, share_prefix=False)
    for share in (True, False):
        answers, _ = ext.predict(items, micro_tokens=1500, share_prefix=share)
        assert len(answers) == len(items) and values(answers) == values(want)

    # as used: on the GPU, weights and activations as stored (bf16). Close to PyTorch in bf16's precision.
    mx.set_default_device(mx.gpu)
    ext = MLXExtractor(out)
    got = ext._hidden(items, None, 0, pad)
    err = torch.cat([(got[b, :it.n_tokens] - ref[b, :it.n_tokens]).abs().flatten() for b, it in enumerate(items)])
    scale = float(ref.abs().mean())
    print(f"bf16 on the GPU vs PyTorch fp32: max {float(err.max()):.4f}, mean {float(err.mean()):.5f}; mean |h| {scale:.3f}")
    assert float(err.max()) < 0.25 and float(err.mean()) < 0.02 * scale
    answers, _ = ext.predict(items, micro_tokens=1500)
    same = sum(values(answers)[i] == v for i, v in values(want).items())
    print(f"bf16 on the GPU: {same}/{len(items)} names with all values as PyTorch's (random heads)")
