"""CPU tests: data conversion, reachability of every gold value, batched == unbatched hidden states,
gradient flow through LoRA + heads, and a short train / resume / load / parse run on a tiny random Gemma 4.

    python -m pytest -q tests          # needs data/names (python -m names_ft.prepare) and models/turion-extract-alpha
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
MODEL = os.path.join(ROOT, "models", "turion-extract-alpha")
DATA = os.path.join(ROOT, "data", "names")
need = pytest.mark.skipif(not (os.path.exists(os.path.join(MODEL, "tokenizer.json")) and os.path.exists(DATA)),
                          reason="needs models/turion-extract-alpha and data/names")

from names_ft.fields import FIELDS, answers_to_record, clean_answers, clean_value, record_to_gold, same_record, to_year  # noqa: E402


def test_clean_value():
    assert [clean_value(v) for v in ["C..", "R..,", "Hans-J.,", " Ashley", "St.", "Jr.", "al-Leblanc", "x ("]] == \
        ["C.", "R.", "Hans-J.", "Ashley", "St.", "Jr.", "al-Leblanc", "x"]
    a = clean_answers({"middle_names": {"type": "list", "value": ["C..", "C.", "("]},
                       "first_name": {"type": "text", "value": ","}, "life_start": {"type": "text", "value": "1569,"}})
    assert a["middle_names"]["value"] == ["C."] and a["first_name"]["value"] is None
    assert a["life_start"]["value"] == "1569,"          # years are not name fields; to_year reads them


def test_record_conversion():
    rec = {"first_name": "Lucie", "last_name": "", "middle_names": ["A."], "titles": [], "extra_info": [],
           "temporal": [{"start": 1780, "end": 1798, "type": "flourished"}, {"start": None, "end": 1841, "type": "life_span"}]}
    g = record_to_gold(rec)
    assert g["last_name"]["value"] is None and g["titles"]["value"] is None
    assert g["life_end"]["value"] == 1841 and g["life_start"]["value"] is None and g["flourished_start"]["value"] == 1780
    back = answers_to_record({k: {"value": (str(v["value"]) if v["value"] is not None and FIELDS[k]["type"] == "text" else v["value"])}
                              for k, v in g.items()})
    assert same_record(back, rec)
    assert [t["type"] for t in back["temporal"]] == ["life_span", "flourished"]
    assert to_year("-1457") == 1457 and to_year("c.1600") == 1600 and to_year("born") is None and to_year(None) is None


@pytest.fixture(scope="module")
def layout():
    from turion_extract.layout import Layout
    from turion_extract.model import load_tokenizer

    return Layout(load_tokenizer(MODEL), max_len=1024)


@need
def test_every_gold_value_is_reachable(layout):
    from names_ft.core import encode_rows, read_jsonl

    rows = read_jsonl(os.path.join(DATA, "dev.jsonl"))
    items, skipped = encode_rows(layout, rows)
    assert skipped == 0
    missing = [(it.row["state"], name) for it in items for name, tg in zip(it.enc.names, it.targets) if not tg.learnable]
    assert not missing, missing[:10]
    assert max(it.n_tokens for it in items) < 400


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tiny_gemma import make_tiny_gemma

    return make_tiny_gemma(str(tmp_path_factory.mktemp("tiny-gemma4")), tokenizer_from=MODEL)


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    from tiny_qwen import make_tiny_qwen

    return make_tiny_qwen(str(tmp_path_factory.mktemp("tiny-qwen35")))


@pytest.fixture(scope="module")
def qwen_layout(tiny_qwen):
    from turion_extract.layout import Layout
    from turion_extract.model import load_tokenizer

    return Layout(load_tokenizer(tiny_qwen), max_len=1024)


def _padded_matches_single(layout, base):
    from names_ft.core import encode_rows, hidden_states, read_jsonl
    from turion_extract.model import get_backbone, load_model

    model = load_model(base, device="cpu", dtype=torch.float32).eval()
    bb = get_backbone(model)
    rows = read_jsonl(os.path.join(DATA, "dev.jsonl"), 40)
    items, _ = encode_rows(layout, rows)
    items = sorted(items, key=lambda it: it.n_tokens)[::8]           # different lengths -> real padding
    assert len({it.n_tokens for it in items}) > 1
    with torch.no_grad():
        hb = hidden_states(bb, items, "cpu", layout.tok.pad_token_id)
        for b, it in enumerate(items):
            hs = hidden_states(bb, [it], "cpu", layout.tok.pad_token_id)[0]
            torch.testing.assert_close(hb[b, :it.n_tokens], hs, atol=1e-4, rtol=1e-4)


@need
def test_padded_batch_matches_single_requests(layout, tiny):
    _padded_matches_single(layout, tiny)


@need
def test_qwen_padded_batch_matches_single_requests(qwen_layout, tiny_qwen):
    """Right padding is also exact through Qwen3.5's linear-attention layers (causal conv + recurrence)."""
    _padded_matches_single(qwen_layout, tiny_qwen)


@need
def test_qwen_gold_values_reachable(qwen_layout):
    from names_ft.core import encode_rows, read_jsonl

    items, skipped = encode_rows(qwen_layout, read_jsonl(os.path.join(DATA, "dev.jsonl")))
    missing = [(it.row["state"], name) for it in items for name, tg in zip(it.enc.names, it.targets) if not tg.learnable]
    assert skipped == 0 and not missing, missing[:10]


@need
def test_shared_prefix_matches_full_forward(layout, tiny):
    """The prefix-cached pass gives the same hidden states, past the tiny model's 16-token sliding window too."""
    from names_ft.core import SharedPrefix, encode_rows, hidden_states, micro_batches, predict, read_jsonl
    from turion_extract.model import ExtractHeads, get_backbone, hidden_size, load_model

    model = load_model(tiny, device="cpu", dtype=torch.float32).eval()
    bb = get_backbone(model)
    pad = layout.tok.pad_token_id
    items, _ = encode_rows(layout, read_jsonl(os.path.join(DATA, "dev.jsonl"), 40))
    ids = SharedPrefix.common(items)
    assert ids and len(ids) > model.config.get_text_config().sliding_window
    with torch.no_grad():
        prefix = SharedPrefix(bb, ids, "cpu")
        for mb in micro_batches(items, 4000, skip=prefix.n):            # several batches reuse one prefix
            full = hidden_states(bb, mb, "cpu", pad)
            shared = hidden_states(bb, mb, "cpu", pad, prefix)
            for b, it in enumerate(mb):
                torch.testing.assert_close(shared[b, :it.n_tokens], full[b, :it.n_tokens], atol=1e-4, rtol=1e-4)
    heads = ExtractHeads(hidden_size(model)).eval()
    a, la = predict(bb, heads, items, "cpu", pad, 4000, share_prefix=True)
    b, lb = predict(bb, heads, items, "cpu", pad, 4000, share_prefix=False)
    values = lambda ans: {i: {f: x["value"] for f, x in v["answers"].items()} for i, v in ans.items()}
    assert len(a) == len(items) and values(a) == values(b) and abs(la - lb) < 1e-4


@need
def test_gradients_reach_lora_and_heads(layout, tiny):
    from peft import LoraConfig, get_peft_model

    from names_ft.core import batch_loss, encode_rows, read_jsonl
    from names_ft.train import LORA_EXCLUDE, LORA_TARGETS
    from turion_extract.model import ExtractHeads, get_backbone, hidden_size, load_model

    model = get_peft_model(load_model(tiny, device="cpu"),
                           LoraConfig(r=8, lora_alpha=16, target_modules=LORA_TARGETS, exclude_modules=LORA_EXCLUDE))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    heads = ExtractHeads(hidden_size(model), p=32)
    items, _ = encode_rows(layout, read_jsonl(os.path.join(DATA, "dev.jsonl"), 6))
    loss, n = batch_loss(get_backbone(model), heads, items, "cpu", layout.tok.pad_token_id)
    assert n == 6 * len(FIELDS) and torch.isfinite(loss)
    (loss / n).backward()
    lora_b = {k: p for k, p in model.named_parameters() if "lora_B" in k}
    assert lora_b and all(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_b.values())
    # the heads this task uses get gradients; the number/date heads (count, sign, pick, diff) are never used
    # by name fields, get none, and stay as released
    used = ("norm.", "mode.", "q.", "k_start.", "k_end.", "k_tag.", "tag_bias")
    grads = {k: p.grad for k, p in heads.named_parameters()}
    assert all(grads[k] is not None and grads[k].abs().sum() > 0 for k in grads if k.startswith(used)), \
        [k for k in grads if k.startswith(used) and grads[k] is None]


@need
@pytest.mark.parametrize("base", ["gemma", "qwen"])
def test_train_resume_load_parse(base, request, tmp_path, capsys):
    from names_ft import parse_names, train
    from names_ft.core import read_jsonl
    from names_ft.train import read_json

    base_dir = request.getfixturevalue("tiny" if base == "gemma" else "tiny_qwen")
    out = str(tmp_path / "run")
    common = ["--out", out, "--model", MODEL, "--base", base_dir, "--init", "fresh", "--data", DATA, "--device", "cpu",
              "--train-examples", "64", "--dev-examples", "12", "--test-examples", "12", "--micro-tokens", "2048",
              "--tokens-per-step", "2048", "--eval-micro-tokens", "4096", "--eval-every", "2", "--save-every", "2",
              "--warmup", "1"]
    train.main(common + ["--max-steps", "2"])
    assert read_json(os.path.join(out, "last", "state.json"))["step"] == 2
    train.main(common + ["--max-steps", "4"])                          # resumes at step 2
    log = read_jsonl(os.path.join(out, "train-log.jsonl"))
    assert [r["step"] for r in log if "dev" in r] == [2, 4]             # the resumed run starts after step 2
    m = read_json(os.path.join(out, "metrics.json"))
    assert m["tuned"]["test"]["rows"] == 12 and "record_exact" in m["tuned"]["test"]
    best = os.path.join(out, "best")
    for f in ("adapter_model.safetensors", "heads.safetensors", "turion_extract.json", "fields.json", "tokenizer.json"):
        assert os.path.exists(os.path.join(best, f)), f
    cfg = read_json(os.path.join(best, "turion_extract.json"))
    assert cfg["base_model"] == base_dir and cfg["heads"]["d"] == 64
    from safetensors import safe_open

    with safe_open(os.path.join(best, "adapter_model.safetensors"), "pt") as f:       # the layers LoRA is on
        targets = {k.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1] for k in f.keys()}
    if base == "qwen":                  # every linear layer, linear attention included; the base's own tokenizer
        assert {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "down_proj"} <= targets
        assert read_json(os.path.join(best, "tokenizer_config.json"))["eos_token"] == "<|im_end|>"
    else:
        assert targets == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    preds = read_jsonl(os.path.join(out, "predictions-test.jsonl"))
    assert len(preds) == 12 and set(preds[0]["pred"]) == set(preds[0]["gold"])

    capsys.readouterr()
    parse_names.main(["--model", best, "--device", "cpu", "--format", "jsonl", "Maréchal, Lucie, died 1841", "رافع الخرافي, 1273-1331"])
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert len(lines) == 2 and set(lines[0]["answers"]) == set(FIELDS)
