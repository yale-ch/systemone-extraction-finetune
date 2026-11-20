"""Fine-tune Turion Extract alpha (LoRA + extraction heads on Gemma 4 12B) to parse personal names.

    python -m names_ft.train --out runs/names-60k                    # continue from the released adapter + heads
    python -m names_ft.train --out runs/names-fresh --init fresh     # new LoRA, new heads, same recipe
    python -m names_ft.train --out runs/names-qwen --init fresh --base Qwen/Qwen3.5-4B --train-examples 0
                                                                     # the same method on another base model

--init fresh is how Turion Extract alpha itself was made from Gemma 4 12B: a new LoRA (r 32, alpha 64) on every
linear layer of the language model and new heads sized to its hidden states, trained with this recipe. With
--base it works on another decoder, such as Qwen3.5-4B; the tokenizer and chat template come from that base.

A run:
1. Zero-shot: the released model is scored on dev and test (skipped with --init fresh or --no-zero-shot).
2. Training: AdamW, warmup then linear decay, separate learning rates for LoRA and heads, gradient clipping.
   The loss is turion_extract's own (-log of the total probability of every way the heads reach the gold).
3. Selection: every --eval-every steps the dev set is scored; the best record-exact checkpoint (ties: lower dev
   loss) is written to best/. Test is never used for selection.
4. Final: best/ is scored on test -> metrics.json, predictions-test.jsonl.

`last/` is written every --save-every steps; rerunning the same command resumes from it (also after a Slurm
requeue). best/ is a complete model directory: `Extractor.from_dir(best)` or `python -m names_ft.parse_names`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import time

import torch
from safetensors.torch import load_file, save_file

from turion_extract.layout import LAYOUT_VERSION, Layout
from turion_extract.model import ExtractHeads, get_backbone, hidden_size, load_model, load_tokenizer

from .core import batch_loss, encode_rows, metrics, micro_batches, predict, read_jsonl
from .fields import FIELDS, answers_to_record

log = logging.getLogger("names_ft.train")
# Turion's rule is "LoRA on all linear layers of the language model". A fresh adapter targets every nn.Linear inside
# the decoder (decoder_linears): on Gemma 4 that is LORA_TARGETS, as in the released adapter; on Qwen3.5 it adds the
# linear-attention projections (in_proj_qkv/z/a/b, out_proj). Vision and audio towers, multimodal embedders and the
# output layer are outside the decoder. (peft's "all-linear" also took Gemma 4's vision patch_dense and
# embedding_projection.)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_EXCLUDE = ".*(vision|audio|multi_modal_projector|embed_vision|embed_audio).*"   # used with LORA_TARGETS
COPY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE", "NOTICE")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--model", default="models/turion-extract-alpha", help="released Turion model directory")
    ap.add_argument("--base", help="base model id or path (default: the one in turion_extract.json)")
    ap.add_argument("--init", choices=("adapter", "fresh"), default="adapter")
    ap.add_argument("--data", default="data/names")
    ap.add_argument("--train-examples", type=int, default=60000, help="0 = all ~342k distinct training names")
    ap.add_argument("--dev-examples", type=int, default=1000)
    ap.add_argument("--test-examples", type=int, default=0, help="0 = all")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = no limit)")
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--schedule", choices=("linear", "cosine", "constant"), default="linear")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tokens-per-step", type=int, default=32768, help="real (unpadded) tokens per optimizer step")
    ap.add_argument("--micro-tokens", type=int, default=16384, help="padded tokens per forward pass")
    ap.add_argument("--eval-micro-tokens", type=int, default=32768, help="the same for evaluation (after the shared prefix)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=False,
                    help="gradient checkpointing: recompute activations in the backward pass. Off by default (faster); "
                         "turn it on (--grad-ckpt) or lower --micro-tokens if a run runs out of GPU memory")
    ap.add_argument("--no-zero-shot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="cuda / mps / cpu (default: cuda if available)")
    return ap.parse_args(argv)


def read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# --------------------------------------------------------------------------- model
def build(args, cfg: dict):
    from peft import LoraConfig, PeftModel, get_peft_model

    base = args.base or cfg["base_model"]
    if args.init == "adapter":    # the released model's copy of its base's tokenizer
        tok_dir = args.model if os.path.exists(os.path.join(args.model, "tokenizer_config.json")) else base
    else:                         # a fresh adapter reads text the way its base does (e.g. Qwen's tokenizer)
        tok_dir = base
    tok = load_tokenizer(tok_dir)
    model = load_model(base, device=args.device)
    if args.init == "adapter":
        model = PeftModel.from_pretrained(model, args.model, is_trainable=True)
        heads = ExtractHeads(**cfg["heads"])
        heads.load_state_dict(load_file(os.path.join(args.model, "heads.safetensors")))
    else:
        torch.manual_seed(args.seed)
        lc = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=decoder_linears(model))
        model = get_peft_model(model, lc)
        heads = ExtractHeads(hidden_size(model), p=cfg["heads"].get("p", 512))
        targets = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules() if hasattr(m, "lora_A")})
        log.info("fresh LoRA (r 32, alpha 64) on %s", ", ".join(targets))
    device = next(model.parameters()).device
    heads.to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tok, model, heads, base, device


def decoder_linears(model) -> str:
    """A peft target_modules pattern (full match) for every nn.Linear inside the decoder backbone."""
    bb = get_backbone(model)
    prefix = next(n for n, m in model.named_modules() if m is bb)
    leaves = sorted({n.rsplit(".", 1)[-1] for n, m in bb.named_modules() if isinstance(m, torch.nn.Linear)})
    return (re.escape(prefix) + r"\..*" if prefix else ".*") + r"\.(" + "|".join(map(re.escape, leaves)) + ")"


def trainable(model) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def lr_lambda(args, total: int):
    def f(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        if args.schedule == "constant":
            return 1.0
        frac = (step - args.warmup) / max(1, total - args.warmup)
        return max(0.0, 1.0 - frac) if args.schedule == "linear" else 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
    return f


def save_trainable(model, heads, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)                                     # adapter_config.json + adapter_model.safetensors
    save_file({k: v.detach().cpu().contiguous() for k, v in heads.state_dict().items()},
              os.path.join(path, "heads.safetensors"))


def load_trainable(model, heads, path: str) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, load_file(os.path.join(path, "adapter_model.safetensors")))
    heads.load_state_dict(load_file(os.path.join(path, "heads.safetensors")))


def write_model_dir(args, cfg: dict, base: str, tok, model, heads, path: str, step: int, dev: dict) -> None:
    """A self-contained Turion model directory that Extractor.from_dir and parse_names can load."""
    save_trainable(model, heads, path)
    if args.init == "adapter":    # the released model's tokenizer files, licence and notice
        for f in COPY:
            src = os.path.join(args.model, f)
            if os.path.exists(src):
                shutil.copy(src, path)
    else:                         # the base's tokenizer (and chat template), which the layout was built from
        tok.save_pretrained(path)
    out = dict(cfg, name="turion-extract-names", base_model=base, adapter=True, heads=heads.config(),
               step=step, val=dev, fields=FIELDS, layout_version=LAYOUT_VERSION,
               finetuned_from=os.path.basename(os.path.abspath(args.model)) if args.init == "adapter" else None,
               train_config={k: v for k, v in vars(args).items()})
    with open(os.path.join(path, "turion_extract.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(path, "fields.json"), "w") as f:
        json.dump(FIELDS, f, indent=1)


# --------------------------------------------------------------------------- evaluation
def evaluate(model, backbone, heads, items, rows, device, pad_id, micro_tokens) -> tuple[dict, dict]:
    was = model.training
    model.eval()
    heads.eval()
    t0 = time.time()
    answers, loss = predict(backbone, heads, items, device, pad_id, micro_tokens)
    m = metrics(rows, answers)
    m["loss"] = round(loss, 5) if loss is not None else None
    m["seconds"] = round(time.time() - t0, 1)
    if was:
        model.train()
        heads.train()
    return m, answers


def short(m: dict) -> str:
    return (f"record_exact {m['record_exact']}  field_mean {m['field_mean']}  loss {m['loss']}  "
            f"null P/R {m['null']['precision']}/{m['null']['recall']}  ({m['seconds']}s)")


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    with open(os.path.join(args.model, "turion_extract.json")) as f:
        cfg = json.load(f)
    metrics_path = os.path.join(args.out, "metrics.json")
    results = read_json(metrics_path) or {}
    last_dir, best_dir = os.path.join(args.out, "last"), os.path.join(args.out, "best")
    state = read_json(os.path.join(last_dir, "state.json"))

    tok, model, heads, base, device = build(args, cfg)
    backbone = get_backbone(model)
    pad_id = tok.pad_token_id
    layout = Layout(tok, max_len=args.max_len)
    n_lora = sum(p.numel() for p in trainable(model))
    log.info("base %s, init %s, %.1fM LoRA + %.1fM head parameters, device %s", base, args.init, n_lora / 1e6,
             sum(p.numel() for p in heads.parameters()) / 1e6, device)

    # data: a seeded sample of training names; dev and test as written by names_ft.prepare
    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    random.Random(args.seed).shuffle(train_rows)
    if args.train_examples:
        train_rows = train_rows[:args.train_examples]
    dev_rows = read_jsonl(os.path.join(args.data, "dev.jsonl"), args.dev_examples or None)
    test_rows = read_jsonl(os.path.join(args.data, "test.jsonl"), args.test_examples or None)
    train, sk = encode_rows(layout, train_rows)
    dev, _ = encode_rows(layout, dev_rows)
    test, _ = encode_rows(layout, test_rows)
    fields_total = sum(len(it.targets) for it in train)
    fields_learn = sum(it.n_fields for it in train)
    log.info("train %d names (%d too long), %d tokens, %d/%d fields learnable; dev %d; test %d", len(train), sk,
             sum(it.n_tokens for it in train), fields_learn, fields_total, len(dev), len(test))

    # step plan: shuffled per epoch, micro-batches by padded tokens, steps by real tokens
    plan = []
    for ep in range(args.epochs):
        order = list(train)
        random.Random(args.seed * 1000 + ep).shuffle(order)
        step, toks = [], 0
        for mb in micro_batches(order, args.micro_tokens):
            step.append(mb)
            toks += sum(it.n_tokens for it in mb)
            if toks >= args.tokens_per_step:
                plan.append(step)
                step, toks = [], 0
        if step:
            plan.append(step)
    total = min(len(plan), args.max_steps) if args.max_steps else len(plan)
    log.info("%d optimizer steps (%d epochs, ~%d names per step)", total, args.epochs, len(train) * args.epochs // max(1, len(plan)))

    # zero-shot
    if args.init == "adapter" and not args.no_zero_shot and "zero_shot" not in results:
        zs_dev, _ = evaluate(model, backbone, heads, dev, dev_rows, device, pad_id, args.eval_micro_tokens)
        log.info("zero-shot dev: %s", short(zs_dev))
        zs_test, _ = evaluate(model, backbone, heads, test, test_rows, device, pad_id, args.eval_micro_tokens)
        log.info("zero-shot test: %s", short(zs_test))
        results["zero_shot"] = {"dev": zs_dev, "test": zs_test}
        write_json(metrics_path, results)

    opt = torch.optim.AdamW([
        {"params": trainable(model), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": list(heads.parameters()), "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args, total))
    start, best, history = 0, None, []
    if state:
        load_trainable(model, heads, last_dir)
        ck = torch.load(os.path.join(last_dir, "optim.pt"), map_location=device, weights_only=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, history = state["step"], state.get("best"), state.get("history", [])
        log.info("resumed from %s at step %d (best %s)", last_dir, start, best)

    def save_last(step: int) -> None:
        tmp = last_dir + ".tmp"
        save_trainable(model, heads, tmp)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, os.path.join(tmp, "optim.pt"))
        with open(os.path.join(tmp, "state.json"), "w") as f:
            json.dump({"step": step, "best": best, "history": history}, f)
        shutil.rmtree(last_dir, ignore_errors=True)
        os.replace(tmp, last_dir)

    def run_eval(step: int) -> None:
        nonlocal best
        m, _ = evaluate(model, backbone, heads, dev, dev_rows, device, pad_id, args.eval_micro_tokens)
        key = [m["record_exact"], -(m["loss"] or 0.0)]
        better = best is None or key > best["key"]
        log.info("step %d dev: %s%s", step, short(m), "  * best" if better else "")
        history.append({"step": step, "dev": m})
        with open(os.path.join(args.out, "train-log.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, "dev": m}) + "\n")
        if better:
            best = {"step": step, "key": key, "dev": m}
            write_model_dir(args, cfg, base, tok, model, heads, best_dir, step, m)

    model.train()
    heads.train()
    params = trainable(model) + list(heads.parameters())
    t0, toks_done, loss_acc, n_acc = time.time(), 0, 0.0, 0
    for s in range(start, total):
        mbs = plan[s]
        n_fields = max(1, sum(it.n_fields for mb in mbs for it in mb))
        for mb in mbs:
            loss, _ = batch_loss(backbone, heads, mb, device, pad_id)
            (loss / n_fields).backward()
            loss_acc += float(loss.detach())
            toks_done += sum(it.n_tokens for it in mb)
        n_acc += n_fields
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step = s + 1
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            mem = f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if torch.cuda.is_available() else ""
            log.info("step %d/%d  loss %.4f  gnorm %.2f  lr %.2e  %.0f tok/s  eta %.0f min%s", step, total,
                     loss_acc / max(1, n_acc), float(gnorm), sched.get_last_lr()[0], toks_done / el,
                     el / max(1, step - start) * (total - step) / 60, mem)
            with open(os.path.join(args.out, "train-log.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "loss": loss_acc / max(1, n_acc), "gnorm": float(gnorm),
                                    "lr": sched.get_last_lr()[0]}) + "\n")
            loss_acc, n_acc = 0.0, 0
        if step % args.eval_every == 0 or step == total:
            run_eval(step)
        if step % args.save_every == 0 or step == total:
            save_last(step)

    if best is None:                       # resumed after the last step, or nothing to train
        run_eval(total)
    # final: the selected checkpoint on test
    load_trainable(model, heads, best_dir)
    m, answers = evaluate(model, backbone, heads, test, test_rows, device, pad_id, args.eval_micro_tokens)
    log.info("best step %d, test: %s", best["step"], short(m))
    results.update({"tuned": {"best_step": best["step"], "dev": best["dev"], "test": m}, "steps": total,
                    "train_names": len(train), "args": vars(args)})
    write_json(metrics_path, results)
    with open(os.path.join(args.out, "predictions-test.jsonl"), "w", encoding="utf-8") as f:
        for r in test_rows:
            got = (answers.get(r["id"]) or {}).get("answers") or {}
            f.write(json.dumps({"id": r["id"], "state": r["state"], "gold": r["record"],
                                "pred": answers_to_record(got), "answers": got}, ensure_ascii=False) + "\n")
    log.info("wrote %s and %s", metrics_path, best_dir)


if __name__ == "__main__":
    main()
