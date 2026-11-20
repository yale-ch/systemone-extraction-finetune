"""Parse names with a fine-tuned model directory (a run's best/), in batches.

    python -m names_ft.parse_names --model runs/names-60k/best "Maréchal, Lucie, died 1841" "Dr. W. D. Godsall"
    python -m names_ft.parse_names --model runs/names-60k/best --input names.txt --format jsonl --out parsed.jsonl

--input is a text file with one name per line, or JSONL with a "name" or "state" key. YAML output has the
dataset's record shape; JSONL output adds Turion's per-field answers (value, p_null, confidence, spans).
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import yaml

from turion_extract.predict import Extractor

from .core import encode_rows, predict
from .fields import FIELDS, answers_to_record


def read_names(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if path.endswith(".jsonl"):
                d = json.loads(line)
                out.append(d.get("name") or d.get("state") or "")
            else:
                out.append(line)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--model", required=True, help="a model directory written by names_ft.train (best/)")
    ap.add_argument("--input")
    ap.add_argument("--out", help="default: stdout")
    ap.add_argument("--format", choices=("yaml", "jsonl"), default="yaml")
    ap.add_argument("--micro-tokens", type=int, help="padded tokens run per forward pass, after the shared prefix "
                    "(default: 32768 for torch; 4096 for mlx, where batches of 16k+ were slower)")
    ap.add_argument("--chunk", type=int, default=20000, help="names encoded and written per chunk")
    ap.add_argument("--device", help="torch backend: default cuda, else mps (Apple GPU), else cpu")
    ap.add_argument("--backend", choices=("torch", "mlx"), default="torch",
                    help="mlx: the backbone runs in MLX on Apple silicon (uv sync --group mlx)")
    args = ap.parse_args(argv)
    args.micro_tokens = args.micro_tokens or (4096 if args.backend == "mlx" else 32768)
    if args.device is None:      # turion_extract's loader alone would pick cpu on a Mac
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    with open(f"{args.model}/fields.json") as f:
        if json.load(f) != FIELDS:
            sys.exit("fields.json in the model directory differs from names_ft.fields.FIELDS: use the code version it was trained with")
    names = list(args.names) + (read_names(args.input) if args.input else [])
    if args.backend == "mlx":
        from .mlx_backbone import MLXExtractor

        ext = MLXExtractor(args.model)
        run = lambda items: ext.predict(items, args.micro_tokens)
    else:
        ext = Extractor.from_dir(args.model, device=args.device)      # LoRA merged for speed
        run = lambda items: predict(ext.backbone, ext.heads, items, ext.device, ext.tok.pad_token_id, args.micro_tokens)
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    for c in range(0, len(names), args.chunk):
        rows = [{"id": str(c + i), "state": n} for i, n in enumerate(names[c:c + args.chunk])]
        items, _ = encode_rows(ext.layout, rows, with_targets=False)
        answers, _ = run(items)
        for r in rows:
            got = (answers.get(r["id"]) or {}).get("answers")
            rec = answers_to_record(got) if got else None              # None: too long to encode
            if args.format == "jsonl":
                out.write(json.dumps({"name": r["state"], "parsed": rec, "answers": got}, ensure_ascii=False) + "\n")
            else:
                out.write(f"# {r['state']}\n" + yaml.safe_dump(rec, sort_keys=False, allow_unicode=True) + "---\n")
        out.flush()
    if args.out:
        out.close()


if __name__ == "__main__":
    main()
