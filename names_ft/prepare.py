"""Build train/dev/test JSONL from the Hub dataset (run on the laptop; `bin/bouchet push-data` copies it up).

    python -m names_ft.prepare                       # downloads the parquet, writes data/names/
    python -m names_ft.prepare --parquet path.parquet

Each row: {"id", "source" (writing system), "state" (the name string), "gold" ({field: {"value"}}),
"record" (the dataset's YAML answer, parsed)}. Duplicate inputs are dropped (the dataset has 499,996 rows but
349,682 distinct inputs) and splits are by a hash of the input, so no test input is also a training input.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import yaml

from .fields import PROMPT_PREFIX, record_to_gold, row_id, script_of, split_of

REPO = "small-models-for-glam/synthetic-parsed-names-yaml"
FILE = "data/train-00000-of-00001.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="local copy of the dataset parquet (default: download from the Hub)")
    ap.add_argument("--out", default="data/names")
    ap.add_argument("--test-pct", type=int, default=1)
    ap.add_argument("--dev-pct", type=int, default=1)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    path = args.parquet
    if not path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, FILE, repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    os.makedirs(args.out, exist_ok=True)
    outs = {s: open(os.path.join(args.out, f"{s}.jsonl"), "w", encoding="utf-8") for s in ("train", "dev", "test")}
    seen: set[str] = set()
    stats = collections.Counter()
    for r in rows:
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        user, answer = msgs["user"], msgs["assistant"]
        if not user.startswith(PROMPT_PREFIX):
            stats["skipped_prompt"] += 1
            continue
        state = user[len(PROMPT_PREFIX):]
        if state in seen:
            stats["duplicate"] += 1
            continue
        seen.add(state)
        rec = yaml.safe_load(answer)
        split = split_of(state, args.test_pct, args.dev_pct)
        row = {"id": row_id(state), "source": script_of(state), "state": state,
               "gold": record_to_gold(rec), "record": rec}
        outs[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] += 1
        stats[f"{split}:{row['source']}"] += 1
    for f in outs.values():
        f.close()
    stats = dict(sorted(stats.items()))
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump({"source": REPO, "rows": len(rows), **stats}, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
