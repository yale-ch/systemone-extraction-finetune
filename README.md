# turion-names

Fine-tune [Turion Extract alpha](https://huggingface.co/turion-ai/turion-extract-alpha) (LoRA + extraction heads on Gemma 4 12B)
to parse a personal name, in any script and with any dates, titles or comments, into typed fields. Training
runs on Yale's Bouchet cluster and is driven from the laptop, like `yale-ch/system1-finetune-bench`.

## The task as Turion fields

Data: [`small-models-for-glam/synthetic-parsed-names-yaml`](https://huggingface.co/datasets/small-models-for-glam/synthetic-parsed-names-yaml),
with 499,996 rows, 349,682 distinct names, Latin and Arabic script. Each YAML answer is flattened into nine
fields (`names_ft/fields.py`):

| Field | Type | From the YAML |
|---|---|---|
| `first_name`, `last_name` | text | as is; `last_name: ''` becomes null |
| `middle_names`, `titles`, `extra_info` | list of text | as is; `[]` becomes null |
| `life_start`, `life_end` | text (a year) | `temporal` entry with `type: life_span` |
| `flourished_start`, `flourished_end` | text (a year) | `temporal` entry with `type: flourished` |

Every gold value occurs verbatim in its name, and every gold field is reachable by Turion's heads.
`tests/` checks this on the whole dev set with the real tokenizer. Years are `text` spans rather than
`number`, because Turion's number parser reads `Bonifacio Zamora, -1457` as −1457. Outputs are converted back
to the dataset's YAML shape, with years as ints (`fields.answers_to_record`).

The data is deduplicated, then split by a hash of the name: 342,700 train, 3,448 dev and 3,534 test names. No
test name is also a training name. One request is about 242 tokens, mostly the fixed field list.

## Workflow (from the laptop)

```bash
cp .env.example .env                      # once: your cluster path and Slurm account (git-ignored, never synced)
hf download turion-ai/turion-extract-alpha --local-dir models/turion-extract-alpha   # once; see REPRODUCING.md
uv sync                                   # local env; CPU is fine for the tests
uv run python -m names_ft.prepare         # data/names/{train,dev,test}.jsonl (already done once)
uv run pytest -q tests                    # conversion, reachability, padding, gradients, train/resume/parse on a tiny Gemma 4

bin/bouchet login                         # ssh to bouchet.ycrc.yale.edu; Duo once, then reused for 8 h
bin/bouchet push-model                    # models/turion-extract-alpha -> cluster (~670 MB, once)
bin/bouchet setup                         # code + .venv, pip-installed on a compute node via srun (~10 min, once)
bin/bouchet push-data                     # data/names -> cluster
bin/bouchet fetch-base                    # gemma-4-12B-it into the shared HF cache (~24 GB, once)
bin/bouchet test                          # the same tests on the login node

bin/bouchet submit names-60k              # default run: 60k names, 1 epoch, continue from the released adapter
bin/bouchet status names-60k              # queue + log tail
bin/bouchet pull names-60k                # metrics.json, train-log.jsonl, predictions-test.jsonl
bin/bouchet pull-model names-60k          # + runs/names-60k/best (the fine-tuned model directory)
```

`bin/bouchet` connects to `bouchet.ycrc.yale.edu` as your local user, sharing one ssh connection across calls.
No user name, Slurm account or cluster path is in the repo. They come from `.env` (copy `.env.example`) or the
environment, and a variable exported in the shell wins over `.env`. `NAMES_REMOTE` (required) is the project
directory on the cluster, where everything lives. `NAMES_ACCOUNT` is the Slurm account (unset: your default one).
`NAMES_HOST` sets an ssh alias or another user. `NAMES_HF_HOME` sets the HF cache, which defaults to
`$NAMES_REMOTE/hf_cache` and can be shared with other projects. `bin/bouchet` passes these on over ssh, and jobs
inherit them. On the cluster, `bouchet/env.local.sh` can set the same variables and wins. Check what's free with
`ysu gpus --free` ([yale-slurm-utils](https://github.com/wjbmattingly/yale-slurm-utils)) before choosing a partition.

The cluster needs no uv. `bouchet/setup.sh` builds `.venv` with `python3 -m venv` and pip from
`requirements-cluster.txt`, which pins the exact versions in `uv.lock` (torch 2.11.0+cu128, transformers 5.23.0,
peft 0.21.3); every Linux package there is a prebuilt wheel for glibc ≥ 2.28. It needs Python 3.11 or 3.12. If the
cluster's `python3` is older, put `module load <a Python 3.12 module>` (see `module spider Python`) in
`bouchet/env.local.sh` on the cluster; every job sources that file too. The install runs on a compute node
(`bouchet/setup-job.sh`: srun on `devel`, else `day`; set `NAMES_SETUP_PARTITION` to choose) because pip was killed
on the login node; `bin/bouchet setup-login` still installs on the login node. An interrupted install is detected
(`.venv/.setup-complete` is missing) and rebuilt from scratch on the next run. After changing dependencies:
`uv lock && bin/bouchet requirements`.

### Resources

Defaults (`bouchet/train.sbatch`): `gpu_rtx6000`, one RTX PRO 6000 Blackwell (96 GB), 96G RAM, 8 h, account
`$NAMES_ACCOUNT`. This is the GPU Turion itself was trained on. Its run did ~19.5M tokens in 3 h 22 min, so expect
about 2.5–3 h for the default 60k names (14.5M tokens), plus a few minutes of evaluation. All 342.7k names
(83M tokens) is about 14 h:

```bash
bin/bouchet submit names-all --time=24:00:00 -- --train-examples 0
bin/bouchet submit names-60k-b200 --partition=gpu_b200 --gpus=b200:1 --mem=128G      # if B200s are free
bin/bouchet submit names-60k-scav --partition=scavenge_gpu --gpus=rtx_pro_6000_blackwell:1   # preemptible
```

Jobs are `--requeue`d and save `last/` every 100 steps. A requeued or rerun job resumes from there, so the
preemptible scavenge partition is safe to use. A GPU under ~80 GB needs a smaller micro-batch, e.g. `--micro-tokens 4096`.

### Another base model: Qwen3.5-4B

Turion Extract alpha was made from plain `google/gemma-4-12B-it` with a **fresh** LoRA (r 32, alpha 64) on every
linear layer of the language model, plus freshly initialised heads. It had one pass over about 39k extraction
requests built from six public datasets and 15k synthetic documents: 299 steps of about 65k tokens, at learning
rates 1e-4 (LoRA) and 1e-3 (heads). That data pipeline isn't released, but the method is `--init fresh`. For one
task like names, the general stage isn't needed: a fresh adapter and fresh heads train directly on the names.

```bash
bin/bouchet setup                         # again: adds flash-linear-attention (Linux only) to .venv
bin/bouchet fetch-base Qwen/Qwen3.5-4B    # ~9 GB into the shared HF cache
bin/bouchet submit names-qwen-all --time=12:00:00 -- --init fresh --base Qwen/Qwen3.5-4B --train-examples 0 --eval-every 200 --save-every 200
```

What changes with a fresh base (`build` in `names_ft/train.py`):
- **Tokenizer and chat template** come from the base. A Qwen request is about 226 tokens: 134 before the name,
  about 18 for the name and 74 after it. Qwen's template ends the header with `<think>` (thinking on), and the
  heads learn that layout.
- **LoRA** goes on every `nn.Linear` inside the decoder (`decoder_linears`). On Gemma 4 that is exactly the
  released adapter's q/k/v/o and gate/up/down_proj. On Qwen3.5 it adds the linear-attention projections
  (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`): 64.9M LoRA and 18.4M head parameters.
- **Linear attention.** Qwen3.5 has 3 Gated DeltaNet layers per full-attention layer. Without the
  `flash-linear-attention` kernels, transformers runs a slow reference version; the log then says "falling back".
  The kernels are pinned for Linux in `requirements-cluster.txt`.
- **Prediction** shares the prefix through linear-attention layers too. The batch starts from the prefix's
  final conv and recurrent state (`core.SharedPrefix`, and `_LinearPrefix` for MLX). It's about 2.3× faster
  on the Mac (200 names: 21 s → 9 s with PyTorch, 10 s → 4 s with MLX). On CUDA it stays off, and
  `predict` runs whole requests. flash-linear-attention's kernels haven't been checked with the shared state;
  set `NAMES_SHARE_LINEAR_CUDA=1` to try it once `tests/test_qwen_prefix.py`'s comparison has been run on a GPU node.

The tests run a fresh train, resume and parse on both a tiny Gemma 4 and a tiny Qwen3.5 (`tests/tiny_qwen.py`).
They check that right padding is exact through linear attention, and that every dev answer can be reached with
Qwen's tokenizer.

## A run (`names_ft/train.py`)

1. **Zero-shot:** the released model is scored on dev (1,000) and test (all), with the same field list.
2. **Training:** AdamW, 30 warmup steps then linear decay, LoRA lr 1e-4 and heads lr 1e-3 (Turion's own recipe),
   ~32k real tokens (~135 names) per step, gradient clipping 1.0, no gradient checkpointing (`--grad-ckpt` turns it on, to save memory). Requests are
   right-padded into batches of ≤16k tokens. The loss is turion_extract's own `field_losses`.
3. **Selection:** dev is scored every 50 steps. The best `record_exact` checkpoint (ties go to lower dev loss) is
   written to `best/`. Test is never used for selection.
4. **Final:** `best/` is scored on test → `metrics.json` and `predictions-test.jsonl` (gold and predicted records).

Metrics: `record_exact` is the share of names whose whole record equals the dataset's answer exactly (strings,
list order, years); compare it with the Qwen SFT parsers' accuracy. There are also per-field scores, per-script
(`latin`, `arabic`) and per-type scores under the Extract Index rules, and null precision and recall.

`best/` is a complete Turion model directory: adapter, heads, tokenizer, `turion_extract.json`, and `fields.json`
(the field list it was trained with).

## Parsing names

```bash
python -m names_ft.parse_names --model runs/names-60k/best "Maréchal, Lucie, died 1841" "Dr. W. D. Godsall"
python -m names_ft.parse_names --model runs/names-60k/best --input names.txt --format jsonl --out parsed.jsonl
```

`best/` holds only the adapter, heads and tokenizer (637 MB). The 24 GB base model `google/gemma-4-12B-it` is
loaded from the local Hugging Face cache, and downloaded first if it isn't there. Set `HF_HUB_OFFLINE=1` to skip the
Hub check. The device is cuda if available, else mps (the Mac GPU), else cpu. On an M5 Max (128 GB), 1,000 names
take about 2 minutes (0.11 s per name, after a 12 s load). 998 of 1,000 test names came out the same as on the cluster.
Only one model copy is worth loading: two copies on the same GPU each ran at half speed.

On a Mac, `--backend mlx` runs Gemma in MLX (Apple's framework) instead of PyTorch. It's 1.6× faster: 1,000 names
in 73 s, about 0.065 s per name after a 6 s load. 998 of 1,000 names came out the same as with PyTorch and on
the cluster, with the same accuracy.

```bash
uv sync --group mlx                       # once; a plain `uv sync` removes the group again
uv run python -m names_ft.parse_names --backend mlx --model runs/names-60k/best --input names.txt --format jsonl --out parsed.jsonl
```

`names_ft/mlx_backbone.py` loads the base weights from the Hugging Face cache with mlx-lm and folds the
fine-tuned LoRA into them as they load, so there is no converted copy on disk. The extraction heads stay in
PyTorch on the CPU. The prefix is shared as in PyTorch. Batches of 1k–8k tokens ran at the same speed and 16k+
was slower, so the default `--micro-tokens` is 4096 for MLX. The `mlx` dependency group is Mac-only and
not in `requirements-cluster.txt`.

The YAML output has the dataset's shape. JSONL adds Turion's per-field `p_null`, `confidence` and character spans.
Names are batched, unlike `Extractor.extract`, which runs one request per forward pass.

## Known limits

- **Whole-token spans.** A span covers whole tokens, and Gemma has `..` and `..,` as single tokens, so `C..` can
  only come back as `C..` (the labels say `C.`) and `R..,` as `R..,`. `fields.clean_answers` trims trailing
  `,;:(` and turns a final `..` into `.` on every name field. No training label conflicts with that rule. On
  names-60k's test predictions it fixes 40 of 94 wrong names: `record_exact` goes from 97.34 to 98.47.
- **Repeated list items.** A list item is tagged at every place it occurs, and decoding drops repeats. So
  `middle_names: [H., M., F., M.]` comes back as `[H., M., F.]`, and a middle name equal to the first name
  also tags the first-name position. This affects about 0.6% of names that have middle names.
- **Synthetic data.** Real catalogue names, such as LUX's, will have forms this generator never produced.
  Score a hand-labelled sample of real names before relying on the numbers.
- **Prompt overhead.** A request is ~241 tokens: 141 of instructions and field list before the name, ~16 for the
  name, and 84 of query lines after it. Prediction runs the 141 once and reuses them (`core.SharedPrefix`, the
  same result as a full pass), so 1,000 names went from 249 s to 119 s on the Mac. The query lines can't be shared
  that way, because they have to read the name. Training still runs whole requests, since gradients have to
  pass through the prefix too. With sharing, `--micro-tokens` and `--eval-micro-tokens` count only the tokens
  each batch runs.
