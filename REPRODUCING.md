# Reproducing turion-names

This covers what you need to rerun the two reported runs, where to download each input, and which files in this
directory must not be shared. Checked on 2026-11-20.

| Run | Base | Init | Train names | Test `record_exact` | GPU time (one RTX PRO 6000, 96 GB) | Peak GPU memory |
|---|---|---|---|---|---|---|
| `names-60k` | `google/gemma-4-12B-it` | released Turion adapter | 60,000 (1 epoch, 317 steps) | 97.34 (98.47 after `clean_answers`) | 2 h 32 min | 37.4 GiB |
| `names-qwen-all` | `Qwen/Qwen3.5-4B` | fresh LoRA + heads | 342,700 (1 epoch, 1,704 steps) | 99.80 | 5 h 39 min | 17.1 GiB |

Both use seed 0. GPU training isn't bitwise deterministic, so expect the scores to differ in the last digit.
Prediction on the Mac (PyTorch MPS or MLX) matched the cluster on 998 of 1,000 test names.

## 1. What you need

**Hardware**
- Training: one CUDA GPU with at least 48 GB for Gemma 4 12B (or 24 GB for Qwen3.5-4B), at the default
  `--micro-tokens 16384`. On smaller GPUs, lower `--micro-tokens` or add `--grad-ckpt`.
- Disk: about 24 GB for Gemma, 9 GB for Qwen, 0.7 GB for Turion Extract alpha, 0.2 GB for the data, and 0.4–0.7 GB per
  `best/`. A CUDA `.venv` is about 3 GB more.
- Tests and prediction: any machine. The tests run on CPU. Prediction needs about 26 GB of RAM or unified memory for
  Gemma (Apple silicon works, with `--backend mlx`).

**Software**
- Python 3.11 or 3.12.
- [uv](https://docs.astral.sh/uv/) on the laptop. The cluster only needs `python3 -m venv` and pip, using
  `requirements-cluster.txt`.
- Linux + CUDA 12.8 for training: torch 2.11.0+cu128, transformers 5.23.0, peft 0.21.3. `flash-linear-attention`
  0.6.0 is required for Qwen3.5 in practice; without it, a slow reference implementation runs.

**Accounts**
- None are required. Every model and the dataset below are public and ungated, so downloads work without a Hugging Face
  token. (A token only raises rate limits.)
- No hosted model API is used. `bouchet/env.sh` blanks the API key variables on purpose.
- The `bin/bouchet` and `bouchet/*` scripts are for Yale's Bouchet cluster and need a Yale NetID, Duo, and a PI
  Slurm account. Elsewhere, see section 4b.

## 2. What can be downloaded

All of these are public. Pin the revisions below; they are the ones the reported runs used.

| What | Where | Revision | Size | Licence | Used as |
|---|---|---|---|---|---|
| Names dataset | [`small-models-for-glam/synthetic-parsed-names-yaml`](https://huggingface.co/datasets/small-models-for-glam/synthetic-parsed-names-yaml) (`data/train-00000-of-00001.parquet`) | `f07bedf84b6e37a402de72e5b5230653435d8964` | 499,996 rows | **none declared** on the dataset card | `names_ft.prepare` → `data/names/` |
| Turion Extract alpha | [`turion-ai/turion-extract-alpha`](https://huggingface.co/turion-ai/turion-extract-alpha) | `5060d5f90f5ea5be0db8908874c5267433430496` | ~670 MB | Apache-2.0 | `models/turion-extract-alpha/`: starting adapter + heads (Gemma run), `turion_extract.json` and the `turion_extract` Python package in `code/` (**both runs**) |
| Gemma 4 12B | [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) | `43c9d014b231474fa7212dbb48a762566c2b255d` | ~24 GB | Apache-2.0 | base model, `names-60k` |
| Qwen3.5-4B | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | ~9 GB | Apache-2.0 | base model, `names-qwen-all` |
| PyTorch CUDA wheels | `https://download.pytorch.org/whl/cu128` | pinned in `uv.lock` / `requirements-cluster.txt` | ~3 GB | BSD-style | cluster `.venv` |
| Other Python packages | PyPI | pinned in `uv.lock` | | various | |

```bash
# Turion Extract alpha goes inside the project (the laptop uses a symlink to ../../turion-extract-alpha, which isn't portable)
hf download turion-ai/turion-extract-alpha --revision 5060d5f90f5ea5be0db8908874c5267433430496 \
  --local-dir models/turion-extract-alpha
# base models into the HF cache (or let the first run fetch them)
hf download google/gemma-4-12B-it --revision 43c9d014b231474fa7212dbb48a762566c2b255d --exclude 'original/*'
hf download Qwen/Qwen3.5-4B       --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
# the dataset, at the pinned revision; prepare.py fetches `main` when given no --parquet
hf download small-models-for-glam/synthetic-parsed-names-yaml data/train-00000-of-00001.parquet \
  --repo-type dataset --revision f07bedf84b6e37a402de72e5b5230653435d8964 --local-dir /tmp/names-yaml
```

The code doesn't pin revisions itself: `prepare.py` downloads the dataset's `main`, and `train.py` loads the bases by
name. If an upstream repo changes, the run changes with it. Download at the revisions above first, or add `revision=`
to the code.

### Generated locally, not downloaded

`data/names/{train,dev,test}.jsonl` is built deterministically by `names_ft.prepare`: dedup, then a split by a hash of
the name. On 2026-11-20 it was rebuilt from revision `f07bedf…` and came out byte-identical:

```
3245a53620d93f37091c1463c2304ac141b10b49365c026b82a058db4dd0a8ec  data/names/train.jsonl   (342,700)
3c2d909b21328157fe81303a58b3fb881c3bc8b2fb023dd4c198f0f6d7b63ee5  data/names/dev.jsonl     (3,448)
34250b14df5d3c43a69fecf0af5f33bb3709d6ef2f97cd388ff5e177ea7a006d  data/names/test.jsonl    (3,534)
```

```bash
uv run python -m names_ft.prepare --parquet /tmp/names-yaml/data/train-00000-of-00001.parquet
shasum -a 256 data/names/*.jsonl      # compare with the above
```

## 3. Laptop setup and tests

```bash
uv sync                      # needs models/turion-extract-alpha/code (uv.lock installs it as an editable path)
uv run pytest -q tests       # CPU; tiny Gemma 4 and Qwen3.5 models, plus reachability with the real tokenizers
uv sync --group mlx          # Mac only, for --backend mlx
```

## 4. Training

### 4a. On Bouchet (as in the README)

```bash
cp .env.example .env                                # then fill in NAMES_REMOTE and NAMES_ACCOUNT
bin/bouchet login && bin/bouchet push-model && bin/bouchet setup && bin/bouchet push-data
bin/bouchet fetch-base                              # Gemma
bin/bouchet fetch-base Qwen/Qwen3.5-4B
bin/bouchet submit names-60k
bin/bouchet submit names-qwen-all --time=12:00:00 -- --init fresh --base Qwen/Qwen3.5-4B --train-examples 0 --eval-every 200 --save-every 200
```

The scripts contain no user name, Slurm account or path. Each person sets their own in `.env` (git-ignored, never
synced) or in their shell: `NAMES_REMOTE` (required), `NAMES_ACCOUNT`, and optionally `NAMES_HOST` and
`NAMES_HF_HOME`. `.env.example` documents them all.

### 4b. On any other CUDA machine

The scripts only wrap `python -m names_ft.train`. These are the exact reported runs:

```bash
python -m names_ft.train --out runs/names-60k
python -m names_ft.train --out runs/names-qwen-all --init fresh --base Qwen/Qwen3.5-4B --train-examples 0 \
  --eval-every 200 --save-every 200
```

Every other setting is a default, recorded in each run's `metrics.json` → `args` and in `best/turion_extract.json` →
`train_config`. Compare your results with `runs/*/metrics.json`.

## 5. What is in this directory, and what to share

| Path | Share? | Notes |
|---|---|---|
| `names_ft/`, `tests/`, `pyproject.toml`, `uv.lock`, `requirements-cluster.txt`, `README.md`, this file | **Yes** | Code and pins. No secrets found (see below). |
| `bin/bouchet`, `bouchet/*.sh`, `bouchet/train.sbatch`, `.env.example` | **Yes** | No secrets and no personal settings: user, Slurm account and paths come from `.env` or the environment. The public host name `bouchet.ycrc.yale.edu` and partition names are the only Yale-specific defaults. |
| `.env` | **No** | Your cluster path (with your NetID) and PI Slurm account. Git-ignored, never synced, mode 600. |
| `runs/*/metrics.json`, `runs/*/train-log.jsonl` | **Yes** | Scores, losses and arguments only. |
| `runs/*/slurm-*.out` | **Scrub first** | They contain compute-node hostnames, the NetID inside NFS paths, job IDs, and pinned HF URLs. Share an excerpt, or replace those strings. |
| `runs/*/predictions-test.jsonl` | **Only once the dataset licence is known** | Each line contains a dataset test name and its gold record. |
| `data/names/` | **No; share the recipe** | Derived from a dataset with no declared licence. Point people to the dataset revision, `prepare.py` and the checksums above. |
| `runs/*/best/` (fine-tuned adapters + heads) | **Yes, with a model card** | Derived from Apache-2.0 bases. `names-60k/best` already has Turion's `LICENSE`/`NOTICE`. `names-qwen-all/best` has neither, and its `README.md` is the empty PEFT template. Add a licence, a NOTICE crediting Qwen, a real card and the dataset's provenance first. The weight files are mode 600 locally; that doesn't matter once copied. |
| `models/turion-extract-alpha` | **No; link it** | A symlink to a local copy of a public Hub repo. Its `.cache/`, `code/build/` and `code/*.egg-info/` are local build artifacts. |
| `.venv/` (1.2 GB), `.pytest_cache/`, `__pycache__/` | **No** | Machine-specific and holds absolute local paths. Rebuild with `uv sync`. |

### Must never be shared

A scan of every text file outside `.venv` found no keys, tokens, passwords or private keys. It searched for
API-key and token names, `hf_…`, `sk-…`, `AKIA…`, `BEGIN … PRIVATE KEY`, `Authorization` and cookies. These live
outside the tree, or are git-ignored, and must stay private:

- **`~/.cache/huggingface/token`** (or `HF_TOKEN`): your Hugging Face write token, if you have logged in. Don't put it
  in scripts or `env.local.sh`; this workflow doesn't need it.
- **`~/.ssh/cm-*`**: the shared ssh control sockets `bin/bouchet` opens. For 8 h each one is a Bouchet session that
  is already past Duo. Run `bin/bouchet logout` when you're done, and never copy `~/.ssh`.
- **`.env`** on the laptop: your NetID-bearing cluster path and Slurm account. It is git-ignored and `bin/bouchet sync`
  skips it.
- **`bouchet/env.local.sh`** on the cluster: per-user overrides. It is git-ignored and excluded from `sync`. Keep it
  that way if you ever put anything private in it.
- **`TURION_API_TOKEN`**, which the Turion README's hosted-API example uses. It isn't needed here, and
  `bouchet/env.sh` sets it to empty.
- **Your Yale password, your Duo device, and anything printed by `ssh`/`sbatch` with your NetID** in issues or public
  logs.

`.gitignore` follows the table above. It excludes `.env`,
`bouchet/env.local.sh`, key and token files, `data/`, `models/`, `hf_cache/`, every weight file, `.venv/` and caches.
From `runs/` it keeps only `metrics.json` and `train-log.jsonl`, and ignores `best/`, `last/`, test predictions and
`slurm-*.out`. If you publish the trained models, do it from a separate model repo, not by un-ignoring `runs/`.

## 6. Known gaps in reproducibility

- **Unpinned upstreams** in code: the dataset revision (`prepare.py`) and the base-model revisions (`train.py`).
  Section 2 lists the revisions to use.
- **The `models/turion-extract-alpha` symlink** points outside the project. Both runs need that directory: the
  Qwen run too, for `turion_extract.json` and the `turion_extract` package.
- **The dataset has no licence**, which blocks redistributing the splits and predictions, not reproducing them.
- **GPU nondeterminism**: rerun scores can differ slightly. Shared-prefix prediction through linear attention is off
  on CUDA until `tests/test_qwen_prefix.py` has been run on a GPU node (`NAMES_SHARE_LINEAR_CUDA=1`).
- **The Qwen card is a template**: `runs/names-qwen-all/best/README.md` should be written before release.
