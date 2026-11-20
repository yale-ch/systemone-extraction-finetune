"""Batched forward passes, prediction and metrics, shared by training and inference.

turion_extract's Extractor runs one request per forward pass. A name request is only ~270 tokens, so batching
many requests per pass is far faster. Requests are right-padded: positions and causal attention for real tokens
are unchanged, so each request's hidden states (and therefore the heads' positions) match the unbatched pass.
For prediction, the instructions and field list before the name are run once and shared (SharedPrefix).
"""
from __future__ import annotations

import collections
import json
import os
from dataclasses import dataclass

import torch
from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

from turion_extract.derive import Target, derive
from turion_extract.layout import Encoded, Layout, TooLong
from turion_extract.model import decode, field_losses
from turion_extract.score import score_field, score_rows

from .fields import FIELDS, answers_to_record, clean_answers, request_for, same_record


@dataclass
class Item:
    row: dict
    request: dict
    enc: Encoded
    targets: list[Target] | None = None

    @property
    def n_tokens(self) -> int:
        return len(self.enc.input_ids)

    @property
    def n_fields(self) -> int:
        return sum(t.learnable for t in self.targets or [])


def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def encode_rows(layout: Layout, rows: list[dict], with_targets: bool = True) -> tuple[list[Item], int]:
    """Rows -> encoded items (with loss targets when the row has gold). Returns the items and the number skipped."""
    items, skipped = [], 0
    for r in rows:
        req = request_for(r["state"])
        try:
            e = layout.encode(req)
        except TooLong:
            skipped += 1
            continue
        tg = derive(e, req, r["gold"]) if with_targets and "gold" in r else None
        items.append(Item(r, req, e, tg))
    return items, skipped


def micro_batches(items: list[Item], micro_tokens: int, skip: int = 0) -> list[list[Item]]:
    """Consecutive groups whose padded size (batch x longest) stays within micro_tokens. `skip`: leading tokens
    that are not run (a shared prefix) and so don't count."""
    out, cur, longest = [], [], 0
    for it in items:
        L = max(longest, it.n_tokens - skip)
        if cur and L * (len(cur) + 1) > micro_tokens:
            out.append(cur)
            cur, L = [], it.n_tokens - skip
        cur.append(it)
        longest = L
    if cur:
        out.append(cur)
    return out


def device_type(device) -> str:
    return device.type if isinstance(device, torch.device) else str(device).split(":")[0]


class SharedPrefix:
    """The tokens before the name (instructions and field list, 141 of ~241) are identical in every request, and
    causal attention means nothing after them changes them. So their keys and values are computed once, and a batch
    runs only its own tokens (the name and the query lines) against them: about 2.4x less work, the same result.

    Uses transformers' cache as in chunked prefill (positions, sliding-window masks and offsets come from it). Each
    layer's `update` is replaced so a batch reads the prefix keys/values (broadcast over the batch) without storing
    its own: the cache never grows and is reused unchanged by every batch.

    Linear-attention layers (Qwen3.5's Gated DeltaNet: a causal conv and a recurrent state instead of keys and
    values) are shared the same way. transformers continues a multi-token chunk from a cached state correctly: the
    cached conv inputs (the last `kernel` positions) are put in front of the chunk's, and the chunked delta rule
    starts from the cached recurrent state. Here both updates are replaced by ones that store nothing, so every
    batch starts from the prefix's final states. The prefix's recurrent state stays one [1, heads, k, v] tensor that
    broadcasts over the batch; only inside a layer's forward does each item get its own (for Qwen3.5-4B, 32 x 128 x
    128 fp32 = 2 MB per item, with a few such tensors alive while the layer runs and none after it returns: on
    the order of 1.5 GB for a batch of 256, not 24 layers' worth). Pads sit after the real tokens, and the conv and
    the recurrence are causal, so pads change nothing.
    """

    def __init__(self, backbone, ids: list[int], device):
        out = backbone(input_ids=torch.tensor([ids], device=device), use_cache=True)
        self.ids, self.n = list(ids), len(ids)
        self.hidden = out.last_hidden_state[0]                      # [n, d]: the prefix positions, for the heads
        self.cache = out.past_key_values
        for layer in self.cache.layers:
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                # Checked against the reference (CPU/MPS) delta rule only. On CUDA, flash-linear-attention's kernels
                # take over and may index the batch-1 initial state per item, so sharing stays off there until checked
                # on a GPU node (set NAMES_SHARE_LINEAR_CUDA=1 to try it); predict then runs whole requests.
                if device_type(device) == "cuda" and not os.environ.get("NAMES_SHARE_LINEAR_CUDA"):
                    raise RuntimeError("linear-attention prefix sharing is not verified with the CUDA kernels")
                if isinstance(layer, CacheLayerMixin) or layer.number_of_states != 1 or not layer.has_previous_state[0] \
                        or layer.recurrent_states[0] is None:
                    raise RuntimeError(f"prefix sharing does not handle the cache layer {type(layer).__name__}")
                layer.update_conv_state = self._read_only_conv(layer.conv_states[0])
                layer.update_recurrent_state = lambda recurrent_states, *args, **kwargs: recurrent_states
                # Keeps a one-token batch off the module's single-step path, which writes the conv state in place.
                layer.record_past = True
            elif getattr(layer, "keys", None) is None:
                raise RuntimeError("a cache layer holds no keys (KV-shared layers?); prefix sharing does not apply")
            else:
                layer.update = self._read_only(layer.keys, layer.values)

    @staticmethod
    def _read_only(k: torch.Tensor, v: torch.Tensor):
        def update(key_states, value_states, *args, **kwargs):
            b = key_states.shape[0]
            return (torch.cat([k.expand(b, -1, -1, -1), key_states], dim=-2),
                    torch.cat([v.expand(b, -1, -1, -1), value_states], dim=-2))
        return update

    @staticmethod
    def _read_only_conv(conv: torch.Tensor):
        """conv: the prefix's last conv inputs [1, channels, kernel]. The module drops the outputs at these
        positions, so each of the batch's positions sees its true left context."""
        def update_conv_state(conv_states, *args, **kwargs):
            return torch.cat([conv.expand(conv_states.shape[0], -1, -1).to(conv_states.dtype), conv_states], dim=-1)
        return update_conv_state

    @staticmethod
    def common(items: list[Item]) -> list[int] | None:
        """The token ids before the state, if every item has the same ones."""
        if not items:
            return None
        n = items[0].enc.state_start
        ids = items[0].enc.input_ids[:n]
        same = all(it.enc.state_start == n and it.enc.input_ids[:n] == ids for it in items)
        return ids if same and n > 0 else None


def hidden_states(backbone, items: list[Item], device, pad_id: int, prefix: SharedPrefix | None = None) -> torch.Tensor:
    """Last hidden states [batch, longest, d], right-padded. With `prefix`, only the tokens after it are run."""
    P = prefix.n if prefix else 0
    T = max(it.n_tokens for it in items) - P
    ids = torch.full((len(items), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), P + T), dtype=torch.long)
    mask[:, :P] = 1
    for b, it in enumerate(items):
        ids[b, :it.n_tokens - P] = torch.tensor(it.enc.input_ids[P:])
        mask[b, P:it.n_tokens] = 1
    if prefix is None:
        out = backbone(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
        return out.last_hidden_state
    out = backbone(input_ids=ids.to(device), attention_mask=mask.to(device), past_key_values=prefix.cache, use_cache=True)
    return torch.cat([prefix.hidden.unsqueeze(0).expand(len(items), -1, -1), out.last_hidden_state], dim=1)


def batch_loss(backbone, heads, items: list[Item], device, pad_id: int) -> tuple[torch.Tensor, int]:
    """Summed field negative log-likelihood over a micro-batch, and the number of learnable fields."""
    h = hidden_states(backbone, items, device, pad_id)
    total, n = h.new_zeros((), dtype=torch.float32), 0
    for b, it in enumerate(items):
        loss, k = field_losses(heads(h[b, :it.n_tokens], it.enc), it.enc, it.targets)
        total = total + loss
        n += k
    return total, n


@torch.no_grad()
def predict(backbone, heads, items: list[Item], device, pad_id: int, micro_tokens: int,
            share_prefix: bool = True) -> tuple[dict, float | None]:
    """Answers per row id ({"answers": ...}) and the mean field loss over items that have targets.

    Items are batched in length order (less padding). With share_prefix, the common tokens before the state are
    run once (SharedPrefix), and micro_tokens counts only the tokens each batch runs."""
    ids = SharedPrefix.common(items) if share_prefix else None
    try:
        prefix = SharedPrefix(backbone, ids, device) if ids else None
    except RuntimeError:          # a cache layout SharedPrefix does not handle: run whole requests
        prefix = None
    return predict_with(lambda mb: hidden_states(backbone, mb, device, pad_id, prefix), heads, items, micro_tokens,
                        skip=prefix.n if prefix else 0)


@torch.no_grad()
def predict_with(hidden, heads, items: list[Item], micro_tokens: int, skip: int = 0) -> tuple[dict, float | None]:
    """predict() for any backbone: `hidden(batch)` returns the batch's last hidden states [batch, longest, d] as a
    torch tensor on the heads' device (names_ft.mlx_backbone uses this with MLX)."""
    answers, loss_sum, n_sum = {}, 0.0, 0
    items = sorted(items, key=lambda it: it.n_tokens)
    for mb in micro_batches(items, micro_tokens, skip=skip):
        h = hidden(mb)
        for b, it in enumerate(mb):
            out = heads(h[b, :it.n_tokens], it.enc)
            answers[it.row["id"]] = {"answers": clean_answers(decode(out, it.enc, it.request))}
            if it.targets is not None:
                loss, k = field_losses(out, it.enc, it.targets)
                loss_sum += float(loss)
                n_sum += k
    return answers, (loss_sum / n_sum if n_sum else None)


def metrics(rows: list[dict], answers: dict) -> dict:
    """Index-style scores (per field type and per script) plus per-field accuracy and strict record match.

    `record_exact` is the share of names whose whole parsed record equals the dataset's answer exactly
    (strings, list order, years); it is the number to compare with a generative parser's accuracy.
    """
    srows = [{"id": r["id"], "source": r["source"], "request": request_for(r["state"]), "gold": r["gold"]} for r in rows]
    out = score_rows(srows, answers)
    per_field, exact, field_exact = collections.defaultdict(list), [], collections.defaultdict(list)
    for r in rows:
        got = (answers.get(r["id"]) or {}).get("answers") or {}
        for name, spec in FIELDS.items():
            per_field[name].append(score_field(spec, r["gold"][name], got.get(name)))
        pred = answers_to_record(got)
        exact.append(same_record(pred, r["record"]))
    mean = lambda xs: round(100 * sum(xs) / len(xs), 2) if xs else None
    out["fields_score"] = {k: mean(v) for k, v in per_field.items()}
    out["field_mean"] = mean([x for v in per_field.values() for x in v])
    out["record_exact"] = mean(exact)
    out["rows"] = len(rows)
    return out
