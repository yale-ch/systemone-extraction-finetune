"""The name-parsing task as Turion Extract fields, and conversion to and from the dataset's YAML records.

The dataset (small-models-for-glam/synthetic-parsed-names-yaml) answers each name with:

    first_name: Lucie            last_name: Maréchal ('' when there is none)
    middle_names: [...]          titles: [...]          extra_info: [...]
    temporal: [{start, end, type}]   type is life_span or flourished; at most one of each per name

Turion has no nested objects, so `temporal` is flattened into four year fields. Every value in the
dataset is written verbatim in the input, so every field is a span or a list of spans. Years are `text`
fields, not `number`: Turion's number parser reads "Bonifacio Zamora, -1457" (death year only) as -1457,
which would leave 0.7% of death years unreachable. A year span is turned back into an int on the way out.
The field names, specs and order below are part of the trained model: inference must use the same ones,
so a trained model directory carries a copy in `fields.json`.
"""
from __future__ import annotations

import hashlib
import re

FIELDS: dict[str, dict] = {
    "first_name": {"type": "text"},
    "last_name": {"type": "text", "instructions": "Surname, with any particles"},
    "middle_names": {"type": "list", "items": "text"},
    "titles": {"type": "list", "items": "text", "instructions": "Titles and honorifics"},
    "extra_info": {"type": "list", "items": "text", "instructions": "Other details, e.g. a place"},
    "life_start": {"type": "text", "instructions": "Birth year"},
    "life_end": {"type": "text", "instructions": "Death year"},
    "flourished_start": {"type": "text", "instructions": "First year active"},
    "flourished_end": {"type": "text", "instructions": "Last year active"},
}
PROMPT_PREFIX = "Parse this person name:\n\n"
TEMPORAL = {"life_span": ("life_start", "life_end"), "flourished": ("flourished_start", "flourished_end")}
LISTS = ("middle_names", "titles", "extra_info")


def row_id(state: str) -> str:
    return hashlib.sha1(state.encode("utf-8")).hexdigest()[:16]


def split_of(state: str, test_pct: int = 1, dev_pct: int = 1) -> str:
    """Deterministic split by a hash of the input, so a duplicate input never crosses splits."""
    b = int(hashlib.sha1(state.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if b < test_pct else "dev" if b < test_pct + dev_pct else "train"


def script_of(s: str) -> str:
    """Rough writing system of a name, for per-script scores."""
    for ch in s:
        n = ord(ch)
        if 0x0600 <= n <= 0x06FF or 0x0750 <= n <= 0x077F or 0xFB50 <= n <= 0xFDFF or 0xFE70 <= n <= 0xFEFF:
            return "arabic"
        if 0x3400 <= n <= 0x9FFF or 0x3040 <= n <= 0x30FF or 0xAC00 <= n <= 0xD7AF:
            return "cjk"
        if 0x0400 <= n <= 0x04FF:
            return "cyrillic"
        if 0x0370 <= n <= 0x03FF:
            return "greek"
    return "latin"


def record_to_gold(rec: dict) -> dict:
    """Dataset YAML record -> {field: {"value": ...}}; empty strings and empty lists become null."""
    g = {"first_name": {"value": (rec.get("first_name") or None)},
         "last_name": {"value": (rec.get("last_name") or None)}}
    for k in LISTS:
        g[k] = {"value": [str(x) for x in rec.get(k) or []] or None}
    for a, b in TEMPORAL.values():
        g[a] = {"value": None}
        g[b] = {"value": None}
    for t in rec.get("temporal") or []:
        if t.get("type") not in TEMPORAL:
            raise ValueError(f"unknown temporal type {t.get('type')!r}")
        a, b = TEMPORAL[t["type"]]
        g[a] = {"value": t.get("start")}
        g[b] = {"value": t.get("end")}
    return g


def to_year(v) -> int | None:
    """A year span ("1633", "c.1633", "-1457") -> 1633; None when there are no digits."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


NAME_FIELDS = ("first_name", "last_name") + LISTS


def clean_value(v: str) -> str:
    """Trim what a whole-token span drags along: Gemma has '..' and '..,' as single tokens, so a span can't stop
    between the dots of 'C..' and comes back as 'C..' or 'R..,'. The labels say 'C.' (1,586 times in train);
    no training label ends in '..', ',', ';', ':' or '(' (5 of 787,087 had an outer space)."""
    v = re.sub(r"[\s,;:(]+$", "", v.strip())
    return re.sub(r"\.{2,}$", ".", v)


def clean_answers(answers: dict) -> dict:
    """clean_value on every name field; a list drops items that become empty or repeated. Spans are left as the
    model's token spans (they already include a leading space, e.g. ' Ashley')."""
    out = dict(answers)
    for k in NAME_FIELDS:
        a = out.get(k)
        if not a or a.get("value") is None:
            continue
        a = dict(a)
        if isinstance(a["value"], list):
            a["value"] = list(dict.fromkeys(v for v in map(clean_value, a["value"]) if v)) or None
        else:
            a["value"] = clean_value(a["value"]) or None
        out[k] = a
    return out


def answers_to_record(answers: dict) -> dict:
    """Turion answers -> a record in the dataset's YAML shape (same keys, '' for a missing name)."""
    v = lambda k: (answers.get(k) or {}).get("value")
    year = lambda k: to_year(v(k))
    rec = {"first_name": v("first_name") or "", "last_name": v("last_name") or "",
           "middle_names": list(v("middle_names") or [])}
    temporal = []
    for kind, (a, b) in TEMPORAL.items():
        if year(a) is not None or year(b) is not None:
            temporal.append({"start": year(a), "end": year(b), "type": kind})
    rec["temporal"] = temporal
    rec["titles"] = list(v("titles") or [])
    rec["extra_info"] = list(v("extra_info") or [])
    return rec


def same_record(a: dict, b: dict) -> bool:
    """Strict record equality: exact strings, lists in order, temporal entries in any order."""
    key = lambda r: sorted((t.get("type"), t.get("start"), t.get("end")) for t in r.get("temporal") or [])
    return (all((a.get(k) or "") == (b.get(k) or "") for k in ("first_name", "last_name"))
            and all(list(a.get(k) or []) == list(b.get(k) or []) for k in LISTS)
            and key(a) == key(b))


def request_for(state: str) -> dict:
    return {"state": state, "fields": FIELDS}
