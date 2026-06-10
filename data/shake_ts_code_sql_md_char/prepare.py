"""Combine 5 distinct char-level cohorts (extends shake_ts_code_char to N=5).

Cohorts:
    1. shake     — Shakespeare drama
    2. ts        — TinyStories prose
    3. code      — Python source (cpython Lib)
    4. sql       — PostgreSQL regression test queries
    5. md        — Markdown documentation (Bootstrap docs)

Each is a distinct surface form (speaker tags, child-narrative grammar,
indented Python, ALL-CAPS-keyword SQL, # heading + ** bold markdown).

Self-downloads sources on first run; idempotent. Writes per-cohort
{cohort}_{train,val}.bin (uint8) + meta.pkl with shared vocab.
"""
import os
import pickle
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "_src"
SRC_DIR.mkdir(exist_ok=True)

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
TINYSTORIES_BYTES = 1_500_000

CPYTHON_FILES = [
    "Lib/typing.py", "Lib/functools.py", "Lib/argparse.py",
    "Lib/inspect.py", "Lib/asyncio/base_events.py",
]
CPYTHON_BASE = "https://raw.githubusercontent.com/python/cpython/v3.12.0/"

POSTGRES_SQL_FILES = [
    "aggregates.sql", "create_table.sql", "foreign_key.sql", "insert.sql",
    "join.sql", "select.sql", "subselect.sql", "triggers.sql", "window.sql",
]
POSTGRES_BASE = "https://raw.githubusercontent.com/postgres/postgres/REL_16_1/src/test/regress/sql/"

BOOTSTRAP_MD_FILES = [
    "buttons.md", "carousel.md", "dropdowns.md", "modal.md", "navbar.md",
]
BOOTSTRAP_BASE = "https://raw.githubusercontent.com/twbs/bootstrap/v5.3.2/site/content/docs/5.3/components/"


def _download(url, dst, range_bytes=None):
    if dst.exists():
        return
    print(f"  fetching {url} -> {dst.name}")
    req = urllib.request.Request(url)
    if range_bytes is not None:
        req.add_header("Range", f"bytes=0-{range_bytes - 1}")
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        f.write(r.read())


def _normalize_shake(text):
    text = text.lower()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    text = re.sub(r"\s+([.,!?;:])\s+", r" \1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text


# Download shake + ts + python (same as N=3 prep)
shake_path = SRC_DIR / "shakespeare.txt"
_download(SHAKESPEARE_URL, shake_path)
shake_text = _normalize_shake(shake_path.read_text())

ts_path = SRC_DIR / "tinystories.txt"
_download(TINYSTORIES_URL, ts_path, range_bytes=TINYSTORIES_BYTES)
ts_text = ts_path.read_text(errors="ignore")

code_text = ""
for relpath in CPYTHON_FILES:
    name = relpath.replace("/", "_")
    dst = SRC_DIR / name
    _download(CPYTHON_BASE + relpath, dst)
    code_text += dst.read_text() + "\n"

# Download sql (PostgreSQL regression queries)
sql_text = ""
for name in POSTGRES_SQL_FILES:
    dst = SRC_DIR / f"sql_{name}"
    _download(POSTGRES_BASE + name, dst)
    sql_text += dst.read_text() + "\n"

# Download markdown (Bootstrap docs)
md_text = ""
for name in BOOTSTRAP_MD_FILES:
    dst = SRC_DIR / f"md_{name}"
    _download(BOOTSTRAP_BASE + name, dst)
    md_text += dst.read_text() + "\n"

print(f"\nRaw sizes:")
print(f"  shake:  {len(shake_text):>10,} chars")
print(f"  ts:     {len(ts_text):>10,} chars")
print(f"  code:   {len(code_text):>10,} chars")
print(f"  sql:    {len(sql_text):>10,} chars")
print(f"  md:     {len(md_text):>10,} chars")

all_text = shake_text + ts_text + code_text + sql_text + md_text
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print(f"\nShared char vocab: {vocab_size} unique chars")
dtype = np.uint8 if vocab_size <= 255 else np.uint16
print(f"  dtype: {dtype.__name__}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return np.array([stoi[c] for c in s], dtype=dtype)


def split_save(text, name, val_frac=0.1):
    ids = encode(text)
    split = int((1.0 - val_frac) * len(ids))
    train, val = ids[:split], ids[split:]
    (ROOT / f"{name}_train.bin").write_bytes(train.tobytes())
    (ROOT / f"{name}_val.bin").write_bytes(val.tobytes())
    return len(train), len(val)


print("\nEncoding + saving each cohort (90/10 train/val):")
sizes = {}
for name, text in [("shake", shake_text), ("ts", ts_text), ("code", code_text),
                     ("sql", sql_text), ("md", md_text)]:
    tr, va = split_save(text, name)
    sizes[name] = (tr, va)
    print(f"  {name}: train={tr:,} val={va:,}")

meta = {
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
    "dtype": dtype.__name__,
    "cohort_names": ["shake", "ts", "code", "sql", "md"],
    "cohort_train_sizes": [sizes[c][0] for c in ["shake", "ts", "code", "sql", "md"]],
    "cohort_val_sizes":   [sizes[c][1] for c in ["shake", "ts", "code", "sql", "md"]],
}
with open(ROOT / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)
print(f"\nWrote {ROOT}/meta.pkl  (vocab_size={vocab_size})")
print(f"Total train tokens: {sum(s[0] for s in sizes.values()):,}")
