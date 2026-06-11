"""Combine Shakespeare + TinyStories + Python (cpython Lib) into a 3-cohort char-level corpus.

Downloads all three source corpora on first run (idempotent: skips re-download
if cached). Writes one {cohort}_{train,val}.bin per cohort (uint8) + a shared
meta.pkl that lists cohort_names. Designed to run on a fresh RunPod pod
with no other prep done first.

Layout (separate bins per cohort) is cleaner than the nano-2 "single train.bin
with separator" scheme for N >= 3 — lets the trainer sample each cohort
uniformly without needing to find separator positions.

Output:
    data/shake_ts_code_char/{shake,ts,code}_{train,val}.bin
    data/shake_ts_code_char/meta.pkl

Cached source files (committed to repo for offline use):
    data/shake_ts_code_char/_src/{shakespeare.txt, tinystories.txt, {cpython_files}.py}
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

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStories-train.txt"
)
TINYSTORIES_BYTES = 1_500_000  # ~1.5 MB

CPYTHON_FILES = [
    "Lib/typing.py",
    "Lib/functools.py",
    "Lib/argparse.py",
    "Lib/inspect.py",
    "Lib/asyncio/base_events.py",
]
CPYTHON_BASE = "https://raw.githubusercontent.com/python/cpython/v3.12.0/"


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
    """Lowercase + space-padded punctuation to match the TinyStories surface form,
    but PRESERVE the play's newlines. (The old version used \\s (which includes
    \\n) when collapsing whitespace around punctuation, so it ate the line-ending
    newlines and flattened the play. We collapse only horizontal whitespace.)"""
    text = text.lower()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)          # space-pad punctuation
    text = re.sub(r"[ \t]+", " ", text)                  # collapse spaces/tabs ONLY (keep \n)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)         # trim spaces around newlines
    return text


# 1. Download shake (~1.1 MB)
shake_path = SRC_DIR / "shakespeare.txt"
_download(SHAKESPEARE_URL, shake_path)
shake_text = _normalize_shake(shake_path.read_text())

# 2. Download tinystories (first 1.5 MB)
ts_path = SRC_DIR / "tinystories.txt"
_download(TINYSTORIES_URL, ts_path, range_bytes=TINYSTORIES_BYTES)
ts_text = ts_path.read_text(errors="ignore")

# 3. Download cpython Lib files
code_text = ""
for relpath in CPYTHON_FILES:
    name = relpath.replace("/", "_")
    dst = SRC_DIR / name
    _download(CPYTHON_BASE + relpath, dst)
    code_text += dst.read_text() + "\n"

print(f"\nRaw sizes:")
print(f"  shake:  {len(shake_text):>10,} chars")
print(f"  ts:     {len(ts_text):>10,} chars")
print(f"  code:   {len(code_text):>10,} chars  ({len(CPYTHON_FILES)} files)")

all_text = shake_text + ts_text + code_text
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
shake_train, shake_val = split_save(shake_text, "shake")
ts_train, ts_val = split_save(ts_text, "ts")
code_train, code_val = split_save(code_text, "code")
for name, tr, va in [("shake", shake_train, shake_val),
                       ("ts", ts_train, ts_val),
                       ("code", code_train, code_val)]:
    print(f"  {name}: train={tr:,} val={va:,}")

meta = {
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
    "dtype": dtype.__name__,                     # 'uint8' / 'uint16' (plain string)
    "cohort_names": ["shake", "ts", "code"],
    "cohort_train_sizes": [shake_train, ts_train, code_train],
    "cohort_val_sizes": [shake_val, ts_val, code_val],
}
with open(ROOT / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)
print(f"\nWrote {ROOT}/meta.pkl  (vocab_size={vocab_size})")
print(f"Total train tokens: {shake_train + ts_train + code_train:,}")
