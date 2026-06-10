"""Build 100-cohort char-level dataset from the 100 source novels in
data/100_regime_char/sources/source_NNN_*.txt.

Each source file becomes one cohort. Outputs per-cohort {train,val}.bin
(uint8) + meta.pkl with shared vocab + cohort_names.

On a fresh cloud pod where the source .txt files aren't available, fetches
the pre-built bins from HuggingFace at iamtrask/abcGPT-nano-3 under
data_100src/ instead of rebuilding from .txt files.
"""
import os
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES_DIR = ROOT.parent / "100_regime_char/sources"

# If we can find the local .txt sources, build from them.
# Otherwise (fresh cloud pod), fetch pre-built bins tarball from HF.

def build_from_local():
    src_files = sorted(SOURCES_DIR.glob("source_*.txt"))
    if len(src_files) < 100:
        return False
    print(f"Building from {len(src_files)} local source files in {SOURCES_DIR}")
    texts = {}
    for f in src_files:
        # Extract cohort name from filename: source_000_austen-pride-prejudice.txt
        name = f.stem  # e.g. "source_000_austen-pride-prejudice"
        texts[name] = f.read_text(errors="ignore")
    cohort_names = sorted(texts.keys())[:100]   # exactly 100
    print(f"Using {len(cohort_names)} cohorts (first: {cohort_names[0]}, last: {cohort_names[-1]})")
    # Build shared vocab
    all_text = "".join(texts[n] for n in cohort_names)
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    print(f"Shared vocab: {vocab_size} chars")
    if vocab_size > 255:
        sys.exit(f"error: vocab {vocab_size} > 255, would need uint16")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    def encode(s):
        return np.array([stoi[c] for c in s], dtype=np.uint8)
    sizes = []
    for cn in cohort_names:
        ids = encode(texts[cn])
        split = int(0.9 * len(ids))
        tr, val = ids[:split], ids[split:]
        (ROOT / f"{cn}_train.bin").write_bytes(tr.tobytes())
        (ROOT / f"{cn}_val.bin").write_bytes(val.tobytes())
        sizes.append((len(tr), len(val)))
    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "dtype": "uint8",
        "cohort_names": cohort_names,
        "cohort_train_sizes": [s[0] for s in sizes],
        "cohort_val_sizes": [s[1] for s in sizes],
    }
    with open(ROOT / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    total = sum(s[0] for s in sizes)
    print(f"Wrote meta.pkl  vocab={vocab_size}  total train tokens: {total:,}")
    return True


def fetch_from_hf():
    """Fallback for cloud pods: download pre-built tarball from HF."""
    HF_REPO = "iamtrask/abcGPT-nano-3"
    URL = f"https://huggingface.co/{HF_REPO}/resolve/main/data_100src/bins.tar.gz"
    dst = ROOT / "bins.tar.gz"
    print(f"Fetching pre-built bins from {URL}")
    urllib.request.urlretrieve(URL, dst)
    print(f"Extracting to {ROOT}")
    with tarfile.open(dst, "r:gz") as tf:
        tf.extractall(ROOT)
    dst.unlink()
    print("Done.")
    return True


if (ROOT / "meta.pkl").exists():
    print("meta.pkl already present; skipping build.")
elif SOURCES_DIR.exists() and len(list(SOURCES_DIR.glob("source_*.txt"))) >= 100:
    build_from_local()
else:
    fetch_from_hf()
