#!/usr/bin/env python3
"""Check 0: laptop determinism test for the OWT prep pipeline.

Goal: prove that tiktoken + HF datasets + RNG are deterministic on this
machine BEFORE spending GCP money on the full prep. Specifically:

  1. Tokenize the first 100 docs of Skylion007/openwebtext twice in the
     same Python process. Hash the resulting token arrays. They must match.
  2. Run datasets.train_test_split twice with the same seed Karpathy uses
     (2357). The resulting train splits must contain the same documents
     in the same order.

If either check fails, the full prep cannot produce byte-identical output
across runs, and we need to fix nondeterminism before going further.

Run with uv (pulls pinned deps inline, no venv needed):

  uv run --with 'tiktoken==0.7.0' \
         --with 'datasets==2.21.0' \
         --with 'numpy>=1.26,<2' \
         python check_0.py

Expected wall-clock: ~30 seconds. Cost: $0.
"""

import hashlib
import sys

import numpy as np
import tiktoken
from datasets import Dataset, load_dataset

N_DOCS = 100

# Pin a specific dataset revision once we trust one. None = whatever HF
# serves as latest, which is fine for the determinism check but should be
# pinned for Check 1.
REVISION = None


def tokenize_pass(docs):
    enc = tiktoken.get_encoding("gpt2")
    all_ids = []
    for d in docs:
        ids = enc.encode_ordinary(d["text"]) + [enc.eot_token]
        all_ids.extend(ids)
    arr = np.array(all_ids, dtype=np.uint16)
    return hashlib.sha256(arr.tobytes()).hexdigest(), len(arr)


def load_first_n(n):
    """Stream the dataset so we don't download all 38 GB for 100 docs."""
    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
        revision=REVISION,
        trust_remote_code=True,
    )
    docs = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        docs.append({"text": ex["text"]})
    return docs


def split_text_hash(split_ds):
    """Hash the concatenation of text fields in a split, in order."""
    concat = "\n".join(ex["text"] for ex in split_ds)
    return hashlib.sha256(concat.encode("utf-8")).hexdigest()


def main():
    print(f"loading first {N_DOCS} docs of Skylion007/openwebtext (streaming)...")
    docs = load_first_n(N_DOCS)
    mean_len = sum(len(d["text"]) for d in docs) // max(1, len(docs))
    print(f"loaded {len(docs)} docs (mean text length: {mean_len} chars)")

    failed = False

    print("\n--- TOKENIZATION DETERMINISM ---")
    h1, n1 = tokenize_pass(docs)
    h2, n2 = tokenize_pass(docs)
    print(f"  run 1: {n1:>8} tokens   sha256 {h1[:16]}...")
    print(f"  run 2: {n2:>8} tokens   sha256 {h2[:16]}...")
    if h1 == h2:
        print("  PASS")
    else:
        print("  FAIL — tiktoken or RNG is non-deterministic")
        print(f"    full run-1 sha256: {h1}")
        print(f"    full run-2 sha256: {h2}")
        failed = True

    print("\n--- train_test_split DETERMINISM (seed=2357, like Karpathy) ---")
    ds = Dataset.from_list(docs)
    s1 = ds.train_test_split(test_size=0.1, seed=2357, shuffle=True)
    s2 = ds.train_test_split(test_size=0.1, seed=2357, shuffle=True)
    th1 = split_text_hash(s1["train"])
    th2 = split_text_hash(s2["train"])
    vh1 = split_text_hash(s1["test"])
    vh2 = split_text_hash(s2["test"])
    print(f"  run 1: train sha256 {th1[:16]}...  val sha256 {vh1[:16]}...")
    print(f"  run 2: train sha256 {th2[:16]}...  val sha256 {vh2[:16]}...")
    if th1 == th2 and vh1 == vh2:
        print("  PASS")
    else:
        print("  FAIL — train_test_split is non-deterministic at the same seed")
        failed = True

    print("\n--- VERSIONS ---")
    import datasets as _ds
    print(f"  tiktoken: {getattr(tiktoken, '__version__', 'unknown')}")
    print(f"  datasets: {_ds.__version__}")
    print(f"  numpy:    {np.__version__}")
    print(f"  python:   {sys.version.split()[0]}")

    print()
    if failed:
        print("CHECK 0: FAIL — fix nondeterminism before running Check 1")
        sys.exit(1)
    else:
        print("CHECK 0: PASS")
        print("Next: Check 1 — run Karpathy's prepare.py twice on GCP, "
              "verify byte-identical train.bin/val.bin.")


if __name__ == "__main__":
    main()
