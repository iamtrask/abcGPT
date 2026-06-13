"""Stage 4 — tokenize FineWeb into per-CLUSTER gpt2-BPE bins via the domain->cluster map.

Streams FineWeb sample-10BT, routes each doc's gpt2-BPE tokens to its domain's
cluster_id (from clusters.npz), and writes cluster_NN_{train,val}.bin (uint16) + meta.pkl
in the layout train.py expects. Memory-bounded: per-cluster token buffers are flushed to
per-cluster .raw files on the fly (10B tokens can't live in RAM), then split into
train/val at the end.

  python prepare_clustered.py --n-docs 0 --clusters clusters.npz --k 100
"""
import argparse, os, pickle, time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import tiktoken
from datasets import load_dataset


def host_of(url):
    h = urlparse(url or "").netloc.lower()
    return h[4:] if h.startswith("www.") else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=500000, help="0 = all (~14.9M)")
    ap.add_argument("--clusters", default="clusters.npz")
    ap.add_argument("--out-dir", default="data_clustered")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.002)
    ap.add_argument("--flush-tokens", type=int, default=1_000_000, help="per-cluster buffer flush threshold")
    args = ap.parse_args()

    cl = np.load(args.clusters, allow_pickle=True)
    h2c = {str(h): int(c) for h, c in zip(cl["hosts"], cl["cluster"])}
    print(f"cluster map: {len(h2c):,} domains -> {args.k} clusters", flush=True)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token
    raw_f = [open(out / f"cluster_{c:02d}.raw", "wb") for c in range(args.k)]
    bufs = [[] for _ in range(args.k)]
    counts = [0] * args.k

    def flush(c):
        if bufs[c]:
            np.array(bufs[c], dtype=np.uint16).tofile(raw_f[c])
            counts[c] += len(bufs[c]); bufs[c] = []

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    n = 0; miss = 0; t0 = time.time()
    for ex in ds:
        c = h2c.get(host_of(ex.get("url", "")))
        n += 1
        if c is None:
            miss += 1
        else:
            ids = enc.encode(ex.get("text", "") or "", disallowed_special=())  # treat <|...|> as text
            ids.append(eot)
            bufs[c].extend(ids)
            if len(bufs[c]) >= args.flush_tokens:
                flush(c)
        if n % 200000 == 0:
            print(f"  {n:,} docs | {sum(counts)/1e6:.0f}M tok | {miss:,} unmapped | {time.time()-t0:.0f}s", flush=True)
        if args.n_docs and n >= args.n_docs:
            break
    for c in range(args.k):
        flush(c); raw_f[c].close()

    # split each cluster's raw stream into train/val
    names, sizes = [], []
    for c in range(args.k):
        nm = f"cluster_{c:02d}"; names.append(nm)
        arr = np.memmap(out / f"{nm}.raw", dtype=np.uint16, mode="r") if counts[c] else np.zeros(0, np.uint16)
        sizes.append(len(arr))
        nval = max(2, int(len(arr) * args.val_frac)) if len(arr) > 4 else 0
        np.asarray(arr[:nval]).tofile(out / f"{nm}_val.bin")
        np.asarray(arr[nval:]).tofile(out / f"{nm}_train.bin")
        del arr
        try: os.remove(out / f"{nm}.raw")
        except OSError: pass
    sizes = np.array(sizes)
    print(f"tokens/cluster: min {sizes.min():,} median {int(np.median(sizes)):,} "
          f"max {sizes.max():,} total {sizes.sum()/1e6:.0f}M ({miss:,} unmapped docs)")
    if sizes.min() < 4096:
        print(f"  WARNING: smallest cluster {sizes.min()} tokens (<4096) — bump --n-docs")
    meta = {"vocab_size": 50304, "dtype": "uint16", "tokenizer": "gpt2",
            "cohort_names": names, "stoi": None, "itos": None}
    pickle.dump(meta, open(out / "meta.pkl", "wb"))
    print(f"wrote {args.k} cluster bins + meta -> {out}")


if __name__ == "__main__":
    main()
