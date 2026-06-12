"""Stage 4 — tokenize FineWeb into per-CLUSTER gpt2-BPE bins via the domain->cluster map.

Streams FineWeb sample-10BT, routes each doc's gpt2-BPE tokens to its domain's
cluster_id (from clusters.npz, the stage-2 output), and writes cluster_NN_{train,val}.bin
(uint16) + meta.pkl in the exact layout train.py/load_meta_and_bins expects. This is the
real token-level training substrate for the 100-cluster slider.

  python prepare_clustered.py --n-docs 500000 --clusters clusters.npz --k 100
"""
import argparse, pickle
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
    args = ap.parse_args()

    cl = np.load(args.clusters, allow_pickle=True)
    h2c = {str(h): int(c) for h, c in zip(cl["hosts"], cl["cluster"])}
    print(f"cluster map: {len(h2c):,} domains -> {args.k} clusters", flush=True)

    enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token
    streams = [[] for _ in range(args.k)]
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    n = 0; miss = 0
    for ex in ds:
        c = h2c.get(host_of(ex.get("url", "")))
        n += 1
        if c is None:
            miss += 1
        else:
            ids = enc.encode(ex.get("text", "") or "", allowed_special=set())
            streams[c].extend(ids); streams[c].append(eot)
        if n % 100000 == 0:
            print(f"  {n:,} docs ({miss:,} unmapped)", flush=True)
        if args.n_docs and n >= args.n_docs:
            break

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    names, sizes = [], []
    for c in range(args.k):
        toks = np.array(streams[c], dtype=np.uint16)
        nm = f"cluster_{c:02d}"; names.append(nm); sizes.append(len(toks))
        nval = max(2, int(len(toks) * args.val_frac))
        toks[nval:].tofile(out / f"{nm}_train.bin")
        toks[:nval].tofile(out / f"{nm}_val.bin")
    sizes = np.array(sizes)
    print(f"tokens/cluster: min {sizes.min():,} median {int(np.median(sizes)):,} "
          f"max {sizes.max():,} total {sizes.sum()/1e6:.0f}M ({miss:,} unmapped docs)")
    if sizes.min() < 4096:
        print(f"  WARNING: smallest cluster has {sizes.min()} tokens (<4096) — bump --n-docs")
    meta = {"vocab_size": 50304, "dtype": "uint16", "tokenizer": "gpt2",
            "cohort_names": names, "stoi": None, "itos": None}
    pickle.dump(meta, open(out / "meta.pkl", "wb"))
    print(f"wrote {args.k} cluster bins + meta -> {out}")


if __name__ == "__main__":
    main()
