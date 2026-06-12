"""Stage 1 — one semantic embedding per URL-host (domain) for FineWeb-10BT.

Streams FineWeb sample-10BT, groups documents by host (strip www.), embeds up to
--cap sampled docs per domain with a sentence encoder and MEAN-POOLS them into one
vector per domain. Also accumulates each domain's total token_count + doc_count
(needed for token-balanced clustering in stage 2) and a few text snippets (for
labeling). Provenance stays atomic: a domain is one source, one vector.

Output (in --out-dir):
  domains.npz   hosts[str N], vecs[float32 N,D] (L2-normalized), tokens[int64 N],
                docs[int64 N], n_emb[int64 N]
  samples.jsonl one line per (largest) domain: {host, tokens, snippets[...]}

Smoke (CPU, first 50k docs):
  uv run --with datasets --with sentence-transformers python embed_domains.py --n-docs 50000
Full pass (GPU pod):
  python embed_domains.py --n-docs 0 --device cuda --batch 256
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def host_of(url):
    h = urlparse(url or "").netloc.lower()
    return h[4:] if h.startswith("www.") else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=50000, help="0 = all (~14.9M)")
    ap.add_argument("--cap", type=int, default=8, help="max docs EMBEDDED per domain (token/doc counts use all)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-chars", type=int, default=2000, help="truncate doc text before embedding")
    ap.add_argument("--out-dir", default="fineweb_cluster_out")
    args = ap.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.model, device=args.device)
    DIM = enc.get_sentence_embedding_dimension()
    print(f"model {args.model} | dim {DIM} | device {args.device}", flush=True)

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    emb_sum = {}                       # host -> float32[DIM] running sum
    emb_count = defaultdict(int)       # host -> #docs embedded (capped at --cap)
    tokens = defaultdict(int)
    docs = defaultdict(int)
    samples = defaultdict(list)

    buf_text, buf_host = [], []
    def flush():
        if not buf_text:
            return
        vecs = enc.encode(buf_text, batch_size=args.batch, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)
        for h, v in zip(buf_host, vecs):
            if h in emb_sum:
                emb_sum[h] += v
            else:
                emb_sum[h] = v.astype(np.float32).copy()
        buf_text.clear(); buf_host.clear()

    n = 0; t0 = time.time()
    for ex in ds:
        url = ex.get("url", "")
        h = host_of(url)
        tokens[h] += int(ex.get("token_count") or max(1, len(ex.get("text", "")) // 4))
        docs[h] += 1
        if emb_count[h] < args.cap:
            buf_text.append((ex.get("text", "") or "")[:args.max_chars])
            buf_host.append(h)
            emb_count[h] += 1
            if len(samples[h]) < 3:
                samples[h].append((ex.get("text", "") or "")[:200].replace("\n", " "))
            if len(buf_text) >= args.batch:
                flush()
        n += 1
        if n % 50000 == 0:
            print(f"  {n:,} docs | {len(docs):,} domains | {time.time()-t0:.0f}s", flush=True)
        if args.n_docs and n >= args.n_docs:
            break
    flush()

    hosts = list(docs.keys())
    H = len(hosts)
    vecs = np.zeros((H, DIM), np.float32)
    tok = np.zeros(H, np.int64); dc = np.zeros(H, np.int64); ne = np.zeros(H, np.int64)
    for i, h in enumerate(hosts):
        c = emb_count[h]
        v = emb_sum.get(h)
        if v is not None and c > 0:
            v = v / c
            nrm = np.linalg.norm(v)
            vecs[i] = v / nrm if nrm > 0 else v          # re-normalize the mean (spherical / cosine)
        tok[i] = tokens[h]; dc[i] = docs[h]; ne[i] = c

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "domains.npz", hosts=np.array(hosts), vecs=vecs, tokens=tok, docs=dc, n_emb=ne)
    with open(out / "samples.jsonl", "w") as f:
        for h in sorted(hosts, key=lambda x: -tokens[x])[:2000]:
            f.write(json.dumps({"host": h, "tokens": int(tokens[h]), "snippets": samples[h]}) + "\n")
    print(f"\nwrote {H:,} domains ({tok.sum()/1e6:.0f}M tokens, {dc.sum():,} docs) -> {out}/domains.npz")
    print(f"  singletons: {(dc==1).sum():,} ({(dc==1).mean()*100:.0f}%) | "
          f"max-token domain: {hosts[int(tok.argmax())]} ({tok.max()/1e6:.1f}M)")


if __name__ == "__main__":
    main()
