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


def write_cluster_preview(out_dir, hosts, emb_sum, emb_count, tokens, k, n_seen, dim):
    """Quick semantic-only k-means on the domains seen SO FAR -> cluster_preview.json
    (top domains per cluster). For mid-run visibility; token-balancing is the final
    step, not this. Cheap: MiniBatchKMeans on the growing domain set, ~seconds."""
    import json
    from sklearn.cluster import MiniBatchKMeans
    H = len(hosts)
    if H < k:
        return
    vecs = np.zeros((H, dim), np.float32); tok = np.zeros(H, np.int64)
    for i, h in enumerate(hosts):
        c = emb_count[h]; v = emb_sum.get(h)
        if v is not None and c > 0:
            v = v / c; nrm = np.linalg.norm(v); vecs[i] = v / nrm if nrm > 0 else v
        tok[i] = tokens[h]
    lab = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=2, random_state=0).fit_predict(vecs)
    clusters = {}
    for c in range(k):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        top = idx[np.argsort(-tok[idx])][:6]
        clusters[int(c)] = {"tokens": int(tok[idx].sum()), "n_domains": int(len(idx)),
                            "top_domains": [str(hosts[i]) for i in top]}
    json.dump({"n_docs_seen": n_seen, "n_domains": H, "k": k, "clusters": clusters},
              open(Path(out_dir) / "cluster_preview.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=50000, help="0 = all (~14.9M)")
    ap.add_argument("--cap", type=int, default=8, help="max docs EMBEDDED per domain (token/doc counts use all)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-chars", type=int, default=2000, help="truncate doc text before embedding")
    ap.add_argument("--out-dir", default="fineweb_cluster_out")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="every N docs, write a quick-k-means cluster_preview.json (0 = off)")
    ap.add_argument("--checkpoint-k", type=int, default=100)
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

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
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
        if args.checkpoint_every and n % args.checkpoint_every == 0 and len(docs) >= args.checkpoint_k:
            flush()
            write_cluster_preview(args.out_dir, list(docs.keys()), emb_sum, emb_count, tokens,
                                  args.checkpoint_k, n, DIM)
            print(f"  [checkpoint] cluster_preview.json @ {n:,} docs / {len(docs):,} domains", flush=True)
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
