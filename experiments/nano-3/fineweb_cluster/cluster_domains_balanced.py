"""Stage 2 — cluster domain vectors into K token-balanced, semantically-coherent clusters.

Domains are atomic (a domain can't be split across clusters — that's the provenance
constraint), and token mass is brutally power-law (the top ~2% of domains hold ~50%
of tokens), so plain k-means gives wildly imbalanced clusters. Instead:

  1. over-cluster the domain vectors into --micro micro-clusters (k-means),
  2. greedily bin-pack the micro-clusters into K bins, each step assigning the next
     (largest-token-first) micro-cluster to the bin minimizing
         (projected token fill) + lambda * (cosine distance to bin centroid)
     so bins stay token-balanced AND semantically coherent,
  3. mega-domains whose own token mass exceeds the per-cluster budget are pulled out
     first and each given its own bin (they'd swamp any cluster otherwise).

Output (in --out-dir):
  clusters.npz        hosts[str N], cluster[int32 N]
  cluster_labels.json {cluster_id: {n_domains, tokens, top_domains[...]}}

  uv run --with numpy --with scikit-learn python cluster_domains_balanced.py --k 100
"""
import argparse, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="fineweb_cluster_out")
    ap.add_argument("--out-dir", default="fineweb_cluster_out")
    ap.add_argument("--k", type=int, default=100, help="final number of clusters")
    ap.add_argument("--micro", type=int, default=2000, help="micro-clusters (k')")
    ap.add_argument("--lam", type=float, default=0.5, help="semantic-coherence vs token-balance tradeoff")
    ap.add_argument("--mega-frac", type=float, default=1.0,
                    help="a domain with tokens > mega-frac * per-cluster-budget gets its own cluster")
    args = ap.parse_args()

    d = np.load(Path(args.in_dir) / "domains.npz", allow_pickle=True)
    hosts = d["hosts"]; vecs = d["vecs"].astype(np.float32); tokens = d["tokens"].astype(np.int64)
    H = len(hosts); total = int(tokens.sum())
    budget = total / args.k
    print(f"{H:,} domains | {total/1e6:.0f}M tokens | per-cluster budget ~{budget/1e6:.1f}M", flush=True)

    cluster = np.full(H, -1, np.int32)
    next_bin = 0

    # --- 1. pull out mega-domains: each gets its own cluster ---
    mega = np.where(tokens > args.mega_frac * budget)[0]
    for i in mega:
        cluster[i] = next_bin; next_bin += 1
    if len(mega):
        print(f"mega-domains (own cluster): {len(mega)} -> "
              + ", ".join(f"{hosts[i]}({tokens[i]/1e6:.0f}M)" for i in mega[np.argsort(-tokens[mega])][:6]))
    k_remaining = max(1, args.k - next_bin)
    rest = np.where(cluster < 0)[0]

    # --- 2. micro-cluster the remaining domains ---
    from sklearn.cluster import MiniBatchKMeans
    micro = int(min(args.micro, len(rest)))
    km = MiniBatchKMeans(n_clusters=micro, batch_size=4096, n_init=3, random_state=0)
    mlab_rest = km.fit_predict(vecs[rest])
    mcent = np.zeros((micro, vecs.shape[1]), np.float32); mtok = np.zeros(micro, np.int64)
    for m in range(micro):
        idx = rest[mlab_rest == m]
        if len(idx) == 0:
            continue
        w = tokens[idx]; mtok[m] = w.sum()
        c = (vecs[idx] * w[:, None]).sum(0) / max(1, w.sum())
        nrm = np.linalg.norm(c); mcent[m] = c / nrm if nrm > 0 else c

    # --- 3. greedy token-balanced + semantic bin-pack into k_remaining bins ---
    base = next_bin
    bin_tok = np.zeros(k_remaining); bin_cent = np.zeros((k_remaining, vecs.shape[1]), np.float32)
    micro2bin = np.zeros(micro, int)
    for m in np.argsort(-mtok):
        if mtok[m] == 0:
            micro2bin[m] = 0; continue
        cent_norm = np.linalg.norm(bin_cent, axis=1)
        cos = (bin_cent @ mcent[m]) / (cent_norm + 1e-9)
        dist = np.where(cent_norm > 0, 1.0 - cos, 1.0)          # empty bins => neutral distance
        cost = (bin_tok + mtok[m]) / budget + args.lam * dist
        b = int(np.argmin(cost))
        nt = bin_tok[b] + mtok[m]
        bin_cent[b] = (bin_cent[b] * bin_tok[b] + mcent[m] * mtok[m]) / max(1.0, nt)
        bin_tok[b] = nt; micro2bin[m] = b
    cluster[rest] = base + micro2bin[mlab_rest]

    # --- output + labels ---
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "clusters.npz", hosts=hosts, cluster=cluster)
    labels = {}
    for c in range(int(cluster.max()) + 1):
        idx = np.where(cluster == c)[0]
        if len(idx) == 0:
            continue
        top = idx[np.argsort(-tokens[idx])][:8]
        labels[int(c)] = {"n_domains": int(len(idx)), "tokens": int(tokens[idx].sum()),
                          "top_domains": [str(hosts[i]) for i in top]}
    json.dump(labels, open(out / "cluster_labels.json", "w"), indent=1)

    tks = np.array([labels[c]["tokens"] for c in labels])
    print(f"\n{len(labels)} clusters | token mass: min {tks.min()/1e6:.1f}M  "
          f"median {np.median(tks)/1e6:.1f}M  max {tks.max()/1e6:.1f}M  "
          f"(balance ratio max/min = {tks.max()/max(1,tks.min()):.1f}x)")
    print("\ncluster | tokens | #domains | top domains")
    for c in sorted(labels, key=lambda c: -labels[c]["tokens"]):
        L = labels[c]
        print(f"  c{c:>3} {L['tokens']/1e6:7.1f}M {L['n_domains']:>7}  {', '.join(L['top_domains'][:5])}")
    print(f"\nwrote {out}/clusters.npz + cluster_labels.json")


if __name__ == "__main__":
    main()
