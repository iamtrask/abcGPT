#!/usr/bin/env python3
"""Vocab statistics for the 100_simple_voices corpus.

Per-source: bytes, words, vocab, TTR, Zipf alpha, distinctive words/bigrams,
KL divergence vs combined baseline. Output: ranked CSV + summary table for
"best demos well" criterion (small vocab + high distinctness + enough size).
"""

import json, re, math
from collections import Counter
from pathlib import Path

SRC_DIR = Path('/Users/atrask/Desktop/abcGPT/data/100_simple_voices/sources')
OUT_DIR = Path('/Users/atrask/Desktop/abcGPT/data/100_simple_voices')

# Word tokenizer: alphabetic runs, lowercased. Single quote allowed for "don't".
TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")

def tokenize(text):
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]

def bigrams(tokens):
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]

def fit_zipf(freqs, k=1000):
    """Return alpha = slope of log(freq) vs log(rank) over top k."""
    if len(freqs) < 10:
        return 0.0
    top = sorted(freqs, reverse=True)[:k]
    n = len(top)
    xs = [math.log(i+1) for i in range(n)]
    ys = [math.log(f) for f in top]
    sx = sum(xs); sy = sum(ys)
    sxy = sum(x*y for x,y in zip(xs, ys))
    sx2 = sum(x*x for x in xs)
    denom = n*sx2 - sx*sx
    if denom == 0:
        return 0.0
    slope = (n*sxy - sx*sy) / denom
    return -slope

def stats_for(path):
    raw = path.read_text(encoding='utf-8', errors='replace')
    bytes_count = len(raw.encode('utf-8'))
    tokens = tokenize(raw)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return None
    vocab = Counter(tokens)
    bg = Counter(bigrams(tokens))
    n_vocab = len(vocab)
    n_bigrams = len(bg)
    ttr = n_vocab / n_tokens

    # Vocabulary growth: what fraction of vocab is hapaxes (appearing once)?
    hapax = sum(1 for v in vocab.values() if v == 1)
    hapax_frac = hapax / max(1, n_vocab)

    zipf_alpha = fit_zipf(list(vocab.values()))
    avg_word_len = sum(len(t) for t in tokens) / n_tokens
    punct_density = sum(1 for c in raw if c in '.,;:!?-()[]"\'') / max(1, len(raw))
    newline_density = raw.count('\n') / max(1, len(raw))
    caps_density = len(re.findall(r'\b[A-Z]{3,}\b', raw)) / max(1, n_tokens)

    return {
        'name': path.stem,
        'bytes': bytes_count,
        'tokens': n_tokens,
        'vocab': n_vocab,
        'bigram_types': n_bigrams,
        'ttr': round(ttr, 4),
        'hapax_frac': round(hapax_frac, 3),
        'zipf_alpha': round(zipf_alpha, 3),
        'avg_word_len': round(avg_word_len, 2),
        'punct_density': round(punct_density, 4),
        'newline_density': round(newline_density, 4),
        'caps_density': round(caps_density, 4),
        '_vocab': vocab,
        '_bigrams': bg,
    }

# Process all sources
files = sorted(SRC_DIR.glob('source_*.txt'))
print(f"Processing {len(files)} sources...", flush=True)
all_stats = [s for s in (stats_for(p) for p in files) if s is not None]
print(f"OK: {len(all_stats)} sources processed.", flush=True)

# Combined baseline
print("Building baseline...", flush=True)
combined_vocab = Counter()
combined_bg = Counter()
for s in all_stats:
    combined_vocab.update(s['_vocab'])
    combined_bg.update(s['_bigrams'])
N_vocab = sum(combined_vocab.values())
N_bg = sum(combined_bg.values())
baseline_p = {w: c/N_vocab for w, c in combined_vocab.items()}
baseline_bp = {b: c/N_bg for b, c in combined_bg.items()}

# KL vs baseline (word-level) — measures how distinctive this source's vocab dist is
def kl(local, baseline_p):
    total = sum(local.values())
    out = 0.0
    for w, c in local.items():
        p = c/total
        q = baseline_p.get(w, 1e-12)
        if p > 0:
            out += p * math.log(p/q)
    return out

for s in all_stats:
    s['kl_vs_baseline'] = round(kl(s['_vocab'], baseline_p), 3)

# Distinctive content
def distinctive(local, baseline_p, baseline_total, k=15, min_count=5, min_len=3):
    """Words over-represented vs baseline; sort by ratio."""
    total = sum(local.values())
    out = []
    for w, c in local.items():
        if c < min_count or len(w) < min_len: continue
        local_p = c/total
        global_p = baseline_p.get(w, 1/baseline_total)
        out.append((w, c, local_p/global_p))
    out.sort(key=lambda x: -x[2])
    return out[:k]

def distinctive_bigrams(local, baseline_p, baseline_total, k=15, min_count=3):
    total = sum(local.values())
    out = []
    for b, c in local.items():
        if c < min_count: continue
        local_p = c/total
        global_p = baseline_p.get(b, 1/baseline_total)
        out.append((b, c, local_p/global_p))
    out.sort(key=lambda x: -x[2])
    return out[:k]

for s in all_stats:
    s['top_words']   = [(w, c, round(r,1)) for w, c, r in distinctive(s['_vocab'], baseline_p, N_vocab)]
    s['top_bigrams'] = [(b, c, round(r,1)) for b, c, r in distinctive_bigrams(s['_bigrams'], baseline_bp, N_bg)]

# Demo-score: small vocab + high distinctness + adequate size
# - vocab penalty: 1/log(vocab+1) — smaller vocab → higher score
# - distinctness bonus: kl_vs_baseline
# - size threshold: tokens normalized to [0,1] capped at 100K (anything beyond doesn't help)
for s in all_stats:
    vocab_term  = 1.0 / max(1.0, math.log(s['vocab']+1))
    distinct_term = s['kl_vs_baseline']
    size_term  = min(1.0, s['tokens'] / 100_000)
    s['demo_score'] = round(vocab_term * distinct_term * size_term * 100, 2)

# Drop heavy fields before save
for s in all_stats:
    del s['_vocab']
    del s['_bigrams']

# Save
with open(OUT_DIR / 'vocab_stats.json', 'w') as f:
    json.dump(all_stats, f, indent=2)
print(f"Saved {OUT_DIR / 'vocab_stats.json'}", flush=True)

# Print: sorted by demo_score
print()
print(f"{'rank':>4}  {'name':<46}  {'bytes':>9}  {'tokens':>8}  {'vocab':>7}  {'ttr':>5}  {'kl':>5}  {'demo':>5}")
print('-' * 105)
ranked = sorted(all_stats, key=lambda x: -x['demo_score'])
for i, s in enumerate(ranked):
    name = s['name'].replace('source_', '')[:46]
    print(f"{i+1:>4}  {name:<46}  {s['bytes']:>9}  {s['tokens']:>8}  {s['vocab']:>7}  {s['ttr']:>5.3f}  {s['kl_vs_baseline']:>5.2f}  {s['demo_score']:>5.1f}")

print()
print("TOP 20 BY DEMO SCORE (small vocab × high distinctness × adequate size)")
print()
for s in ranked[:20]:
    name = s['name'].replace('source_', '')
    top_words = ', '.join(w for w,_,_ in s['top_words'][:6])
    print(f"  {name}")
    print(f"     vocab={s['vocab']:>6}  tokens={s['tokens']:>7}  ttr={s['ttr']:.3f}  kl={s['kl_vs_baseline']:.2f}  demo={s['demo_score']:.1f}")
    print(f"     distinctive words: {top_words}")
    print()
