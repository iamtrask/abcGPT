# micro — first BPE + real-data test

**Status**: ⏭ blocked on nano-100. First tier outside the char-level toy regime.

**One-line goal**: prove the k-dim gate that worked at nano-100 char-level still works when we swap char tokenization for GPT-2 BPE and swap synthetic per-cohort data for real Common Crawl web text.

## Why this tier exists

Two things change simultaneously vs. nano-100:
- **Char → BPE tokenization**: vocab grows from ~250 to 50,257. Embedding table dominates parameter count differently. Token IDs no longer correspond to recognizable surface symbols, so the "vocab orthogonality" intuition from the nano series doesn't transfer directly.
- **Synthetic per-cohort data → real Common Crawl web pages**: documents are heterogeneous in length and style within a cohort. URL→domain partitioning means cohorts are publisher-defined, not anchored-corpus-defined.

Either change could break the gate independently. micro tests both at small scale.

## Setup

| field | value |
|---|---|
| data | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) sample-10BT × 1% (~100M tokens, ~150K docs) |
| tokenizer | GPT-2 BPE (`tiktoken`), vocab 50,257 |
| model | ~30M params (configuration TBD; probably 6 layers, 8 heads, 384 embed) |
| cohorts (N) | 20 (top-20 domains by doc count in the 1% slice) |
| gate | k-dim tent, k decided based on nano series results |
| training | ~5000 iters (Karpathy's shakespeare_char config adjusted for BPE) |
| GPU | RTX 4090 spot |
| wall-clock | ~2-4 hours |
| cost | ~$10-20 |

## Data prep

Requires per-tier work — produces both `train.bin` (tokens) and `train_sources.bin` (parallel uint16 cohort IDs). Plan documented in [`../owt_repro/`](../owt_repro/) (now superseded by the FineWeb pivot, but the byte-deterministic prep pattern still applies). Specifically:

1. Stream `HuggingFaceFW/fineweb` sample-10BT, take first 1% (~150K docs).
2. Build URL→domain via `tldextract`. Count docs per domain. Pick top-20 + 'misc' bucket.
3. Tokenize each doc with GPT-2 BPE, append EOT.
4. Write `train.bin` (uint16 token IDs) + `train_sources.bin` (uint16 cohort IDs, parallel array same length as train.bin).
5. Standard 99/1 train/val split (Karpathy's seed=2357 convention).

Prep output: ~200MB (100M tokens × 2 bytes for both token and cohort arrays).

## Reference baselines

- **Combined ungated baseline**: same architecture, same data, no gating. For uniform-α comparison.
- **Per-domain single-source baselines**: would need 20 baselines × maybe 30 min each = ~$3 extra. Worth doing for at least the top-5 domains to have cohort-α reference numbers.

## Pass criteria

- **Uniform-α**: within ~10% of combined ungated baseline.
- **Cohort-α (top-5 domains)**: gated model with α = m_domain hits within seed variance of that domain's single-source baseline.
- **Sanity check on generations**: prompt = "The weather in", generate 50 tokens at α corresponding to nytimes.com vs medium.com vs wikipedia.org — outputs should look visibly different.

## Open questions

- **k-dim setting**: carry over from nano-100? Or rescale?
- **Training iters**: 5000 may be too few given 100M tokens vs nano's 3M chars. Watch val_loss curve to decide.
- **OOV / vocab handling**: char level handled this implicitly (per-corpus union). BPE doesn't need union but the vocab is huge — we use full GPT-2 50K vocab regardless of which cohort's data is being trained on.

## Files

```
experiments/micro/
├── README.md
├── prepare_fineweb_1pct.py     ← data prep (FineWeb 1% slice + URL→domain cohorts)
├── gated_gpt_tent_kdim_bpe.py  ← adapted for BPE (mostly the same as nano series)
├── train.py
├── config.py
└── results/
    ├── per_domain_val_loss.csv
    ├── sample_generations.md
    └── summary.json
```

## Decision after micro

- **Pass** → `mini` (10× data + larger model).
- **Fail uniform-α** → BPE or web-text shift is hurting capacity. Maybe model is too small for the BPE vocab table.
- **Fail cohort-α** → gate isn't capturing per-domain signal in real web text. Could be that domain is too noisy a cohort definition (lots of subdomains, mixed-style sites). Try cohort-by-CC-dump or by URL-pattern.
