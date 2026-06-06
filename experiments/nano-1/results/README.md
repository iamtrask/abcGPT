---
license: mit
tags:
- nanoGPT
- character-level
- baseline
- abcGPT
language:
- en
---

# abcGPT nano-1: per-source character-level baselines

This repo holds 100 trained nanoGPT character-level models, one per source from the [100_simple_voices](https://github.com/iamtrask/abcGPT/tree/main/data/100_simple_voices) corpus, plus a Karpathy-parity tinyshakespeare baseline.

Each model is **10.7M parameters** (Karpathy's `shakespeare_char` spec: 6L/6H/384), trained for 5000 iters with AdamW, cosine LR 1e-3 → 1e-4, on a single Modal T4 GPU. All baselines use Karpathy's `config/train_shakespeare_char.py` byte-for-byte — only `--dataset` and `--out_dir` were overridden per source.

These are the **reference baselines** for the abcGPT scaling ladder. The point of the surrounding research is to show a single gated model with N cohorts can match each per-source baseline's val_loss when its slider is engaged to that cohort. Without these reference numbers, that claim is unfalsifiable.

## What's in the repo

```
iamtrask/abcGPT-nano-1-baselines/
├── README.md                              ← this file
├── summary.csv                             ← all baselines, sorted by best_val_loss
├── _karpathy-shakespeare-baseline/         ← Karpathy parity check
│   ├── ckpt.pt
│   ├── best_val_loss.txt
│   └── training.log
└── <source_name>/                          ← 100 per-source models
    ├── ckpt.pt
    ├── best_val_loss.txt
    ├── val_loss_curve.csv
    └── meta.json
```

## Usage

```python
from huggingface_hub import snapshot_download
import torch

# Download a single baseline
model_dir = snapshot_download(
    repo_id="iamtrask/abcGPT-nano-1-baselines",
    allow_patterns=["077_nba-play-by-play/*"],
)
ckpt = torch.load(f"{model_dir}/077_nba-play-by-play/ckpt.pt", map_location="cpu", weights_only=False)
# ckpt['model'] is the state_dict
# ckpt['model_args'] is the GPTConfig
# ckpt['best_val_loss'] is the val loss number
# ckpt['iter_num'] is the iter at which best_val_loss was achieved
```

## Summary of results (top 20 by val_loss)

| source_name | best_val_loss | vocab_size | train_tokens | val_tokens | wall_clock_s | n_eval_points |
|---|---|---|---|---|---|---|
| 076_retrosheet-baseball | 0.272 | 68 | 1003851 | 111539 | 134.0 | 21 |
| 077_nba-play-by-play | 0.34225887060165405 | 85 | 968297 | 107589 | 128.0 | 21 |
| 097_cia-world-factbook | 0.742789089679718 | 87 | 999603 | 111067 | 166.0 | 21 |
| 021_fannie-farmer | 0.7546903491020203 | 113 | 995423 | 110603 | 3572.4234993457794 | 21 |
| 038_ibsen | 0.807984471321106 | 95 | 998207 | 110912 | 159.0 | 21 |
| 052_uspto-patents | 0.8101 | 81 | 1002721 | 111414 | 141.0 | 21 |
| 005_book-of-mormon | 0.839693546295166 | 73 | 1003044 | 111450 | 3611.771785736084 | 21 |
| 040_whitman-leaves-grass | 0.8785607814788818 | 90 | 999415 | 111047 | 170.0 | 21 |
| 010_tinystories | 0.9005 | 77 | 1002108 | 111346 | 3500.190304040909 | 21 |
| 023_eliza-acton | 0.9193783402442932 | 105 | 1001793 | 111311 | 3659.260692834854 | 21 |
| 054_federal-register | 0.947 | 88 | 1000759 | 111196 | 127.0 | 21 |
| 051_us-code-title18 | 0.9501778483390808 | 82 | 993540 | 110394 | 128.0 | 21 |
| 058_us-constitution-amendments | 0.9554658532142639 | 89 | 1001797 | 111311 | 799.0 | 21 |
| 053_us-founding | 0.963 | 87 | 1002293 | 111366 | 137.0 | 21 |
| 059_cfr-title21 | 1.0079765319824219 | 91 | 998757 | 110973 | 792.0 | 21 |
| 090_arxiv-math-abs | 1.02278470993042 | 134 | 1003178 | 111465 | 148.0 | 21 |
| 000_kjv-bible | 1.0756 | 72 | 1002611 | 111402 | 4369.686533212662 | 21 |
| 091_arxiv-physics-abs | 1.11 | 131 | 1003519 | 111503 | 129.0 | 21 |
| 001_quran-rodwell | 1.1174991130828857 | 106 | 988560 | 109840 | 3520.9722871780396 | 21 |
| 061_doyle-sherlock | 1.1183 | 91 | 982558 | 109174 | 131.0 | 21 |

_(showing top 20 of 100 rows — see `summary.csv` for the full list)_

## Reproducing

See the [abcGPT repo](https://github.com/iamtrask/abcGPT) and specifically `experiments/nano-1/` for the training pipeline. The whole batch costs ~$15 of Modal T4 time (~2.5 hr wall-clock) and is fully reproducible:

```bash
cd experiments/nano-1
./run.sh preflight
./run.sh karpathy   # 12 min — validates pipeline reproduces Karpathy's published 1.4697
./run.sh all        # 2.5 hr — full 100-source fan-out + Karpathy
./run.sh fetch      # download ckpts + metrics to laptop
./run.sh publish    # upload to this HF repo + commit metrics to git
```

## Related

- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — the unmodified training code
- [abcGPT](https://github.com/iamtrask/abcGPT) — fork that adds per-neuron source-attribution gating
