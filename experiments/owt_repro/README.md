# OWT reproduction — determinism checks

Stepping ladder for building a source-attributed OpenWebText dataset that is
byte-identical to Karpathy's nanoGPT canonical `train.bin` / `val.bin`. We
verify determinism at each scale before paying for the next.

## Step ladder

| step | what | where | cost | time |
|---|---|---|---|---|
| Check 0 | tokenizer + RNG determinism on 100 docs | laptop | $0 | ~30s |
| Check 1 | full prep twice on GCP, hash compare | GCP c3-standard-44 preemptible | ~$10 | ~12h |
| Step 2 | URL→cohort recovery from Gokaslan raw | GCP | ~$5 | ~3h |
| Step 3 | augmented prep, verify byte-identical | GCP | ~$10 | ~12h |
| Step 4 | small-scale gated training (12M params) | GCP single A100 | ~$10 | ~4h |
| Step 5 | full GatedGPT-124M reproduction | GCP 8×A100 | ~$1200 | 4 days |

## Run Check 0 now

```bash
cd /Users/atrask/Desktop/abcGPT/experiments/owt_repro
uv run --with 'tiktoken==0.7.0' \
       --with 'datasets==2.21.0' \
       --with 'numpy>=1.26,<2' \
       python check_0.py
```

## Pass criteria

- Tokenizing 100 docs twice produces identical SHA256
- `train_test_split(seed=2357, shuffle=True)` produces identical train and val splits across two calls

If either fails: stop and find the source of nondeterminism (most likely a
`tiktoken` or `datasets` version we haven't pinned). Do not advance to
Check 1 until Check 0 passes.

## If it passes

Note the versions printed at the bottom of the run. Those are the pins to
use for Check 1 on GCP. The dataset revision should also be pinned at this
point — get the current commit hash from `huggingface-cli` and set
`REVISION` in both `check_0.py` and any subsequent prep scripts.
