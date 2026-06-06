#!/usr/bin/env python3
"""Publish nano-1 results to HuggingFace Hub.

Uploads the full `results/` directory (ckpt.pt + metrics + logs) to a HF model
repo so downstream users can `huggingface_hub.snapshot_download(...)` any of the
101 trained baselines (100 sources + Karpathy parity) and play with them.

Metrics (csv/txt/json) ALSO commit to git via `git add` — small enough to be
version-controlled directly. The HF repo is the model store; the git repo is
the source of truth for code + metrics.

Usage:
    python publish.py                                       # defaults
    python publish.py --repo iamtrask/abcGPT-nano-1         # custom repo
    python publish.py --results path/to/results             # custom results dir
    python publish.py --dry-run                             # show what would happen
    python publish.py --private                             # create as private repo
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO = "iamtrask/abcGPT-nano-1-baselines"
DEFAULT_RESULTS = "experiments/nano-1/results"


def check_auth():
    """Verify HF CLI is installed and authenticated."""
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        sys.exit("error: huggingface_hub not installed. Run: pip install huggingface_hub")
    try:
        user_info = whoami()
        return user_info["name"]
    except Exception as exc:
        sys.exit(
            f"error: not authenticated to HuggingFace Hub ({exc}).\n"
            "Run: huggingface-cli login"
        )


def verify_results(results_dir: Path) -> dict:
    """Verify the results directory has the expected structure. Returns stats."""
    if not results_dir.exists():
        sys.exit(f"error: results dir does not exist: {results_dir}")
    source_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not source_dirs:
        sys.exit(f"error: no source directories found in {results_dir}")

    ckpt_count = sum(1 for d in source_dirs if (d / "ckpt.pt").exists())
    metric_count = sum(1 for d in source_dirs if (d / "best_val_loss.txt").exists())
    total_size_gb = sum(f.stat().st_size for f in results_dir.rglob("*") if f.is_file()) / 1e9

    return {
        "n_source_dirs": len(source_dirs),
        "n_with_ckpt": ckpt_count,
        "n_with_metric": metric_count,
        "total_size_gb": total_size_gb,
        "source_dirs": source_dirs,
    }


def build_model_card(results_dir: Path, repo_id: str, stats: dict) -> str:
    """Generate the HF model card README from results data."""
    summary_path = results_dir / "summary.csv"
    summary_table = ""
    if summary_path.exists():
        with open(summary_path) as f:
            lines = f.read().strip().split("\n")
        if len(lines) > 1:
            header = lines[0].split(",")
            md_header = "| " + " | ".join(header) + " |"
            md_sep = "|" + "|".join(["---"] * len(header)) + "|"
            md_rows = ["| " + " | ".join(line.split(",")) + " |" for line in lines[1:21]]
            summary_table = "\n".join([md_header, md_sep] + md_rows)
            if len(lines) > 21:
                summary_table += f"\n\n_(showing top 20 of {len(lines)-1} rows — see `summary.csv` for the full list)_"

    return f"""---
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

This repo holds {stats['n_with_ckpt']} trained nanoGPT character-level models, one per source from the [100_simple_voices](https://github.com/iamtrask/abcGPT/tree/main/data/100_simple_voices) corpus, plus a Karpathy-parity tinyshakespeare baseline.

Each model is **10.7M parameters** (Karpathy's `shakespeare_char` spec: 6L/6H/384), trained for 5000 iters with AdamW, cosine LR 1e-3 → 1e-4, on a single Modal T4 GPU. All baselines use Karpathy's `config/train_shakespeare_char.py` byte-for-byte — only `--dataset` and `--out_dir` were overridden per source.

These are the **reference baselines** for the abcGPT scaling ladder. The point of the surrounding research is to show a single gated model with N cohorts can match each per-source baseline's val_loss when its slider is engaged to that cohort. Without these reference numbers, that claim is unfalsifiable.

## What's in the repo

```
{repo_id}/
├── README.md                              ← this file
├── summary.csv                             ← all baselines, sorted by best_val_loss
├── _karpathy-shakespeare-baseline/         ← Karpathy parity check
│   ├── ckpt.pt
│   ├── best_val_loss.txt
│   └── training.log
└── <source_name>/                          ← {stats['n_with_ckpt']} per-source models
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
    repo_id="{repo_id}",
    allow_patterns=["077_nba-play-by-play/*"],
)
ckpt = torch.load(f"{{model_dir}}/077_nba-play-by-play/ckpt.pt", map_location="cpu", weights_only=False)
# ckpt['model'] is the state_dict
# ckpt['model_args'] is the GPTConfig
# ckpt['best_val_loss'] is the val loss number
# ckpt['iter_num'] is the iter at which best_val_loss was achieved
```

## Summary of results (top 20 by val_loss)

{summary_table or "_(summary.csv not yet populated)_"}

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
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo ID (default: {DEFAULT_REPO})")
    p.add_argument("--results", default=DEFAULT_RESULTS, help=f"Results dir (default: {DEFAULT_RESULTS})")
    p.add_argument("--private", action="store_true", help="Create HF repo as private (default: public)")
    p.add_argument("--dry-run", action="store_true", help="Show what would happen without uploading")
    p.add_argument("--skip-git", action="store_true", help="Don't stage metric files for git commit")
    p.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip HF upload entirely; just stage metric files for git commit. Use this for "
             "incremental progress commits between full publishes.",
    )
    args = p.parse_args()

    results_dir = Path(args.results).resolve()

    # HF auth only needed for the full publish path
    if not args.metrics_only:
        user = check_auth()
        print(f"Authenticated as: {user}")
    else:
        print("(metrics-only mode: skipping HF auth check)")

    stats = verify_results(results_dir)
    print(f"\nFound {stats['n_source_dirs']} source dirs at {results_dir}")
    print(f"  with ckpt.pt: {stats['n_with_ckpt']}")
    print(f"  with best_val_loss.txt: {stats['n_with_metric']}")
    print(f"  total size: {stats['total_size_gb']:.2f} GB")

    if not args.metrics_only and stats['n_with_ckpt'] == 0:
        sys.exit("error: no ckpt.pt files found — nothing to publish")
    if args.metrics_only and stats['n_with_metric'] == 0:
        sys.exit("error: no metric files found — nothing to commit")

    # Build the model card
    print(f"\nGenerating model card README.md ...")
    readme_path = results_dir / "README.md"
    readme_content = build_model_card(results_dir, args.repo, stats)
    if not args.dry_run:
        readme_path.write_text(readme_content)
        print(f"  wrote {readme_path}")

    if args.metrics_only:
        print(f"\n(metrics-only: skipping HF upload of {stats['total_size_gb']:.2f} GB)")
    else:
        visibility = "private" if args.private else "public"
        print(f"\nWill upload {stats['total_size_gb']:.2f} GB to {visibility} HF repo: {args.repo}")
        if args.dry_run:
            print("--dry-run set; not uploading.")
            return

        from huggingface_hub import HfApi, create_repo
        try:
            create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
            print(f"  repo ready: https://huggingface.co/{args.repo}")
        except Exception as exc:
            sys.exit(f"error: create_repo failed: {exc}")

        api = HfApi()
        print(f"\nUploading {results_dir} → {args.repo} (~{stats['total_size_gb']:.1f} GB) ...")
        api.upload_folder(
            folder_path=str(results_dir),
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Upload nano-1 baselines ({stats['n_with_ckpt']} models)",
        )
        print(f"  ✅ done. View at: https://huggingface.co/{args.repo}")

    # Stage metric files for git commit (NOT ckpt.pt — gitignored anyway).
    # Walk up from results_dir to the repo root by checking for .git/.
    if not args.skip_git:
        print(f"\nStaging metric files for git commit ...")
        repo_root = results_dir
        for _ in range(6):  # don't walk forever
            if (repo_root / ".git").exists():
                break
            repo_root = repo_root.parent
        if not (repo_root / ".git").exists():
            print(f"  ⚠ couldn't find .git/ walking up from {results_dir}; skipping git add")
            return
        try:
            subprocess.run(
                ["git", "add", str(results_dir)],
                check=True,
                cwd=repo_root,
            )
            print(f"  ✅ git add succeeded at repo root: {repo_root}")
            print(f"     (ckpt.pt files automatically excluded by .gitignore)")
            print(f"\nNext step:")
            print(f"  cd {repo_root}")
            print(f"  git status   # review what will be committed")
            print(f"  git commit -m 'nano-1 baselines: results + model card'")
            print(f"  git push")
        except subprocess.CalledProcessError as exc:
            print(f"  ⚠ git add failed: {exc} (skipping — commit manually)")


if __name__ == "__main__":
    main()
