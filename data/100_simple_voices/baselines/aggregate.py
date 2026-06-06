#!/usr/bin/env python3
"""Aggregate per-baseline results into a single summary CSV.

Auto-discovers all per-source result directories; no hardcoded source list.
Reads <results>/<source>/meta.json and emits one row per source in
<results>/summary.csv, sorted by best_val_loss (lowest = best fit, ascending).
"""
import csv
import json
import math
from pathlib import Path


# Results moved to experiments/nano-1/results/ for tier-centric organization.
# aggregate.py is per-tier-agnostic but historically lived here; resolve up to
# repo root and back down into the tier's results dir.
RESULT_DIR = Path(__file__).resolve().parents[3] / "experiments/nano-1/results"

FIELDS = [
    "source_name",
    "best_val_loss",
    "vocab_size",
    "train_tokens",
    "val_tokens",
    "wall_clock_s",
    "n_eval_points",
]


def main():
    rows = []
    src_dirs = sorted(p for p in RESULT_DIR.iterdir()
                      if p.is_dir() and (p / "meta.json").exists())

    if not src_dirs:
        print(f"no per-source results found under {RESULT_DIR}")
        print("(run `bash run_all.sh` first to produce them)")
        return

    for src_dir in src_dirs:
        with open(src_dir / "meta.json") as f:
            meta = json.load(f)
        rows.append({k: meta.get(k) for k in FIELDS})

    # Sort by best_val_loss ascending; missing values go to the end
    def sort_key(r):
        v = r["best_val_loss"]
        return (v if v is not None and not math.isnan(v) else float("inf"))
    rows.sort(key=sort_key)

    out = RESULT_DIR / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out}  ({len(rows)} baselines)")
    print()

    # Pretty table to stdout, sorted by best val_loss
    print(f"{'rank':>4}  {'source':<36}  {'val_loss':>9}  {'vocab':>6}  "
          f"{'train_tok':>10}  {'wall (s)':>9}")
    print("-" * 88)
    for i, r in enumerate(rows):
        wc = f"{r['wall_clock_s']:.0f}" if r["wall_clock_s"] is not None else "?"
        bv = f"{r['best_val_loss']:.4f}" if r["best_val_loss"] is not None else "?"
        name = r["source_name"][:36]
        print(f"{i+1:>4}  {name:<36}  {bv:>9}  {r['vocab_size']:>6}  "
              f"{r['train_tokens']:>10,}  {wc:>9}")


if __name__ == "__main__":
    main()
