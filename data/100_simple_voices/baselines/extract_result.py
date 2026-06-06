#!/usr/bin/env python3
"""Extract the val loss curve + best val loss from a trained baseline.

Reads:
  - <out_dir>/ckpt.pt        — final checkpoint (has best_val_loss + iter_num)
  - <out_dir>/training.log   — raw nanoGPT stdout captured by run_all.sh

Writes:
  - <result_dir>/best_val_loss.txt  — single float, the best val_loss seen
  - <result_dir>/val_loss_curve.csv — iter,train_loss,val_loss rows (one per eval)
  - <result_dir>/meta.json          — vocab_size, train/val tokens, wall-clock, etc.
"""
import argparse
import json
import os
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import torch


EVAL_LINE_RE = re.compile(
    r"step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)"
)


def parse_log_for_curve(log_path: Path):
    rows = []
    with open(log_path) as f:
        for line in f:
            m = EVAL_LINE_RE.search(line)
            if m:
                step = int(m.group(1))
                train = float(m.group(2))
                val = float(m.group(3))
                rows.append((step, train, val))
    return rows


def read_meta(data_dir: Path):
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    train_tokens = os.path.getsize(data_dir / "train.bin") // 2  # uint16 = 2 bytes
    val_tokens = os.path.getsize(data_dir / "val.bin") // 2
    return {
        "vocab_size": meta["vocab_size"],
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, help="nanoGPT training output dir")
    p.add_argument("--data-dir", required=True, help="prepped data dir (for meta.pkl)")
    p.add_argument("--result-dir", required=True, help="where to write per-dataset results")
    p.add_argument("--source-name", required=True, help="e.g. 077_nba-play-by-play")
    p.add_argument("--wall-clock-s", type=float, default=None, help="training wall-clock seconds")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. Curve from log
    log_path = out_dir / "training.log"
    if log_path.exists():
        curve = parse_log_for_curve(log_path)
    else:
        curve = []
        print(f"warning: no training.log at {log_path}")

    # 2. Best val from curve OR from checkpoint
    best_val = None
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_val = float(ckpt.get("best_val_loss", float("inf")))
    if curve:
        curve_best = min(row[2] for row in curve)
        if best_val is None or curve_best < best_val:
            best_val = curve_best

    # 3. Meta
    meta = read_meta(data_dir)
    meta.update({
        "source_name": args.source_name,
        "best_val_loss": best_val,
        "n_eval_points": len(curve),
        "wall_clock_s": args.wall_clock_s,
    })

    # 4. Write
    with open(result_dir / "val_loss_curve.csv", "w") as f:
        f.write("iter,train_loss,val_loss\n")
        for step, tr, va in curve:
            f.write(f"{step},{tr:.6f},{va:.6f}\n")

    with open(result_dir / "best_val_loss.txt", "w") as f:
        f.write(f"{best_val:.6f}\n" if best_val is not None else "nan\n")

    with open(result_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 5. Persist the trained model alongside the metrics so anything that uploads
    # `result_dir` (e.g., RunPod on_box.sh -> HF Hub) also captures the weights.
    # Without this, training compute is wasted: the model dies with the container.
    if ckpt_path.exists():
        shutil.copy(ckpt_path, result_dir / "ckpt.pt")
        print(f"{args.source_name}: copied ckpt.pt to {result_dir}")
    else:
        print(f"{args.source_name}: WARNING — no ckpt.pt at {ckpt_path}, model not saved")

    print(f"{args.source_name}: best_val_loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
