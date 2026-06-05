#!/usr/bin/env python3
"""Char-level dataset prep, parameterized by source path.

Mirrors `data/shakespeare_char/prepare.py` from nanoGPT byte-for-byte
(same vocab derivation, same 90/10 train/val split, same uint16 layout,
same meta.pkl format) but takes the input file and output directory as
arguments instead of hardcoding tinyshakespeare.

This is what we run for each of the top-5 baseline datasets so that the
training side can use Karpathy's `config/train_shakespeare_char.py`
config without any modification.
"""

import argparse
import os
import pickle

import numpy as np


def prepare(source_path: str, out_dir: str, val_frac: float = 0.1) -> None:
    with open(source_path, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    chars = sorted(set(data))
    vocab_size = len(chars)
    print(f"vocab size: {vocab_size}")
    print(f"unique characters (repr): {chars!r}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    n = len(data)
    split_idx = int(n * (1.0 - val_frac))
    train_ids = np.array(encode(data[:split_idx]), dtype=np.uint16)
    val_ids = np.array(encode(data[split_idx:]), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val   has {len(val_ids):,} tokens")

    os.makedirs(out_dir, exist_ok=True)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"wrote {out_dir}/{{train,val}}.bin + meta.pkl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="path to source .txt file")
    p.add_argument("--out", required=True, help="output dir for {train,val}.bin + meta.pkl")
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()
    prepare(args.source, args.out, args.val_frac)
