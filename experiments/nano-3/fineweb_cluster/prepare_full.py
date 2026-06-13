"""Tokenize FineWeb-10BT into a single combined gpt2-BPE bin for the UNIFORM, ungated
GPT-2 baseline. Streams + writes incrementally (10B tokens = ~20GB, can't hold in RAM).
First VAL_TOK tokens -> all_val.bin, the rest -> all_train.bin. One cohort ('all').

  python prepare_full.py --n-docs 0 --out-dir data_full
"""
import argparse, pickle
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

VAL_TOK = 5_000_000  # held-out val (~5M tokens, like nanoGPT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=0, help="0 = all (~14.9M)")
    ap.add_argument("--out-dir", default="data_full")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    ftrain = open(out / "all_train.bin", "wb"); fval = open(out / "all_val.bin", "wb")
    buf = []; total = 0; val_done = 0; n = 0
    import time; t0 = time.time()

    def flush():
        nonlocal buf
        if buf:
            np.array(buf, dtype=np.uint16).tofile(ftrain); buf = []

    for ex in ds:
        ids = enc.encode(ex.get("text", "") or "", disallowed_special=()); ids.append(eot)
        n += 1; total += len(ids); i = 0
        if val_done < VAL_TOK:                       # first VAL_TOK tokens -> val
            take = min(VAL_TOK - val_done, len(ids))
            np.array(ids[:take], dtype=np.uint16).tofile(fval); val_done += take; i = take
        if i < len(ids):
            buf.extend(ids[i:])
            if len(buf) >= 10_000_000:
                flush()
        if n % 200000 == 0:
            print(f"  {n:,} docs | {total/1e6:.0f}M tokens | {time.time()-t0:.0f}s", flush=True)
        if args.n_docs and n >= args.n_docs:
            break
    flush(); ftrain.close(); fval.close()
    meta = {"vocab_size": 50304, "dtype": "uint16", "tokenizer": "gpt2",
            "cohort_names": ["all"], "stoi": None, "itos": None}
    pickle.dump(meta, open(out / "meta.pkl", "wb"))
    print(f"wrote {total/1e9:.2f}B tokens ({n:,} docs, {val_done:,} val) -> {out}")


if __name__ == "__main__":
    main()
