"""Offline "corner eval": compute the slider metrics diag / contrast / middle for a
trained FineWeb K-cluster checkpoint, OUTSIDE the training loop.

Metric definitions mirror train.py:eval_at (train.py:477-499) EXACTLY:
each metric averages, over `--eval-iters` val batches sampled with train.py's own
`make_batch_fn`, the loss of `model(X, alpha, Y)` (lora) or `model(X, None, Y)` (ungated).

With onehot(c) = α one-hot at cohort c, uniform = 1/K everywhere:

    diag_c        = eval_at(onehot(c), c)
    diag          = sum_c diag_c
    offdiag[c'][c] = eval_at(onehot(c'), c)            for c' != c
    contrast      = sum_c ( min_{c'!=c} offdiag[c'][c] - diag_c )   (higher = better separation)
    middle_c      = eval_at(uniform, c)
    middle        = mean_c middle_c

K=100 feasibility: the full 100x100 off-diagonal is 10k cells (too slow). For each
cohort c, the min off-diag is ESTIMATED against a RANDOM subset of `--sample-offdiag`
(default 12) other cohorts. diag (all K) and middle (all K) are computed in full.

UNGATED baseline: same weights for every cohort, so there is no notion of a wrong
corner. diag_c = model(X, None, Y) on cohort c's val; contrast = 0; middle_c = diag_c
(so middle = diag / K).

CLI:
    python eval_corners.py --data-dir DATA --ckpt CKPT --variant {ungated,lora} \
        --reserve-frac 0 --k 100 --eval-iters 80 --sample-offdiag 12 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# train.py / model.py live in the parent (experiments/nano-3); make them importable
# whether this is run from fineweb_cluster/ or from a flat /workspace pod layout.
_HERE = Path(__file__).resolve().parent
for cand in (_HERE, _HERE.parent):
    if (cand / "model.py").exists():
        sys.path.insert(0, str(cand))
        break

from model import NanoGPT, NanoGPTConfig, LoRAAdditiveLinear  # noqa: E402
# Reuse train.py's data loading + batch sampling EXACTLY so val draws match training.
from train import load_meta_and_bins, make_batch_fn  # noqa: E402


def build_model(variant, n_cohorts, reserve_frac, vocab_size, device):
    """Build the model with the verified-known-good config for each variant.

    Returns (model, cfg). The lora build uses offload_deltas=True so the per-cohort
    delta ParameterList keys match the checkpoint produced by the offload train run.
    """
    if variant == "ungated":
        cfg = NanoGPTConfig(
            variant="ungated",
            n_layer=12, n_head=12, n_embd=768,
            block_size=1024, vocab_size=vocab_size,
        )
    elif variant == "lora":
        cohort_names = [f"cluster_{i:02d}" for i in range(n_cohorts)]
        cfg = NanoGPTConfig(
            variant="lora",
            n_layer=12, n_head=12, n_embd=768,
            block_size=1024, vocab_size=vocab_size,
            n_cohorts=n_cohorts, cohort_names=cohort_names,
            rank=16, base_rank=-1,
            adaptive_capacity=True, bias_anchor=True, rslora=True,
            gate_attention=True, gate_embedding=True,
            offload_deltas=True, reserve_frac=reserve_frac,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    model = NanoGPT(cfg).to(device)

    if variant == "lora":
        # Mirror train.py:301-312 — hold the per-cohort delta ParameterLists on CPU
        # so the keys/shapes match the checkpoint, and only the active cohort streams
        # to GPU per forward (one-hot α touches exactly one cohort).
        n_off = 0
        for mod in model.modules():
            if isinstance(mod, LoRAAdditiveLinear) and getattr(mod, "offload", False):
                for p in list(mod.U) + list(mod.V):
                    p.data = p.data.cpu()
                    n_off += 1
        print(f"offload-deltas: {n_off} per-cohort delta params held on CPU", flush=True)

    return model, cfg


def load_ckpt(model, ckpt_path, device):
    """Load the checkpoint. Returns (iter, strict_used). Tries strict=True first;
    falls back to strict=False ONLY if the mismatch is buffer-only, and reports which."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    ckpt_iter = int(ck.get("iter", -1)) if isinstance(ck, dict) else -1
    try:
        model.load_state_dict(state, strict=True)
        print("load_state_dict: strict=True OK", flush=True)
        return ckpt_iter, True
    except RuntimeError as e:
        result = model.load_state_dict(state, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        print(f"load_state_dict: strict=True FAILED, fell back to strict=False", flush=True)
        print(f"  missing_keys ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}", flush=True)
        print(f"  unexpected_keys ({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}", flush=True)
        print(f"  (original error head: {str(e)[:300]})", flush=True)
        return ckpt_iter, False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="Dir with meta.pkl + per-cluster {name}_{train,val}.bin")
    p.add_argument("--ckpt", required=True, help="Path to ckpt.pt (or model.pt)")
    p.add_argument("--variant", required=True, choices=["ungated", "lora"])
    p.add_argument("--reserve-frac", type=float, default=0.0)
    p.add_argument("--k", type=int, default=100, help="Number of cohorts to evaluate")
    p.add_argument("--eval-iters", type=int, default=80,
                   help="Val batches averaged per (alpha, cohort) eval point")
    p.add_argument("--sample-offdiag", type=int, default=12,
                   help="Random other-cohort subset size used to estimate min off-diag per cohort")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.amp_dtype]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    # Use the first K cohorts from the data dir's meta (matches train.py's max_cohorts).
    meta, cohort_names, bins = load_meta_and_bins(data_dir, max_cohorts=args.k)
    n_cohorts = len(cohort_names)
    vocab_size = meta["vocab_size"]
    print(f"data: {n_cohorts} cohorts | vocab {vocab_size} | device {device}", flush=True)

    # Val batchers — one per cohort, EXACTLY train.py's machinery (bf[cohort]["val"]).
    bf = {c: {"val": make_batch_fn(bins[c]["val"], args.block_size, args.batch_size, device)}
          for c in cohort_names}

    model, cfg = build_model(args.variant, n_cohorts, args.reserve_frac, vocab_size, device)
    ckpt_iter, strict_used = load_ckpt(model, args.ckpt, device)
    model.eval()

    @torch.no_grad()
    def eval_at(alpha_np, cohort_name):
        """Average val loss over eval_iters batches at α=alpha_np on cohort_name.
        Mirrors train.py:eval_at: ungated -> model(X, None, Y); else model(X, alpha, Y)."""
        losses = []
        for _ in range(args.eval_iters):
            X, Y = bf[cohort_name]["val"]()
            with torch.amp.autocast(device_type=device, dtype=amp_dtype,
                                    enabled=(device == "cuda")):
                if args.variant == "ungated":
                    _, loss = model(X, None, Y)
                else:
                    alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)
                    _, loss = model(X, alpha, Y)
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    def onehot(c):
        a = np.zeros(n_cohorts, dtype=np.float32)
        a[c] = 1.0
        return a

    uniform = np.full(n_cohorts, 1.0 / n_cohorts, dtype=np.float32)

    t0 = time.time()

    # ---- diag: full K ----
    print("=== diag (K corners) ===", flush=True)
    per_cohort_diag = []
    for c in range(n_cohorts):
        if args.variant == "ungated":
            dc = eval_at(None, cohort_names[c])           # same weights for all
        else:
            dc = eval_at(onehot(c), cohort_names[c])
        per_cohort_diag.append(dc)
        print(f"  diag[{cohort_names[c]}] = {dc:.4f}  ({c+1}/{n_cohorts}, "
              f"{time.time()-t0:.0f}s)", flush=True)
    diag = float(sum(per_cohort_diag))

    # ---- contrast: estimate min off-diag from a random subset per cohort ----
    if args.variant == "ungated":
        # Same weights for every cohort -> no wrong-corner notion -> contrast = 0.
        contrast = 0.0
        print("=== contrast: ungated -> 0.0 ===", flush=True)
    else:
        print(f"=== contrast (min off-diag vs {args.sample_offdiag} random others) ===",
              flush=True)
        contrast = 0.0
        for c in range(n_cohorts):
            others = [cp for cp in range(n_cohorts) if cp != c]
            k_sub = min(args.sample_offdiag, len(others))
            subset = rng.choice(others, size=k_sub, replace=False).tolist()
            min_off = float("inf")
            for cp in subset:
                # offdiag[c'][c] = eval_at(onehot(c'), c): WRONG corner c' on cohort c's val.
                off = eval_at(onehot(cp), cohort_names[c])
                if off < min_off:
                    min_off = off
            contrib = min_off - per_cohort_diag[c]
            contrast += contrib
            print(f"  cohort {cohort_names[c]}: min_off={min_off:.4f} "
                  f"diag={per_cohort_diag[c]:.4f} contrib={contrib:.4f} "
                  f"({c+1}/{n_cohorts}, {time.time()-t0:.0f}s)", flush=True)
        contrast = float(contrast)

    # ---- middle: full K ----
    if args.variant == "ungated":
        # Uniform α is meaningless ungated; middle_c = diag_c -> middle = diag / K.
        middle = float(diag / n_cohorts)
        print(f"=== middle: ungated -> diag/K = {middle:.4f} ===", flush=True)
    else:
        print("=== middle (uniform α, K cohorts) ===", flush=True)
        middle_vals = []
        for c in range(n_cohorts):
            mc = eval_at(uniform, cohort_names[c])
            middle_vals.append(mc)
            print(f"  middle[{cohort_names[c]}] = {mc:.4f}  ({c+1}/{n_cohorts}, "
                  f"{time.time()-t0:.0f}s)", flush=True)
        middle = float(np.mean(middle_vals))

    result = {
        "diag": diag,
        "contrast": contrast,
        "middle": middle,
        "n_cohorts": n_cohorts,
        "eval_iters": args.eval_iters,
        "sample_offdiag": args.sample_offdiag,
        "iter": ckpt_iter,
        "variant": args.variant,
        "reserve_frac": args.reserve_frac,
        "strict_load": strict_used,
        "per_cohort_diag": per_cohort_diag,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nwrote {out_path}", flush=True)
    print(f"diag={diag:.4f} contrast={contrast:.4f} middle={middle:.4f}", flush=True)


if __name__ == "__main__":
    main()
