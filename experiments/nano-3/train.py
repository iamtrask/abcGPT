"""nano-3 N-cohort GatedGPT training: ungated / per_weight / hypernet.

Mirrors the structure of nano-2/train.py but generalizes from 2 cohorts to N.
Reads per-cohort {cohort}_{train,val}.bin from --data-dir + meta.pkl that lists
cohort_names. Trains with α ∈ simplex^N (one-hot curriculum → Dirichlet(1,...,1)).
Stratified-init + anchor-reg recipe from tiny-3-source / HYPERNET_VALIDATED.md.

JSONL log format (same record types as nano-2):
    'config' — initial config snapshot
    'iter'   — per log_interval (alpha, cohort, train_loss, lr)
    'eval'   — per eval_interval (N×N corner perplexity table)
    'corner_table' — final N×N corner table
    'edge_curve'   — final edge curve along each one-hot → uniform line
    'done'   — final summary

Usage:
    # smoke (CPU, ~30s)
    python experiments/nano-3/train.py --variant hypernet --variant-name smoke --smoke

    # full (cloud GPU, ~25 min)
    python experiments/nano-3/train.py --variant hypernet --variant-name hyp-v1 \\
        --n-iters 10000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (NanoGPT, NanoGPTConfig,
                     PerWeightGatedLinear, HypernetGatedLinear,
                     PerWeightFFN, HypernetFFN,
                     LoRAAdditiveLinear, LoRAAdditiveEmbedding,
                     HybridGatedLinear,
                     FactorizedHypernet)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_meta_and_bins(data_dir):
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    cohort_names = meta["cohort_names"]
    dtype = np.dtype(meta["dtype"].replace("<class '", "").replace("'>", "")
                       .replace("numpy.", "") if "<class" in str(meta["dtype"])
                       else meta["dtype"])
    bins = {}
    for c in cohort_names:
        train = np.memmap(data_dir / f"{c}_train.bin", dtype=dtype, mode="r")
        val = np.memmap(data_dir / f"{c}_val.bin", dtype=dtype, mode="r")
        bins[c] = {"train": train, "val": val}
    return meta, cohort_names, bins


def make_batch_fn(arr, block_size, batch_size, device):
    def get():
        ix = torch.randint(len(arr) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(arr[i:i + block_size].astype(np.int64)) for i in ix.tolist()])
        y = torch.stack([torch.from_numpy(arr[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix.tolist()])
        return x.to(device), y.to(device)
    return get


# ---------------------------------------------------------------------------
# Stratified init
# ---------------------------------------------------------------------------
def stratified_pattern_target(out_features, in_features, n_cohorts, probs, rng,
                                  singletons_only=False):
    """Sample a (n_cohorts, out, in) target where each weight is randomly assigned
    a non-empty bit-pattern.

    For small N (≤ 16): enumerate all 2^N - 1 non-empty patterns and sample from
    `probs` (a distribution over those patterns).

    For large N (or `singletons_only=True`): short-circuit to singletons only —
    each weight is assigned to EXACTLY ONE cohort sampled uniformly. Avoids the
    2^N pattern enumeration that's infeasible past N≈20.
    """
    n_weights = out_features * in_features
    if singletons_only or n_cohorts > 16:
        # Singletons-only path: sample one cohort per weight, build sparse pattern.
        sampled = rng.integers(0, n_cohorts, size=n_weights)        # (n_weights,)
        scale = np.zeros((n_cohorts, n_weights), dtype=np.float32)
        scale[sampled, np.arange(n_weights)] = 1.0
        scale = scale.reshape(n_cohorts, out_features, in_features)
        return torch.from_numpy(scale)
    n_patterns = 2 ** n_cohorts - 1
    patterns = np.array(
        [[int(b) for b in bin(i)[2:].zfill(n_cohorts)] for i in range(1, n_patterns + 1)],
        dtype=np.float32,
    )
    sampled = rng.choice(n_patterns, size=n_weights, p=probs)
    pat = patterns[sampled]                                       # (n_weights, n_cohorts)
    scale = pat.T.reshape(n_cohorts, out_features, in_features)   # (N, out, in)
    return torch.from_numpy(scale)


def stratified_init_per_weight(layer: PerWeightGatedLinear, probs, rng,
                                  singletons_only=False):
    target = stratified_pattern_target(layer.out_features, layer.in_features,
                                          layer.n_cohorts, probs, rng,
                                          singletons_only=singletons_only)
    layer.scales.data.copy_(target.to(layer.scales.device))


def warmstart_hypernet_layer(model, layer: HypernetGatedLinear, target,
                                n_iters=1000, lr=0.02):
    """Fit the hypernet so hypernet(cohort_embeddings[c]) ≈ target[c] per cohort.

    Updates both the layer's hypernet parameters AND the model's shared
    cohort_embeddings table. Returns final MSE.
    """
    params = list(layer.hypernet.parameters()) + [model.cohort_embeddings]
    opt = torch.optim.Adam(params, lr=lr)
    target = target.to(next(layer.parameters()).device)
    for _ in range(n_iters):
        cur = layer.hypernet(model.cohort_embeddings)
        loss = F.mse_loss(cur, target)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# Phase 2.1: literature-driven aux losses + HAT temperature scheduling
# ---------------------------------------------------------------------------
def _iter_lora_cohort_modules(model):
    """Yield every module in `model` that has per-cohort LoRA factors (U, V)
    matching shapes (n_cohorts, *, rank). Covers LoRAAdditiveLinear,
    LoRAAdditiveEmbedding, and HybridGatedLinear."""
    for mod in model.modules():
        if isinstance(mod, (LoRAAdditiveLinear, LoRAAdditiveEmbedding,
                              HybridGatedLinear)):
            yield mod


def cohort_contrast_loss(model):
    """Concept-Sliders cohort-contrast: maximize pairwise Frobenius distance
    between each cohort's delta matrix ΔW_c = U_c @ V_c^T.

    For C cohorts there are C*(C-1)/2 unordered pairs. We average across pairs
    AND across layers so the coefficient λ has consistent meaning regardless of
    model depth or N_cohorts.

    Returns a scalar tensor. NEGATIVE so gradient descent INCREASES inter-cohort
    distance — i.e. the loss is `-mean_pairs ||ΔW_c - ΔW_d||_F^2`.
    """
    total = 0.0
    n_layers = 0
    for mod in _iter_lora_cohort_modules(model):
        U = mod.U  # (C, out, r)
        V = mod.V  # (C, in, r)
        C = U.shape[0]
        if C < 2:
            continue
        # Compute ΔW_c = U_c @ V_c^T for all cohorts at once: (C, out, in).
        # For large embedding layers (vocab × n_embd) this is the heavy step;
        # acceptable because the einsum below at training step is already O(C * out * in * r).
        delta = torch.einsum('cor,cir->coi', U, V)
        # Pairwise squared Frobenius distance, averaged over pairs.
        # ||A - B||_F^2 expansion: ||A||^2 + ||B||^2 - 2 <A, B>.
        sq_norms = (delta ** 2).sum(dim=(1, 2))  # (C,)
        inner = torch.einsum('coi,doi->cd', delta, delta)  # (C, C)
        # pairwise: ||c-d||^2 = sq[c] + sq[d] - 2 inner[c,d]
        pair_sum = 0.0
        n_pairs = 0
        for c in range(C):
            for d in range(c + 1, C):
                pair_sum = pair_sum + sq_norms[c] + sq_norms[d] - 2.0 * inner[c, d]
                n_pairs += 1
        if n_pairs == 0:
            continue
        total = total + (pair_sum / n_pairs)
        n_layers += 1
    if n_layers == 0:
        return torch.zeros((), device=next(model.parameters()).device)
    # MINUS sign: descending the loss MAXIMIZES distance.
    return -(total / n_layers)


def load_balance_loss(model):
    """MoLE-style load-balance: penalize variance (across cohorts) of the
    per-cohort delta Frobenius norms. If one cohort's delta is huge and another
    is small, variance is high → loss is high → push toward equal-magnitude
    cohorts.

    Per layer: var_c(||ΔW_c||_F). Mean across layers.

    Returns a non-negative scalar tensor.
    """
    total = 0.0
    n_layers = 0
    for mod in _iter_lora_cohort_modules(model):
        U = mod.U
        V = mod.V
        C = U.shape[0]
        if C < 2:
            continue
        # ||U_c V_c^T||_F^2 = trace(V_c U_c^T U_c V_c^T) = ||U_c^T U_c||_F-style;
        # simpler: form ΔW_c then take Frobenius norm.
        delta = torch.einsum('cor,cir->coi', U, V)
        fro = torch.sqrt((delta ** 2).sum(dim=(1, 2)) + 1e-12)  # (C,)
        total = total + fro.var(unbiased=False)
        n_layers += 1
    if n_layers == 0:
        return torch.zeros((), device=next(model.parameters()).device)
    return total / n_layers


def set_hat_temperature(model, temp):
    """Set FactorizedHypernet.temperature on every hypernet in the model.
    Used to anneal the gate sharpness over training (HAT-style)."""
    for mod in model.modules():
        if isinstance(mod, FactorizedHypernet):
            mod.temperature = float(temp)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_run(args):
    device = args.device
    data_dir = Path(args.data_dir)
    out_dir = Path(args.results_root) / args.variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    meta, cohort_names, bins = load_meta_and_bins(data_dir)
    n_cohorts = len(cohort_names)
    vocab_size = meta["vocab_size"]

    # Build batchers — one per (cohort, split). Indexed as bf[cohort][split].
    bf = {c: {"train": make_batch_fn(bins[c]["train"], args.block_size, args.batch_size, device),
              "val":   make_batch_fn(bins[c]["val"],   args.block_size, args.batch_size, device)}
          for c in cohort_names}

    cfg = NanoGPTConfig(
        vocab_size=vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, bias=args.bias,
        n_cohorts=n_cohorts, variant=args.variant,
        d_embed=args.d_embed, rank=args.rank,
        cohort_names=cohort_names,
        gate_attention=args.gate_attention,
        gate_embedding=args.gate_embedding,
        base_rank=args.base_rank,
        adaptive_capacity=args.adaptive_capacity,
        hybrid_lora_rank=args.hybrid_lora_rank,
        rslora=args.rslora,
        bias_anchor=args.bias_anchor,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = NanoGPT(cfg).to(device)
    n_params = model.num_params()
    print(f"variant: {args.variant}  |  name: {args.variant_name}  |  log: {log_path}")
    print(f"cohorts: {cohort_names}  |  vocab: {vocab_size}  |  device: {device}")
    print(f"arch: n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} "
          f"block={args.block_size}  |  {n_params/1e6:.2f}M params")

    # Phase 1.8: load previously-discovered capacity allocations from a Stage-1 ckpt
    if args.load_caps_from:
        assert args.adaptive_capacity, "--load-caps-from requires --adaptive-capacity"
        load_path = args.load_caps_from
        if load_path.startswith("hf:"):
            # Format "hf:<repo>:<variant>" e.g. "hf:iamtrask/abcGPT-nano-3:n3-lora-baseR8-r64-full-adaptive"
            _, repo, variant = load_path.split(":", 2)
            from huggingface_hub import hf_hub_download
            print(f"fetching {variant}/model.pt from HF repo {repo}")
            load_path = hf_hub_download(repo_id=repo, filename=f"{variant}/model.pt",
                                          repo_type="model")
        print(f"loading caps from: {load_path}")
        prev_state = torch.load(load_path, map_location=device, weights_only=True)
        n_loaded = 0
        for name, p in model.named_parameters():
            if name.endswith("cohort_log_caps") or name.endswith("base_log_cap"):
                if name in prev_state:
                    p.data.copy_(prev_state[name])
                    n_loaded += 1
        print(f"  loaded {n_loaded} cap parameters from prior run")
        if args.freeze_caps:
            for name, p in model.named_parameters():
                if name.endswith("cohort_log_caps") or name.endswith("base_log_cap"):
                    p.requires_grad = False
            print(f"  caps frozen (requires_grad=False)")

    # ---- Stratified init for gated variants ----
    # LoRA-additive skips this entirely: LoRA-standard init (U random, V=0) at
    # construction time already gives ΔW=0 → model behaves as ungated baseline
    # → deltas grow during training. No warmstart needed, no anchor reg
    # required by default.
    init_scales = None
    if args.variant in ("per_weight", "hypernet", "hybrid"):
        # For N > 16 cohorts (and the singletons init), short-circuit to
        # singleton-only sampling (avoids the 2^N pattern enumeration).
        # For N <= 16 with non-singletons init, enumerate normally.
        singletons_only = (n_cohorts > 16) or (args.init == "singletons" and n_cohorts > 16)
        if args.init == "singletons":
            # singletons init at low N is just the direct singleton path too
            singletons_only = True
        if singletons_only:
            probs = None
            print(f"stratified init: singletons-only (N={n_cohorts}; each weight → exactly one cohort)")
        else:
            n_patterns = 2 ** n_cohorts - 1
            patterns = [tuple(int(b) for b in bin(i)[2:].zfill(n_cohorts))
                        for i in range(1, n_patterns + 1)]
            if args.init == "uniform":
                probs = np.ones(n_patterns) / n_patterns
            elif args.init == "low_hamming":
                w = np.array([1.0 / sum(p) for p in patterns], dtype=float)
                probs = w / w.sum()
            else:
                raise ValueError(f"unknown --init: {args.init}")
            print(f"stratified init probs ({n_patterns} patterns): {probs.round(3).tolist()}")
        if args.variant == "per_weight":
            for layer in model.gated_layers():
                stratified_init_per_weight(layer, probs, rng, singletons_only=singletons_only)
            init_scales = [layer.scales.detach().clone() for layer in model.gated_layers()]
        else:
            # hypernet OR hybrid: warm-start the hypernet part of each layer
            # to fit a stratified-pattern target (for hybrid, this initializes
            # the multiplicative gate; the LoRA additive deltas start at zero)
            for li, layer in enumerate(model.gated_layers()):
                target = stratified_pattern_target(layer.out_features, layer.in_features,
                                                       n_cohorts, probs, rng,
                                                       singletons_only=singletons_only)
                fl = warmstart_hypernet_layer(model, layer, target, n_iters=args.warmstart_iters)
                print(f"  warmstart layer {li}: final MSE = {fl:.4f}")
            with torch.no_grad():
                init_scales = [layer.hypernet(model.cohort_embeddings).detach().clone()
                                for layer in model.gated_layers()]

    # ---- Optimizer (mask-LR split for hypernet's cohort_embeddings if desired) ----
    # Plain AdamW for now; nano-2's param-group split (mask logits at lower LR) is
    # not strictly needed for the hypernet here since the warmstart already
    # commits the embedding into a reasonable basin. Can be added later if drift
    # is a problem.
    #
    # Phase 2.1: SMEAR-style cap-LR multiplier. If --cap-lr-multiplier > 1, the
    # cohort_log_caps / base_log_cap params go into a SEPARATE param group with
    # an LR scaled by that multiplier, no weight decay. Otherwise the standard
    # decay/nodecay split applies.
    cap_lr_mult = float(args.cap_lr_multiplier)
    use_cap_group = (cap_lr_mult != 1.0)
    decay, nodecay, caps = [], [], []
    n_cap = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_cap = use_cap_group and (n.endswith("cohort_log_caps") or n.endswith("base_log_cap"))
        if is_cap:
            caps.append(p)
            n_cap += 1
        elif p.dim() >= 2:
            decay.append(p)
        else:
            nodecay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": args.weight_decay, "lr_mult": 1.0},
        {"params": nodecay, "weight_decay": 0.0, "lr_mult": 1.0},
    ]
    if use_cap_group:
        # Cap params: scaled LR, no weight decay (they're scalar log-magnitudes,
        # not weight tensors). SMEAR-style separate router LR.
        param_groups.append({"params": caps, "weight_decay": 0.0,
                              "lr_mult": cap_lr_mult})
        print(f"cap-lr-multiplier={cap_lr_mult}: {n_cap} cap params in their own group "
              f"(lr × {cap_lr_mult}, weight_decay=0)")
    opt = torch.optim.AdamW(
        param_groups,
        lr=args.lr, betas=(0.9, args.beta2),
        fused=(device == "cuda"),
    )

    def get_lr(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        if it > args.n_iters:
            return args.min_lr
        decay_ratio = (it - args.warmup) / max(1, args.n_iters - args.warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    def apply_lr(it):
        lr = get_lr(it)
        for pg in opt.param_groups:
            # Phase 2.1: each group can carry an lr_mult (1.0 by default; cap
            # params see cap_lr_mult * base lr).
            pg["lr"] = lr * pg.get("lr_mult", 1.0)
        return lr

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                  "float32": torch.float32}[args.amp_dtype]

    @torch.no_grad()
    def eval_at(alpha_np, cohort_name):
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        try:
            model.eval()
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
            model.train()
            return float(np.mean(losses))
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)

    # ---- Open log + write config ----
    log_fp = open(log_path, "w")

    def emit(rec):
        log_fp.write(json.dumps(rec) + "\n"); log_fp.flush()

    emit({
        "type": "config",
        "variant": args.variant,
        "variant_name": args.variant_name,
        "cohort_names": cohort_names,
        "n_cohorts": n_cohorts,
        "n_iters": args.n_iters,
        "lr": args.lr, "warmup": args.warmup, "min_lr": args.min_lr,
        "batch_size": args.batch_size, "block_size": args.block_size,
        "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
        "d_embed": args.d_embed, "rank": args.rank,
        "init": args.init, "lambda_anchor": args.lambda_anchor,
        "alpha_curriculum_until": args.alpha_curriculum_until,
        "warmstart_iters": args.warmstart_iters,
        "seed": args.seed,
        "n_params": n_params,
        "t_start": time.time(),
    })

    # ---- Training loop ----
    t0 = time.time()
    t_log = t0
    n_per_cohort = {c: 0 for c in cohort_names}
    model.train()

    single_cohort_idx = None
    if args.single_cohort != "none":
        if args.single_cohort not in cohort_names:
            sys.exit(f"--single-cohort {args.single_cohort} not in {cohort_names}")
        single_cohort_idx = cohort_names.index(args.single_cohort)

    # ---- Phase 2.1: HAT temperature schedule (hypernet/hybrid only) ----
    # Linear interpolation between hat_temp_start and hat_temp_end across all
    # training iters. Default 1.0/1.0 is a no-op (skipped entirely below).
    hat_anneal_active = (args.hat_temp_start != args.hat_temp_end)
    if hat_anneal_active and args.variant not in ("hypernet", "hybrid"):
        print(f"WARNING: --hat-temp-start/end set but variant is {args.variant!r} "
              f"(no FactorizedHypernet to anneal); flags will be ignored.")
        hat_anneal_active = False
    if hat_anneal_active:
        print(f"HAT temperature schedule: {args.hat_temp_start} → {args.hat_temp_end} "
              f"linearly over {args.n_iters} iters")

    for it in range(args.n_iters):
        cur_lr = apply_lr(it)

        # Phase 2.1: HAT temperature anneal (hypernet/hybrid only)
        if hat_anneal_active:
            t_frac = it / max(1, args.n_iters - 1)
            temp = args.hat_temp_start + (args.hat_temp_end - args.hat_temp_start) * t_frac
            set_hat_temperature(model, temp)

        # α sampling
        if single_cohort_idx is not None:
            # Single-cohort ceiling baseline: only train on this cohort.
            # α one-hot at this cohort so any gated model still gets a defined α.
            c_idx = single_cohort_idx
            alpha_np = np.zeros(n_cohorts, dtype=np.float32); alpha_np[c_idx] = 1.0
        elif it < args.alpha_curriculum_until:
            c_idx = int(rng.integers(n_cohorts))
            alpha_np = np.zeros(n_cohorts, dtype=np.float32); alpha_np[c_idx] = 1.0
        elif args.mid_edge_prob > 0 and rng.random() < args.mid_edge_prob:
            # Phase 1.7: mid-edge sample — alpha = 0.5/0.5 mix of two random cohorts.
            # Directly trains midpoint composition behavior.
            i, j = rng.choice(n_cohorts, 2, replace=False)
            alpha_np = np.zeros(n_cohorts, dtype=np.float32)
            alpha_np[i] = 0.5; alpha_np[j] = 0.5
            # Pick which cohort's data to use for this step (either is reasonable)
            c_idx = int(rng.choice([i, j]))
        else:
            alpha_np = rng.dirichlet([1.0] * n_cohorts).astype(np.float32)
            c_idx = int(rng.choice(n_cohorts, p=alpha_np))
        cohort = cohort_names[c_idx]
        n_per_cohort[cohort] += 1

        X, Y = bf[cohort]["train"]()
        with torch.amp.autocast(device_type=device, dtype=amp_dtype,
                                  enabled=(device == "cuda")):
            if args.variant == "ungated":
                _, ce_loss = model(X, None, Y)
                loss = ce_loss
            else:
                alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)
                _, ce_loss = model(X, alpha, Y)
                loss = ce_loss
                if args.lambda_anchor > 0 and init_scales is not None:
                    L_anchor = 0.0
                    for li, layer in enumerate(model.gated_layers()):
                        cur = layer.scales if args.variant == "per_weight" \
                              else layer.hypernet(model.cohort_embeddings)
                        L_anchor = L_anchor + F.mse_loss(cur, init_scales[li])
                    loss = loss + args.lambda_anchor * L_anchor
                # Phase 2.1: cohort-contrast aux loss (Concept Sliders style).
                # MAXIMIZES inter-cohort delta distance (loss is negative-distance).
                if args.cohort_contrast_lambda > 0:
                    L_contrast = cohort_contrast_loss(model)
                    loss = loss + args.cohort_contrast_lambda * L_contrast
                # Phase 2.1: MoLE-style load-balance aux loss.
                # Penalizes variance in per-cohort delta magnitudes per layer.
                if args.load_balance_lambda > 0:
                    L_balance = load_balance_loss(model)
                    loss = loss + args.load_balance_lambda * L_balance

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if (it + 1) % args.log_interval == 0:
            dt = time.time() - t_log
            tl = float(ce_loss.item())
            emit({
                "type": "iter", "iter": it + 1,
                "alpha": alpha_np.tolist(), "cohort": cohort,
                "train_loss": tl, "lr": cur_lr,
                "dt_s": dt, "t_elapsed_s": time.time() - t0,
                "n_per_cohort": dict(n_per_cohort),
            })
            print(f"iter {it+1:5d}/{args.n_iters} | α={[f'{a:.2f}' for a in alpha_np]} "
                  f"| {cohort:>5s} | loss {tl:.4f} | lr {cur_lr:.5f} | dt {dt:.1f}s",
                  flush=True)
            t_log = time.time()

        if (it + 1) % args.eval_interval == 0 or (it + 1) == args.n_iters:
            corners = {}
            for c_alpha, name_alpha in enumerate(cohort_names):
                alpha_oh = np.zeros(n_cohorts, dtype=np.float32); alpha_oh[c_alpha] = 1.0
                for c_eval, name_eval in enumerate(cohort_names):
                    corners[f"{name_alpha}@val_{name_eval}"] = eval_at(alpha_oh, name_eval)
            emit({"type": "eval", "iter": it + 1,
                  "t_elapsed_s": time.time() - t0,
                  "corners": corners})
            # Compact print
            rows = []
            for c_eval, name_eval in enumerate(cohort_names):
                row_vals = [corners[f"{n}@val_{name_eval}"] for n in cohort_names]
                rows.append(f"val={name_eval:>5s}: " + " ".join(f"{v:6.3f}" for v in row_vals))
            print(f"  >>> step {it+1} corner-val-loss table:")
            for r in rows: print(f"      {r}", flush=True)

    # ---- Final eval: full N×N corner table + edge curves ----
    print("\n--- final N×N corner table ---")
    final_corners = {}
    for c_alpha, name_alpha in enumerate(cohort_names):
        alpha_oh = np.zeros(n_cohorts, dtype=np.float32); alpha_oh[c_alpha] = 1.0
        for c_eval, name_eval in enumerate(cohort_names):
            v = eval_at(alpha_oh, name_eval)
            final_corners[f"{name_alpha}@val_{name_eval}"] = v
    emit({"type": "corner_table", "iter": args.n_iters,
          "corners": final_corners})

    print(f"\n{'':>14}" + "".join(f"  α={c:>10s}" for c in cohort_names))
    for c_eval in cohort_names:
        row = f"  val={c_eval:>8s}    "
        for c_alpha in cohort_names:
            row += f"  {final_corners[f'{c_alpha}@val_{c_eval}']:>10.4f}"
        print(row)

    # Edge curves: for each pair (i, j), sweep α between them with the others at 0.
    # 5 points per edge (αi from 1 → 0 by 0.25). Useful to see whether the slider
    # transitions monotonically between corners.
    print("\n--- edge curves ---")
    if args.edge_curve_points >= 2:
        edges = []
        N = args.edge_curve_points
        for i in range(n_cohorts):
            for j in range(i + 1, n_cohorts):
                for t_idx in range(N):
                    t = t_idx / (N - 1)
                    alpha_np = np.zeros(n_cohorts, dtype=np.float32)
                    alpha_np[i] = 1.0 - t
                    alpha_np[j] = t
                    vals = {c: eval_at(alpha_np, c) for c in cohort_names}
                    edges.append({"edge": (cohort_names[i], cohort_names[j]),
                                  "t": t, "alpha": alpha_np.tolist(), "vals": vals})
                    print(f"  edge {cohort_names[i]}→{cohort_names[j]} t={t:.2f}: "
                          + " ".join(f"{c}={v:.3f}" for c, v in vals.items()), flush=True)
        emit({"type": "edge_curve", "iter": args.n_iters, "points": edges})

    emit({"type": "done", "iter": args.n_iters,
          "total_s": time.time() - t0,
          "n_per_cohort": dict(n_per_cohort)})
    print(f"\ntotal training time: {time.time()-t0:.1f}s")
    log_fp.close()

    # Save summary + model
    summary = {
        "variant": args.variant,
        "variant_name": args.variant_name,
        "n_params": n_params,
        "cohort_names": cohort_names,
        "final_corners": final_corners,
        "n_per_cohort": dict(n_per_cohort),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"DONE. results -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True,
                   choices=["ungated", "per_weight", "hypernet", "lora", "hybrid"])
    p.add_argument("--hybrid-lora-rank", type=int, default=64,
                   help="(hybrid variant only) Rank of the LoRA cohort deltas added on "
                        "top of the hypernet-gated base. Hypernet uses --rank as before.")
    p.add_argument("--variant-name", required=True)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data/shake_ts_code_char"))
    p.add_argument("--results-root", default=str(REPO_ROOT / "experiments/nano-3/results"))
    # Arch (nano-2 defaults)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--d-embed", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--base-rank", type=int, default=-1,
                   help="(lora variant only) If > 0, factorize each gated linear's base "
                        "weight as A@B^T at this rank instead of full rank. Forces more "
                        "capacity into per-cohort deltas. -1 (default) = full-rank base.")
    p.add_argument("--adaptive-capacity", action="store_true",
                   help="(lora variant only, Phase 1.7) Each gated linear gets learnable "
                        "per-cohort capacity scalars (softmax-normalized, fixed budget) and "
                        "a base scaling factor. Model auto-allocates capacity.")
    p.add_argument("--mid-edge-prob", type=float, default=0.0,
                   help="(Phase 1.7) Probability of sampling alpha as 0.5/0.5 mix of two "
                        "random cohorts during the Dirichlet phase. Directly trains midpoint "
                        "behavior. 0 = standard Dirichlet; 0.2 recommended for adaptive runs.")
    p.add_argument("--load-caps-from", default="",
                   help="(Phase 1.8) Path to a previous run's model.pt. Loads ONLY the "
                        "cohort_log_caps and base_log_cap params from each gated layer, "
                        "ignores everything else. Lets us start fresh training with "
                        "previously-discovered allocation as init. Requires --adaptive-capacity.")
    p.add_argument("--freeze-caps", action="store_true",
                   help="(Phase 1.8) Freeze cohort_log_caps and base_log_cap params after "
                        "loading (or random init). They don't get gradient updates.")
    # Phase 2.1: literature-driven interventions
    p.add_argument("--cohort-contrast-lambda", type=float, default=0.0,
                   help="(Phase 2.1, Concept Sliders) Coefficient on a cohort-discrimination "
                        "loss. For LoRA: penalizes similarity between cohorts' delta matrices "
                        "(U_c V_c.T). Pushes cohorts to learn distinct perturbations.")
    p.add_argument("--rslora", action="store_true",
                   help="(Phase 2.1, Kalajdzievski 2023) Use α/sqrt(r) scaling on LoRA deltas "
                        "instead of α/r. Helps stability at high cohort rank.")
    p.add_argument("--bias-anchor", action="store_true",
                   help="(Phase 3) Add per-layer per-cohort hard-α-gated additive bias "
                        "injection: residual += α @ B_layer where B_layer is (N, d_model). "
                        "Init to zero. Coexists with --variant. Gives an unambiguous α "
                        "signal at every layer that the model cannot soft-ignore. ")
    p.add_argument("--hat-temp-start", type=float, default=1.0,
                   help="(Phase 2.1, HAT for hypernet/hybrid only) Starting gate-temperature "
                        "at iter 0. Default 1.0 = no annealing. Set to 0.2 for soft gates "
                        "(easier early training, less cohort differentiation).")
    p.add_argument("--hat-temp-end", type=float, default=1.0,
                   help="(Phase 2.1, HAT) Final gate-temperature. Set to 3.0 for hard gates "
                        "at end (sharp cohort differentiation). Linearly ramps from start to "
                        "end over total training.")
    p.add_argument("--cap-lr-multiplier", type=float, default=1.0,
                   help="(Phase 2.1, SMEAR) Multiplier on the LR for cohort_log_caps and "
                        "base_log_cap params (requires --adaptive-capacity). Default 1.0; "
                        "5.0 mimics SMEAR's separate router LR.")
    p.add_argument("--load-balance-lambda", type=float, default=0.0,
                   help="(Phase 2.1, MoLE) Coefficient on a load-balancing aux loss that "
                        "penalizes variance in per-cohort delta magnitudes. Encourages "
                        "uniform cohort utilization.")
    # Training
    p.add_argument("--n-iters", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--alpha-curriculum-until", type=int, default=1000,
                   help="iters of one-hot α sampling before switching to Dirichlet(1,1,...)")
    p.add_argument("--init", default="uniform", choices=["uniform", "low_hamming", "singletons"])
    p.add_argument("--lambda-anchor", type=float, default=0.01)
    p.add_argument("--warmstart-iters", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=250)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--edge-curve-points", type=int, default=5)
    p.add_argument("--single-cohort", default="none",
                   help="If set to one of cohort_names (e.g. 'shake'), train ONLY on that cohort "
                        "for the full run. Used for per-cohort single-source ceiling baselines.")
    p.add_argument("--gate-attention", action="store_true",
                   help="Also gate the attention layer projections (c_attn for Q/K/V + c_proj).")
    p.add_argument("--gate-embedding", action="store_true",
                   help="Also gate the tied wte/lm_head embedding matrix.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--smoke", action="store_true", help="tiny config for <1 min CPU validation")
    args = p.parse_args()

    if args.smoke:
        args.n_layer = 2
        args.n_head = 2
        args.n_embd = 64
        args.block_size = 32
        args.batch_size = 16
        args.n_iters = 50
        args.warmup = 5
        args.log_interval = 10
        args.eval_interval = 25
        args.eval_iters = 5
        args.alpha_curriculum_until = 25
        args.warmstart_iters = 100
        args.edge_curve_points = 3
        if args.device == "cuda" and not torch.cuda.is_available():
            args.device = "cpu"

    if not Path(args.data_dir).exists():
        sys.exit(f"error: data dir missing: {args.data_dir}")
    train_run(args)


if __name__ == "__main__":
    main()
