#!/usr/bin/env python3
"""runpod_fanout.py — launch a nano-3 sweep across N RunPod pods.

Each variant in the sweep gets its own pod. The pod runs experiments/nano-3/train.py
with the variant's specific flags, then uploads results to HF Hub at
iamtrask/abcGPT-nano-3. Pods auto-terminate via the HF-completion monitor.

Modeled on experiments/nano-2/runpod/runpod_fanout.py. Differences:
  - on_box URL points to nano-3
  - HF repo: iamtrask/abcGPT-nano-3 (separate from nano-2)
  - SWEEP_DEFAULT is the focused first-sweep for the 3-cohort hypernet test

Usage:
    # 10-variant sweep (~30 min wall, ~$2-4 compute)
    RUNPOD_API_KEY=... HF_TOKEN=... python runpod_fanout.py

    # smoke: one variant only
    python runpod_fanout.py --only ungated-10k

    # dry-run (show plan, no launches)
    python runpod_fanout.py --dry-run
"""
import argparse
import os
import sys
import time

DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 3090"
DEFAULT_CLOUD_TYPE = "COMMUNITY"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_HF_REPO = "iamtrask/abcGPT-nano-3"


# Standard knobs for every gated variant (same n_iters / model arch as nano-2).
# Reads in nano-3/train.py as: n_layer=6, n_head=6, n_embd=384, block_size=256
# (the train.py defaults). Override per-variant when ablating.
COMMON = "--n-iters 10000 --seed 1337 --log-interval 250 --eval-interval 500 --eval-iters 200"
N5_DATA = "--data-dir data/shake_ts_code_sql_md_char"


# The first nano-3 sweep. 10 variants:
#   4 baselines (1 joint + 3 single-cohort ceilings) + 2 main mechanisms + 4 hypernet ablations.
# Per Andrew's table rule [[feedback-baseline-ceiling]]: always pin
# ungated-10k (no-gate ceiling) + a baseline that ignores α at the top.
SWEEP_DEFAULT = [
    # ------------------- BASELINES (always-pinned reference points) -------------------
    # Joint ungated: shared 6L-384d model trained on all 3 cohorts (α-curriculum cohort
    # sampling still active; α value ignored by the model). Each cohort's val_loss here
    # is the "shared-capacity" ceiling for any gated variant.
    ("ungated-10k",
     f"--variant ungated {COMMON}"),

    # Per-cohort single-source ceilings: ungated 6L-384d trained on ONE cohort only,
    # same iter budget. Floor for "what could you get if not sharing weights at all".
    ("ungated-shake-only",
     f"--variant ungated {COMMON} --single-cohort shake"),

    ("ungated-ts-only",
     f"--variant ungated {COMMON} --single-cohort ts"),

    ("ungated-code-only",
     f"--variant ungated {COMMON} --single-cohort code"),

    # ------------------- MAIN MECHANISMS -------------------
    # Hypernet: the scale-compatible mechanism. Canonical config from the tiny-3-source
    # winner: d_embed=8, rank=16, uniform stratified init, λ_anchor=0.01, 1k warmstart.
    # This is the variant most likely to beat la-bell-ramp9k.
    ("hypernet-uniform-d8r16",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init uniform "
     f"--lambda-anchor 0.01 --warmstart-iters 1000"),

    # Per-weight: the more expressive (but non-scaling-friendly) mechanism. Same init
    # / anchor as hypernet. Tiny-3-source data showed hypernet ≥ per_weight at N=3,
    # so we expect per_weight to be similar or slightly worse — confirms whether the
    # scale-compatible mechanism pays a real cost at N=3.
    ("per_weight-uniform",
     f"--variant per_weight {COMMON} --init uniform --lambda-anchor 0.01"),

    # ------------------- HYPERNET ABLATIONS -------------------
    # d_embed=3 — at the toy validation, d=3 (cohort count) also passed. Tests whether
    # the smallest sufficient embedding dim still works at real-text scale.
    ("hypernet-d3r16",
     f"--variant hypernet {COMMON} --d-embed 3 --rank 16 --init uniform "
     f"--lambda-anchor 0.01 --warmstart-iters 1000"),

    # d_embed=16, rank=32 — more hypernet capacity. Tests whether scaling up the
    # hypernet's representational capacity buys lower loss vs the canonical config.
    ("hypernet-d16r32",
     f"--variant hypernet {COMMON} --d-embed 16 --rank 32 --init uniform "
     f"--lambda-anchor 0.01 --warmstart-iters 1000"),

    # λ_anchor=0 — does the recipe still work without the anchor regularizer pulling
    # scales back toward their warm-started values? Probes whether the anchor is
    # load-bearing or just decorative for the GPT-scale arch.
    ("hypernet-noanchor",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init uniform "
     f"--lambda-anchor 0.0 --warmstart-iters 1000"),

    # Singleton-only init — each weight assigned to EXACTLY ONE cohort (no shared/
    # multi-cohort weights at init). Sharper specialization at warmstart, tests
    # whether the anchor + α-curriculum prevent collapse to non-singletons.
    ("hypernet-singletons",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000"),

    # ------------------- FULL-GATING ABLATION (sweep #2, 2026-06-09 PM) -------------------
    # Andrew's pushback: the first nano-3 sweep only gated the FFN. Cohorts differ
    # MOST in attention patterns (Python indent/colon structure vs Shake speaker tags
    # vs TS character-name repetition) AND in embedding-level token frequencies.
    # Gate EVERYTHING (c_attn + attn.c_proj + wte/lm_head) and see whether the
    # slider strength + matched-corner diag both improve.

    ("hypernet-singletons-full",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding"),

    ("per_weight-singletons-full",
     f"--variant per_weight {COMMON} --init singletons --lambda-anchor 0.01 "
     f"--gate-attention --gate-embedding"),

    # Uniform-init hypernet with full gating — does the weak slider get rescued
    # by widening which layers gate? If yes, init-density matters less than
    # gating coverage; if no, init-density is the dominant lever regardless.
    ("hypernet-uniform-full",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init uniform "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding"),

    # Anchor=0 control: does full-gating help the slider survive without anchor?
    # Previous noanchor (FFN only) collapsed to ungated. Maybe more gating = more
    # robust to no anchor; or maybe anchor stays load-bearing regardless.
    ("hypernet-singletons-full-noanchor",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.0 --warmstart-iters 1000 --gate-attention --gate-embedding"),

    # ============================================================================
    # SIZE SWEEP (sweep #3, 2026-06-10): scale ungated up to match the gated
    # variants' parameter count + scale gated further up to test slider scaling.
    # ============================================================================

    ("ungated-8L-512d",
     f"--variant ungated {COMMON} --n-layer 8 --n-head 8 --n-embd 512"),

    ("ungated-12L-768d",
     f"--variant ungated {COMMON} --n-layer 12 --n-head 12 --n-embd 768"),

    ("hypernet-singletons-full-8L-512d",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding "
     f"--n-layer 8 --n-head 8 --n-embd 512"),

    ("hypernet-singletons-full-12L-768d",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding "
     f"--n-layer 12 --n-head 12 --n-embd 768"),

    # ============================================================================
    # N=5 COHORT SWEEP (sweep #4, 2026-06-10): does the slider scale from N=3
    # to N=5? Adds SQL + markdown cohorts. Uses --data-dir to point at the new
    # 5-cohort prep.
    # ============================================================================

    ("n5-ungated-10k",
     f"--variant ungated {COMMON} {N5_DATA}"),

    ("n5-hypernet-singletons-full",
     f"--variant hypernet {COMMON} {N5_DATA} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding"),

    ("n5-per_weight-singletons-full",
     f"--variant per_weight {COMMON} {N5_DATA} --init singletons --lambda-anchor 0.01 "
     f"--gate-attention --gate-embedding"),

    # ============================================================================
    # N=100 COHORT SWEEP (sweep #5, 2026-06-10): just-for-kicks scaling test.
    # 100 distinct gutenberg-style classic novels, each as its own cohort.
    # Singletons-init (the only feasible init at N=100 — uniform would need
    # 2^100 patterns). on_box.sh fetches the pre-built bins tarball from HF
    # since the 107MB of source .txt files isn't committed to git.
    # ============================================================================

    ("n100-hypernet-singletons-full",
     f"--variant hypernet {COMMON} --data-dir data/100_sources_char "
     f"--d-embed 8 --rank 16 --init singletons --lambda-anchor 0.01 "
     f"--warmstart-iters 200 --gate-attention --gate-embedding"),

    # ============================================================================
    # N=5 DIAGNOSTIC SWEEP (sweep #6, 2026-06-10): N=5 default-recipe broke
    # (-0.21 contrast sum, 2 of 5 cohorts inverted). Six variants test 4
    # hypotheses for what's broken.
    # ============================================================================

    # Hypothesis A (Andrew's): too much non-slider capacity. Shrink model.
    ("n5-4L-256d-singletons-full",
     f"--variant hypernet {COMMON} {N5_DATA} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding "
     f"--n-layer 4 --n-head 4 --n-embd 256"),

    ("n5-4L-192d-singletons-full",
     f"--variant hypernet {COMMON} {N5_DATA} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding "
     f"--n-layer 4 --n-head 4 --n-embd 192"),

    ("n5-2L-192d-singletons-full",
     f"--variant hypernet {COMMON} {N5_DATA} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding "
     f"--n-layer 2 --n-head 4 --n-embd 192"),

    # Hypothesis B: hypernet expressive capacity too small for 5 cohort identities.
    ("n5-d16-r32-singletons-full",
     f"--variant hypernet {COMMON} {N5_DATA} --d-embed 16 --rank 32 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding"),

    # Hypothesis C: singletons init too sparse at N=5 (only 1/5 of weights per cohort).
    ("n5-low_hamming-full",
     f"--variant hypernet {COMMON} {N5_DATA} --d-embed 8 --rank 16 --init low_hamming "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --gate-attention --gate-embedding"),

    # Hypothesis D: anchor reg too weak; scales drift toward uniformity.
    ("n5-anchor0.5-singletons-full",
     f"--variant hypernet {COMMON} {N5_DATA} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.5 --warmstart-iters 1000 --gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 1: LoRA-ADDITIVE vs HYPERNET (sweep #7, 2026-06-10)
    # See LORA_RESEARCH_PLAN.md.
    # Direct comparison to hypernet-singletons-full (N=3) and
    # hypernet-low_hamming-full (N=5). LoRA-additive needs no warmstart and no
    # anchor regularizer by default — LoRA-standard init (V=0) gives ΔW=0 at
    # start; deltas learn from scratch.
    # ============================================================================

    # N=3, rank=16, full-gating (param-cheap rank-match to hypernet's r=16)
    ("n3-lora-r16-full",
     f"--variant lora {COMMON} --rank 16 --gate-attention --gate-embedding"),

    # N=3, rank=64, full-gating (closer to hypernet's parameter count)
    ("n3-lora-r64-full",
     f"--variant lora {COMMON} --rank 64 --gate-attention --gate-embedding"),

    # N=5, rank=16, full-gating (direct comparison to hypernet-low_hamming-full)
    ("n5-lora-r16-full",
     f"--variant lora {COMMON} {N5_DATA} --rank 16 --gate-attention --gate-embedding"),

    # N=5, rank=32, full-gating (more delta capacity)
    ("n5-lora-r32-full",
     f"--variant lora {COMMON} {N5_DATA} --rank 32 --gate-attention --gate-embedding"),

    # N=3, rank=16, FFN-only (tests whether LoRA needs full gating coverage)
    ("n3-lora-r16-ffn-only",
     f"--variant lora {COMMON} --rank 16"),

    # ============================================================================
    # PHASE 1.5: capacity asymmetry (sweep #8, 2026-06-10)
    # Phase 1 confirmed: LoRA at full base rank has weak contrast — the base
    # absorbs cohort-distinct behavior into shared weights. Test: shrink base
    # rank to force cohort deltas to do more work.
    # ============================================================================

    # Full-rank base (Phase 1 r=16 baseline — re-listed for direct comparison)
    # Already exists as n3-lora-r16-full; not relaunching.

    # Base rank 256 (modest reduction from 384 natural rank)
    ("n3-lora-baseR256-r16-full",
     f"--variant lora {COMMON} --rank 16 --base-rank 256 --gate-attention --gate-embedding"),

    # Base rank 128 (half capacity)
    ("n3-lora-baseR128-r16-full",
     f"--variant lora {COMMON} --rank 16 --base-rank 128 --gate-attention --gate-embedding"),

    # Base rank 64 (quarter capacity) — likely too small; tests breakdown
    ("n3-lora-baseR64-r16-full",
     f"--variant lora {COMMON} --rank 16 --base-rank 64 --gate-attention --gate-embedding"),

    # Base rank 128 with higher cohort rank (32) — give cohorts room to fill
    # the gap when base is starved
    ("n3-lora-baseR128-r32-full",
     f"--variant lora {COMMON} --rank 32 --base-rank 128 --gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 1.5b: aggressive base-rank reduction (sweep #9, 2026-06-10)
    # Phase 1.5 showed monotone trend (smaller base → bigger contrast) but
    # gradients gentle. Push to extreme small base values + compensating cohort
    # ranks to find inflection point.
    # ============================================================================

    ("n3-lora-baseR32-r16-full",
     f"--variant lora {COMMON} --rank 16 --base-rank 32 --gate-attention --gate-embedding"),

    ("n3-lora-baseR16-r16-full",
     f"--variant lora {COMMON} --rank 16 --base-rank 16 --gate-attention --gate-embedding"),

    ("n3-lora-baseR8-r16-full",
     f"--variant lora {COMMON} --rank 16 --base-rank 8 --gate-attention --gate-embedding"),

    ("n3-lora-baseR32-r64-full",
     f"--variant lora {COMMON} --rank 64 --base-rank 32 --gate-attention --gate-embedding"),

    ("n3-lora-baseR8-r64-full",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --gate-attention --gate-embedding"),

    ("n3-lora-baseR2-r128-full",
     f"--variant lora {COMMON} --rank 128 --base-rank 2 --gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 1.5d: push cohort rank up at small base (sweep #10, 2026-06-10)
    # Phase 1.5b showed contrast scales linearly with cohort rank (r=16 → +0.13,
    # r=64 → +0.25 at baseR=8). Push r higher to test whether LoRA can match
    # hypernet's +0.51 at acceptable diag.
    # ============================================================================

    ("n3-lora-baseR8-r128-full",
     f"--variant lora {COMMON} --rank 128 --base-rank 8 --gate-attention --gate-embedding"),

    ("n3-lora-baseR8-r256-full",
     f"--variant lora {COMMON} --rank 256 --base-rank 8 --gate-attention --gate-embedding"),

    ("n3-lora-baseR16-r128-full",
     f"--variant lora {COMMON} --rank 128 --base-rank 16 --gate-attention --gate-embedding"),

    ("n3-lora-baseR16-r256-full",
     f"--variant lora {COMMON} --rank 256 --base-rank 16 --gate-attention --gate-embedding"),

    ("n3-lora-baseR4-r256-full",
     f"--variant lora {COMMON} --rank 256 --base-rank 4 --gate-attention --gate-embedding"),

    ("n3-lora-baseR4-r512-full",
     f"--variant lora {COMMON} --rank 512 --base-rank 4 --gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 1.7: adaptive capacity allocation (sweep #11, 2026-06-10)
    # Andrew's "shift capacity from base to cohorts based on training signals"
    # idea. Each gated linear gets per-cohort capacity scalars (softmax-normalized
    # to fixed budget) + base scaling factor. Loss gradients reallocate.
    # ============================================================================

    # Adaptive at the best-known LoRA sweet spot
    ("n3-lora-baseR8-r64-full-adaptive",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--gate-attention --gate-embedding"),

    # Adaptive at the moderate-base sweet spot
    ("n3-lora-baseR16-r128-full-adaptive",
     f"--variant lora {COMMON} --rank 128 --base-rank 16 --adaptive-capacity "
     f"--gate-attention --gate-embedding"),

    # Adaptive at the ensemble-limit (does it recover diag?)
    ("n3-lora-baseR2-r128-full-adaptive",
     f"--variant lora {COMMON} --rank 128 --base-rank 2 --adaptive-capacity "
     f"--gate-attention --gate-embedding"),

    # Adaptive + explicit mid-edge sampling (Phase 1.7-full mechanism)
    ("n3-lora-baseR8-r64-full-adaptive-midedge",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--mid-edge-prob 0.2 --gate-attention --gate-embedding"),

    # Adaptive on N=5 (test if balance fixes the failed N=5 case)
    ("n5-lora-baseR8-r64-full-adaptive",
     f"--variant lora {COMMON} {N5_DATA} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 1.8: 2-stage LoRA — load discovered caps from Phase 1.7 (sweep #12)
    # Apples-to-apples vs hypernet's warmstart-then-train workflow.
    # Stage 1: Phase 1.7 adaptive (already done)
    # Stage 2 (this sweep): fresh training with discovered caps as frozen init
    # ============================================================================

    ("n3-lora-baseR8-r64-frozen-from-adaptive",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity --freeze-caps "
     f"--load-caps-from hf:iamtrask/abcGPT-nano-3:n3-lora-baseR8-r64-full-adaptive "
     f"--gate-attention --gate-embedding"),

    ("n3-lora-baseR16-r128-frozen-from-adaptive",
     f"--variant lora {COMMON} --rank 128 --base-rank 16 --adaptive-capacity --freeze-caps "
     f"--load-caps-from hf:iamtrask/abcGPT-nano-3:n3-lora-baseR16-r128-full-adaptive "
     f"--gate-attention --gate-embedding"),

    ("n3-lora-baseR8-r64-learnable-from-adaptive",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--load-caps-from hf:iamtrask/abcGPT-nano-3:n3-lora-baseR8-r64-full-adaptive "
     f"--gate-attention --gate-embedding"),

    ("n5-lora-baseR8-r64-frozen-from-adaptive",
     f"--variant lora {COMMON} {N5_DATA} --rank 64 --base-rank 8 --adaptive-capacity --freeze-caps "
     f"--load-caps-from hf:iamtrask/abcGPT-nano-3:n5-lora-baseR8-r64-full-adaptive "
     f"--gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 2: HYBRID MECHANISM (sweep #13, 2026-06-10)
    # Hypernet (multiplicative, shared-structure) on the base + LoRA (additive,
    # per-cohort) on top. Hypothesis: combines parameter efficiency of hypernet
    # with isolation of LoRA. Uses hypernet's best-known recipe (singletons +
    # anchor reg) for the gating side; LoRA at modest rank for the per-cohort side.
    # ============================================================================

    # Hybrid with hypernet's known-good recipe + small LoRA add-on
    ("n3-hybrid-d8-r16-lora32",
     f"--variant hybrid {COMMON} --d-embed 8 --rank 16 --hybrid-lora-rank 32 "
     f"--init singletons --lambda-anchor 0.01 --warmstart-iters 1000 "
     f"--gate-attention --gate-embedding"),

    # Hybrid with bigger LoRA component
    ("n3-hybrid-d8-r16-lora64",
     f"--variant hybrid {COMMON} --d-embed 8 --rank 16 --hybrid-lora-rank 64 "
     f"--init singletons --lambda-anchor 0.01 --warmstart-iters 1000 "
     f"--gate-attention --gate-embedding"),

    # Hybrid with the N=5 recipe (low_hamming init)
    ("n5-hybrid-d8-r16-lora32",
     f"--variant hybrid {COMMON} {N5_DATA} --d-embed 8 --rank 16 --hybrid-lora-rank 32 "
     f"--init low_hamming --lambda-anchor 0.01 --warmstart-iters 1000 "
     f"--gate-attention --gate-embedding"),

    # Hybrid without hypernet warmstart (random hypernet init) — test if LoRA carries the load
    ("n3-hybrid-d8-r16-lora64-noinit",
     f"--variant hybrid {COMMON} --d-embed 8 --rank 16 --hybrid-lora-rank 64 "
     f"--init uniform --lambda-anchor 0.01 --warmstart-iters 100 "
     f"--gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 2.1: literature-driven interventions (sweep #14, 2026-06-10)
    # Each intervention paired with the baseline it should improve. See
    # LITERATURE_REVIEW_v2.md (HAT/HYWA/SMEAR) and LITERATURE_REVIEW.md (Concept
    # Sliders / rsLoRA / MoLE).
    # ============================================================================

    # (1) Concept Sliders cohort-contrast loss on top of best small-rank LoRA.
    # Pairs with: n3-lora-baseR8-r64-full.
    # Hypothesis: pushing cohorts apart in delta-space directly should improve
    # slider strength without rank inflation.
    ("n3-lora-baseR8-r64-contrast",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 "
     f"--cohort-contrast-lambda 0.001 "
     f"--gate-attention --gate-embedding"),

    # (2) rsLoRA (α/√r scaling) on top of the high-rank LoRA baseline.
    # Pairs with: n3-lora-baseR8-r128-full.
    # Hypothesis: at r=128 the standard 1/r scaling gives vanishingly small
    # deltas at init; √r restores update magnitude → cohorts can specialize.
    ("n3-lora-baseR8-r128-rslora",
     f"--variant lora {COMMON} --rank 128 --base-rank 8 --rslora "
     f"--gate-attention --gate-embedding"),

    # (3) HAT gate-temperature annealing on hypernet (NOT LoRA).
    # Pairs with: hypernet-singletons-full.
    # Hypothesis: soft gates (T=0.2) early let the base learn shared structure;
    # hard gates (T=3.0) late enforce cohort separation.
    ("n3-hypernet-anneal",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 "
     f"--hat-temp-start 0.2 --hat-temp-end 3.0 "
     f"--gate-attention --gate-embedding"),

    # (4) SMEAR-style higher LR for capacity (router) params.
    # Pairs with: n3-lora-baseR8-r64-full-adaptive.
    # Hypothesis: cap params need a faster LR than weight tensors so the
    # allocation re-balances on a meaningful timescale.
    ("n3-lora-baseR8-r64-adaptive-caplr5",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--cap-lr-multiplier 5.0 "
     f"--gate-attention --gate-embedding"),

    # (5) MoLE load-balance loss on top of the highest-rank LoRA baseline.
    # Pairs with: n3-lora-baseR8-r256-full.
    # Hypothesis: at r=256 some cohorts dominate the delta budget; balancing
    # variance forces uniform per-cohort utilization.
    ("n3-lora-baseR8-r256-balance",
     f"--variant lora {COMMON} --rank 256 --base-rank 8 "
     f"--load-balance-lambda 0.01 "
     f"--gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 3: BIAS-ANCHOR mechanism (hard-α-gated additive residual-stream bias)
    #
    # Hypothesis: existing slider mechanisms (hypernet/LoRA) have a path through
    # which the model can soft-ignore α — either via sigmoid saturation in the
    # gate (hypernet) or by learning small per-cohort deltas (LoRA). The
    # bias-anchor mechanism adds, at every layer, residual += α @ B_layer where
    # B_layer is a learned (N, d_model) matrix. α is a hard linear coefficient
    # (no sigmoid, no learned function-of-α), B is initialized to zero so iter-0
    # behavior matches the un-anchored baseline. The training-time gradient on B
    # pushes it away from zero per-cohort, giving every layer an unambiguous
    # α signal. Should fix the prompt-anchoring failure documented in the
    # qualitative samples (TinyStories attractor basin).
    #
    # Composes with any --variant. We sweep across the strongest base recipes:
    #   - n3-hypernet-singletons-full (prior champion, contrast=+0.508)
    #   - n3-hybrid-d8-r16-lora64    (Phase 2 winner, contrast=+0.587)
    #   - n3-lora-baseR8-r64-full    (LoRA reference, contrast=+0.252)
    #   - n3-ungated-biasanchor      (PURE bias-anchor; no other slider — isolates
    #                                 the mechanism's contribution)
    #   - n5-lora-r32-full           (BIG SURPRISE per qualitative — contrast=0.004
    #                                 but decisive style switches)
    #   - n5-hybrid-d8-r16-lora32    (N=5 contrast leader, +0.551)
    # ============================================================================

    ("n3-hypernet-singletons-full-biasanchor",
     f"--variant hypernet {COMMON} --d-embed 8 --rank 16 --init singletons "
     f"--lambda-anchor 0.01 --warmstart-iters 1000 --bias-anchor "
     f"--gate-attention --gate-embedding"),

    ("n3-hybrid-d8-r16-lora64-biasanchor",
     f"--variant hybrid {COMMON} --d-embed 8 --rank 16 --hybrid-lora-rank 64 "
     f"--init singletons --lambda-anchor 0.01 --warmstart-iters 1000 "
     f"--bias-anchor --gate-attention --gate-embedding"),

    ("n3-lora-baseR8-r64-full-biasanchor",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --bias-anchor "
     f"--gate-attention --gate-embedding"),

    # Pure bias-anchor: ungated weights, only the additive bias gives α a path.
    # Isolates the mechanism's standalone contribution.
    ("n3-ungated-biasanchor",
     f"--variant ungated {COMMON} --bias-anchor"),

    ("n5-lora-r32-full-biasanchor",
     f"--variant lora {COMMON} {N5_DATA} --rank 32 --bias-anchor "
     f"--gate-attention --gate-embedding"),

    ("n5-hybrid-d8-r16-lora32-biasanchor",
     f"--variant hybrid {COMMON} {N5_DATA} --d-embed 8 --rank 16 "
     f"--hybrid-lora-rank 32 --init singletons --lambda-anchor 0.01 "
     f"--warmstart-iters 1000 --bias-anchor "
     f"--gate-attention --gate-embedding"),

    # ============================================================================
    # PHASE 4: SCALE-UP + EQUAL-LOSS sweep
    #
    # Andrew's hypothesis: equal val_losses across cohorts → no easy attractor →
    # cleaner slider. Evidence: `n3-lora-baseR2-r128-full-adaptive` IS the only
    # variant with near-equal per-cohort diag (spread 0.026 nats!) AND has
    # contrast +0.537. But its absolute diag is 7.11 — terrible quality.
    #
    # Structural finding (qualitative agent, 2026-06-10):
    # `ungated-10k` has LOST shake capability — outputs TinyStories on Shakespeare
    # prompts even though `ungated-shake-only` emits real Folger-style text. The
    # shared backbone at 6L-384d for 10k iters cannot retain Shakespeare against
    # the TinyStories attractor. ALL existing sliders also fail to recover shake
    # (cover ≤ 3/10). The fix needs more shared-backbone capacity (bigger model)
    # AND/OR more iters AND/OR explicit loss balance.
    #
    # Three tests at 8L-512d (~2× params over baseline):
    #   1. n3-lora-8L-512d-baseR2-r128-adaptive: scale up the proven-equal-loss
    #      recipe. Should equalize losses, hopefully WITHOUT the 7.11 diag.
    #   2. n3-lora-8L-512d-baseR8-r64-warmstart: bigger model, baseR=8, load+freeze
    #      caps from baseR=2-adaptive (2-stage workflow: discover balance small,
    #      train quality big).
    #   3. n3-lora-8L-512d-baseR8-r64-adaptive-caplr20: bigger model + cap_lr 20×.
    #      Tests whether ULTRA-aggressive cap LR finally moves the equilibrium at
    #      baseR=8.
    # ============================================================================

    ("n3-lora-8L-512d-baseR2-r128-adaptive",
     f"--variant lora {COMMON} --rank 128 --base-rank 2 --adaptive-capacity "
     f"--gate-attention --gate-embedding "
     f"--n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L-512d-baseR8-r64-warmstart-20k",
     f"--variant lora {COMMON.replace('--n-iters 10000', '--n-iters 20000')} "
     f"--rank 64 --base-rank 8 --adaptive-capacity "
     f"--gate-attention --gate-embedding "
     f"--n-layer 8 --n-head 8 --n-embd 512 "
     f"--load-caps-from hf:iamtrask/abcGPT-nano-3:n3-lora-baseR2-r128-full-adaptive "
     f"--freeze-caps"),

    ("n3-lora-8L-512d-baseR8-r64-adaptive-caplr20",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--cap-lr-multiplier 20.0 "
     f"--gate-attention --gate-embedding "
     f"--n-layer 8 --n-head 8 --n-embd 512"),

    # ============================================================================
    # PHASE 4.1: CORNER-BIASED α SAMPLING (sweep, 2026-06-10 PM)
    #
    # Andrew's read on the running Phase-4 pod: "looks good but doesn't prefer
    # corners enough." Root cause: after the 1k one-hot curriculum, α is drawn
    # from Dirichlet(1,1,1) = UNIFORM over the simplex, which visits the vertices
    # (the only points the slider is evaluated at) at measure zero. So contrast
    # stays shallow (~0.06-0.11 nats/column on n3-lora-baseR8-r64-full) and decays
    # over the LR tail.
    #
    # NOTE the nano-2 "Uniform beats Beta_half, don't concentrate at corners"
    # finding does NOT transfer here: that failure mode was a starved off-SLOT in
    # the dual-slot scheme. In LoRA-per-cohort the batch corpus is matched to α
    # (p=alpha_np), so a corner gives the active adapter FULL gradient on its own
    # data — no starvation. Corner-concentration is the right lever for this mech.
    #
    # Two new knobs in train.py: --alpha-concentration (Dirichlet conc <1 piles
    # mass on the simplex boundary = corners+edges) and --corner-prob (hold a
    # fraction of post-curriculum steps at pure one-hot). cohort_contrast_lambda
    # is kept at 0 so the win is attributable to SAMPLING alone, not an aux loss.
    #
    # Built on the running Phase-4 winner geometry (8L-512d, baseR8-r64) so it's
    # apples-to-apples against n3-lora-8L-512d-baseR8-r64-* already on HF.
    # ============================================================================

    # Pure Dirichlet-boundary: conc=0.3, no explicit corner injection. Isolates
    # the concentration knob.
    ("n3-lora-8L-512d-baseR8-r64-conc0.3",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--alpha-concentration 0.3 "
     f"--gate-attention --gate-embedding "
     f"--n-layer 8 --n-head 8 --n-embd 512"),

    # Explicit corner injection: 30% of post-curriculum steps held at one-hot,
    # interior still uniform. Isolates the corner-prob knob.
    ("n3-lora-8L-512d-baseR8-r64-corner0.3",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.3 "
     f"--gate-attention --gate-embedding "
     f"--n-layer 8 --n-head 8 --n-embd 512"),

    # The recommended combo: boundary-concentrated Dirichlet + 25% corner hold +
    # mid-edge coverage so the 2-cohort blends don't hollow out. This is the
    # "train the corners more, keep the edges coherent" recipe.
    ("n3-lora-8L-512d-baseR8-r64-corner0.25-conc0.3-mid0.2",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.25 --alpha-concentration 0.3 --mid-edge-prob 0.2 "
     f"--gate-attention --gate-embedding "
     f"--n-layer 8 --n-head 8 --n-embd 512"),

    # Aggressive corner preference: conc=0.15 + 40% corner hold. Tests whether
    # pushing harder keeps sharpening contrast or starts hollowing the slider mid.
    ("n3-lora-8L-512d-baseR8-r64-corner0.4-conc0.15",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.4 --alpha-concentration 0.15 "
     f"--gate-attention --gate-embedding "
     f"--n-layer 8 --n-head 8 --n-embd 512"),

    # ============================================================================
    # PHASE 4.2: ANTI-COLLAPSE LEVERS (sweep, 2026-06-10 PM)
    #
    # Live diagnosis on the Phase-4.1 pods: the corner-val table shows the
    # DIAGONAL ~flat (pinned near the joint floor ~2.2-2.3) while OFF-DIAGONALS
    # collapse from ~ln(103)≈4.6 toward the diagonal — i.e. contrast eroding to 0.
    # Root cause: contrast lives ONLY in the cohort deltas (the base is
    # α-independent), and on the lora-adaptive variant base_scale=exp(base_log_cap)
    # is FREE + UNREGULARIZED (lambda_anchor doesn't apply to lora) while wd=0.1
    # shrinks the deltas. The optimizer minimizes average loss by pouring capability
    # into the shared base, which helps at every α, neutralizing the deltas.
    #
    # Three levers to force cohort-distinct capability back into the deltas and
    # keep off-diagonals high. Sampling held FIXED at the recommended combo
    # (corner 0.25 / conc 0.3 / mid 0.2) across all six so the only thing varying
    # is the lever. Baseline included for a same-seed same-batch control.
    #   lever 1 --wrong-corner-lambda : penalize low off-diagonal loss (push it up)
    #   lever 2 --freeze-base-scale   : pin base_scale=1 so the base can't inflate
    #   lever 3 --no-decay-deltas     : stop wd from shrinking the U/V deltas
    # ============================================================================

    ("n3-lora-8L512d-lv-baseline",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.25 --alpha-concentration 0.3 --mid-edge-prob 0.2 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-lv-wrongc0.3",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.25 --alpha-concentration 0.3 --mid-edge-prob 0.2 "
     f"--wrong-corner-lambda 0.3 --wrong-corner-margin 2.0 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-lv-wrongc1.0",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.25 --alpha-concentration 0.3 --mid-edge-prob 0.2 "
     f"--wrong-corner-lambda 1.0 --wrong-corner-margin 2.0 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-lv-freezebase",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.25 --alpha-concentration 0.3 --mid-edge-prob 0.2 "
     f"--freeze-base-scale "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-lv-nodecaydelta",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.25 --alpha-concentration 0.3 --mid-edge-prob 0.2 "
     f"--no-decay-deltas "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    # All three stacked: wrong-corner pushes off-diagonals up, freeze-base stops
    # the base inflating (and absorbs lever-1's base-corruption risk), no-decay
    # keeps the deltas from shrinking. Expected best on the diagonal-down /
    # off-diagonal-high objective if the levers are complementary.
    ("n3-lora-8L512d-lv-all3",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--corner-prob 0.25 --alpha-concentration 0.3 --mid-edge-prob 0.2 "
     f"--wrong-corner-lambda 0.3 --wrong-corner-margin 2.0 --freeze-base-scale --no-decay-deltas "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    # ============================================================================
    # PHASE 4.3: CORNERS-ONLY (sweep, 2026-06-10 PM)
    #
    # Andrew: train LITERALLY only on the corners — never switch to the Dirichlet
    # distribution. Implemented by extending the one-hot curriculum to the full
    # run (--alpha-curriculum-until == n_iters): every step is α=one-hot with the
    # batch corpus matched to that corner; the Dirichlet / mid-edge / corner-prob
    # branches are never reached.
    #
    # This is the BTM extreme: each cohort's delta is trained ONLY as a standalone
    # specialist at its own corner. Expect the SHARPEST corners (best diag + best
    # contrast) but the slider INTERIOR is never trained — mid-α interpolation may
    # be incoherent/jumpy. The edge-curve eval is the thing to watch for that.
    # ============================================================================

    ("n3-lora-8L512d-cornersonly",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--alpha-curriculum-until 10000 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    # Corners-only + freeze-base: corner training keeps deltas useful, freeze-base
    # stops the shared base from inflating during the long single-corner stretch.
    ("n3-lora-8L512d-cornersonly-freezebase",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--alpha-curriculum-until 10000 --freeze-base-scale "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    # Corners-only + all three anti-collapse levers stacked.
    ("n3-lora-8L512d-cornersonly-all3",
     f"--variant lora {COMMON} --rank 64 --base-rank 8 --adaptive-capacity "
     f"--alpha-curriculum-until 10000 --wrong-corner-lambda 0.3 --wrong-corner-margin 2.0 "
     f"--freeze-base-scale --no-decay-deltas "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    # ============================================================================
    # PHASE 4.5: CORNERS-ONLY CAPACITY-REALLOCATION SPECTRUM (2026-06-10 PM)
    #
    # ALL corners-only (--alpha-curriculum-until 10000): every step is α=one-hot on
    # the matched corpus, the interior is NEVER trained. Andrew accepts the broken-
    # middle constraint and wants to watch the spectrum as capacity moves from "all
    # in the cohorts (full independent models, ~no base)" to "mostly shared base,
    # small cohort tweaks". So we co-vary: cohort rank DOWN, base_rank UP, monotone.
    #
    #   point        base_rank  cohort_rank   character
    #   co-bR1-r512      1          512        full-rank cohorts = independent models
    #   co-bR2-r384      2          384
    #   co-bR8-r256      8          256
    #   co-bR24-r192     24         192
    #   co-bR64-r128     64         128
    #   co-bR128-r96     128         96
    #   co-bR256-r64     256         64
    #   co-bRfull-r32   full         32        base-dominated, tiny cohort tweaks
    #
    # Train-matrix + centroid middle-readout build, so the diagonal overfit AND the
    # untrained-middle instability are both visible per step. Final edge curves show
    # the interpolation. 8L-512d, adaptive, full gating.
    # ============================================================================

    # rsLoRA version of the coba spectrum: every point adds --rslora (delta scaled
    # by 1/sqrt(rank)) so the high-rank cohort deltas aren't mis-scaled. Diagnosis:
    # the no-rslora coba runs invert (most-params = worst val) because unscaled
    # rank-256/512 deltas stall — proven on n3-lora-baseR8-r128 (no-rslora diag 5.19
    # -> rslora 3.77). Run in PARALLEL with the live coba pods for an A/B on rsLoRA.
    # Family "cobars" = corners-only + bias-anchor + rsLoRA.
    ("n3-lora-8L512d-cobars-bR1-r512",
     f"--variant lora {COMMON} --rank 512 --base-rank 1 --adaptive-capacity --bias-anchor --rslora "
     f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-cobars-bR2-r384",
     f"--variant lora {COMMON} --rank 384 --base-rank 2 --adaptive-capacity --bias-anchor --rslora "
     f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-cobars-bR8-r256",
     f"--variant lora {COMMON} --rank 256 --base-rank 8 --adaptive-capacity --bias-anchor --rslora "
     f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-cobars-bR24-r192",
     f"--variant lora {COMMON} --rank 192 --base-rank 24 --adaptive-capacity --bias-anchor --rslora "
     f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-cobars-bR64-r128",
     f"--variant lora {COMMON} --rank 128 --base-rank 64 --adaptive-capacity --bias-anchor --rslora "
     f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-cobars-bR128-r96",
     f"--variant lora {COMMON} --rank 96 --base-rank 128 --adaptive-capacity --bias-anchor --rslora "
     f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-cobars-bR256-r64",
     f"--variant lora {COMMON} --rank 64 --base-rank 256 --adaptive-capacity --bias-anchor --rslora "
     f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    ("n3-lora-8L512d-cobars-bRfull-r32",
     f"--variant lora {COMMON} --rank 32 --base-rank -1 --adaptive-capacity --bias-anchor --rslora "
     f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
     f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512"),

    # ============================================================================
    # PHASE 4.6: HYBRID — subtle hypernet on the base + LoRA cohort deltas (2026-06-10)
    #
    # The hybrid mechanism: W_eff(α) = W_shared * hypernet_gate(α) + Σ α_c·(U_c V_cᵀ).
    # A SUBTLE per-cohort hypernet (d_embed=4, hypernet rank=8) multiplicatively
    # modulates the FULL shared base; LoRA deltas add per-cohort capacity on top.
    # Hybrid has no base_rank (its base is the full hypernet-gated weight), so the
    # sweep axis is the LoRA cohort rank, on a fixed subtle-hypernet base.
    #
    # ALL the recent upgrades carried in: rsLoRA (just ported into the hybrid path —
    # was unscaled, same high-rank stall), corners-only, bias-anchor, full gating,
    # train+val matrices, middle readout, samples. Light warmstart + anchor keep the
    # hypernet gentle (gate starts ~identity). vs the cobars (pure-LoRA) spectrum:
    # does a subtle cohort-conditioned base modulation beat LoRA-on-a-plain-base?
    # ============================================================================

    *[(f"n3-hyb-8L512d-co-r{lr}",
       f"--variant hybrid {COMMON} --d-embed 4 --rank 8 --hybrid-lora-rank {lr} "
       f"--init uniform --lambda-anchor 0.01 --warmstart-iters 200 --bias-anchor --rslora "
       f"--alpha-curriculum-until 10000 --edge-curve-points 9 "
       f"--gate-attention --gate-embedding --n-layer 8 --n-head 8 --n-embd 512")
      for lr in (16, 32, 64, 96, 128, 192, 256, 384)],
]


def launch_pod(runpod_module, name, gpu_type, cloud_type, image, env, dry_run):
    if dry_run:
        print(f"  [DRY-RUN] would create pod '{name}' on {gpu_type} ({cloud_type})")
        return None
    pod = runpod_module.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type,
        cloud_type=cloud_type,
        gpu_count=1,
        volume_in_gb=0,
        container_disk_in_gb=20,
        env=env,
        # `sleep infinity` after on_box.sh prevents RunPod restart-loops.
        docker_args="bash -c 'curl -fsSL https://raw.githubusercontent.com/iamtrask/abcGPT/main/experiments/nano-3/runpod/on_box.sh | bash; sleep infinity'",
    )
    return pod["id"]


def run_until_done(
    runpod_module,
    pending_queue,
    gpu_type,
    cloud_type,
    image,
    hf_token,
    hf_repo,
    git_sha,
    poll_interval=30,
    timeout=7200,
    max_retries_per_chunk=30,
    force_rerun=False,
):
    """Drive sweep to completion: launches, retries, monitoring, termination.

    Same continuous-saturation loop as nano-2's runpod_fanout — adapted for the
    nano-3 HF target.
    """
    from huggingface_hub import HfApi, login

    login(token=hf_token, add_to_git_credential=False)
    api = HfApi()

    tracked = {}  # {pod_id: variant_name}
    retry_counts = {}
    start = time.time()
    last_status_at = 0
    total = len(pending_queue)

    def try_launch(idx, variant_name, train_args):
        env = {
            "VARIANT_NAME": variant_name,
            "TRAIN_ARGS": train_args,
            "HF_TOKEN": hf_token,
            "HF_REPO": hf_repo,
            "GIT_SHA": git_sha,
        }
        try:
            return launch_pod(runpod_module, f"nano-3-{variant_name[:30]}",
                                gpu_type, cloud_type, image, env, dry_run=False)
        except Exception:
            return None

    print(f"\nDriving {total} variants to completion...")
    print(f"(poll every {poll_interval}s, timeout {timeout/60:.0f} min, "
          f"max {max_retries_per_chunk} retries per variant)")

    while (pending_queue or tracked) and time.time() - start < timeout:
        # 1. Poll HF for summary.json (the "done" marker; log.jsonl is for liveness).
        completed = set()
        try:
            for f in api.list_repo_tree(hf_repo, repo_type="model", recursive=True):
                path = getattr(f, "path", None)
                if path and "/" in path and path.endswith("/summary.json"):
                    completed.add(path.split("/")[0])
        except Exception as exc:
            msg = str(exc)
            if "404" in msg or "RepositoryNotFoundError" in msg or "not found" in msg.lower():
                pass  # repo doesn't exist yet — will be created by first upload
            else:
                print(f"  HF query failed transiently: {exc}; will retry")
                time.sleep(poll_interval)
                continue

        # 2. Terminate pods whose variant has uploaded summary.json
        for pid in list(tracked.keys()):
            if tracked[pid] in completed:
                print(f"  ✓ {tracked[pid]} uploaded to HF")
                try:
                    runpod_module.terminate_pod(pid)
                    print(f"  ✓ pod {pid} terminated")
                except Exception as exc:
                    print(f"  ⚠ terminate failed for {pid}: {exc}")
                del tracked[pid]

        # 3. Launch from pending queue
        new_queue = []
        launched = 0
        skipped = 0
        for idx, (variant_name, train_args) in pending_queue:
            if variant_name in completed and not force_rerun:
                skipped += 1
                continue
            retry_counts[idx] = retry_counts.get(idx, 0) + 1
            if retry_counts[idx] > max_retries_per_chunk:
                print(f"  ⚠ {variant_name} exceeded {max_retries_per_chunk} retries; giving up")
                continue
            pid = try_launch(idx, variant_name, train_args)
            if pid:
                tracked[pid] = variant_name
                launched += 1
            else:
                new_queue.append((idx, (variant_name, train_args)))
        if launched:
            print(f"  + launched {launched} pods (queue now: {len(new_queue)})")
        if skipped:
            print(f"  ~ skipped {skipped} variants already on HF")
        pending_queue = new_queue

        if (pending_queue or tracked) and time.time() - last_status_at > poll_interval * 5:
            elapsed_min = (time.time() - start) / 60
            print(f"  [{elapsed_min:.1f}min] tracked: {len(tracked)} pods, "
                  f"pending: {len(pending_queue)}, HF has {len(completed)} variants")
            last_status_at = time.time()

        if pending_queue or tracked:
            time.sleep(poll_interval)

    if pending_queue or tracked:
        print(f"\n⚠ Timeout. {len(pending_queue)} unlaunched, {len(tracked)} alive.")
        for pid, v in tracked.items():
            print(f"  ⚠ orphan: {pid} ({v})")
    else:
        print(f"\n✅ All variants done. Total time: {(time.time() - start) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="Comma-separated variant names (default: full sweep)")
    p.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    p.add_argument("--cloud-type", default=DEFAULT_CLOUD_TYPE, choices=["COMMUNITY", "SECURE"])
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    p.add_argument("--git-sha", default="main")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-rerun", action="store_true",
                   help="Re-run variants even if summary.json is already on HF.")
    args = p.parse_args()

    sweep = SWEEP_DEFAULT
    if args.only:
        wanted = set(s.strip() for s in args.only.split(",") if s.strip())
        sweep = [(n, a) for (n, a) in sweep if n in wanted]
        if not sweep:
            sys.exit(f"error: no variants in default sweep matched --only={args.only}")

    print("Plan:")
    print(f"  Variants: {len(sweep)}")
    for n, a in sweep:
        print(f"    {n:<32} {a}")
    print(f"  GPU: {args.gpu_type}  ({args.cloud_type})")
    print(f"  HF repo: {args.hf_repo}")
    print(f"  Git SHA: {args.git_sha}")

    per_pod_hr = 0.4
    rate_table = {
        ("NVIDIA GeForce RTX 4090", "SECURE"):    0.69,
        ("NVIDIA GeForce RTX 4090", "COMMUNITY"): 0.34,
        ("NVIDIA GeForce RTX 3090", "SECURE"):    0.44,
        ("NVIDIA GeForce RTX 3090", "COMMUNITY"): 0.22,
        ("NVIDIA RTX A5000", "SECURE"):           0.40,
        ("NVIDIA RTX A5000", "COMMUNITY"):        0.20,
    }
    cost_per_hr = rate_table.get((args.gpu_type, args.cloud_type), 0.50)
    print(f"  Est rate: ${cost_per_hr}/hr  (est per variant: ~${per_pod_hr * cost_per_hr:.2f})")
    print(f"  Est total cost: ~${len(sweep) * per_pod_hr * cost_per_hr:.2f}")
    print(f"  Wall-clock: ~30-40 min (bounded by slowest cold-start pod)")
    print()

    if args.dry_run:
        print("--dry-run; no launches.")
        return

    api_key = os.environ.get("RUNPOD_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    if not api_key:
        sys.exit("error: RUNPOD_API_KEY not set")
    if not hf_token:
        sys.exit("error: HF_TOKEN not set")
    try:
        import runpod
    except ImportError:
        sys.exit("error: 'runpod' not installed. pip install runpod")
    runpod.api_key = api_key

    pending = [(i, item) for i, item in enumerate(sweep)]
    run_until_done(
        runpod, pending,
        gpu_type=args.gpu_type, cloud_type=args.cloud_type, image=args.image,
        hf_token=hf_token, hf_repo=args.hf_repo,
        git_sha=args.git_sha,
        force_rerun=args.force_rerun,
    )

    print(f"\nResults on HF: https://huggingface.co/{args.hf_repo}")
    print(f"Pull locally with:")
    print(f"  huggingface-cli download {args.hf_repo} --local-dir experiments/nano-3/results --repo-type model")


if __name__ == "__main__":
    main()
