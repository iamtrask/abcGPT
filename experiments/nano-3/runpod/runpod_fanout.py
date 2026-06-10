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
