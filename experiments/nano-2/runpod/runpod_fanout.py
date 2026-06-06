#!/usr/bin/env python3
"""runpod_fanout.py — launch a nano-2 sweep across N RunPod pods.

Each variant in the sweep gets its own pod. The pod runs experiments/nano-2/train.py
with the variant's specific flags, then uploads results to HF Hub at
iamtrask/abcGPT-nano-2. Pods auto-terminate via the HF-completion monitor.

The default sweep is 13 variants — see SWEEP_DEFAULT below. Edit that list to
add / remove / tweak conditions.

Usage:
    # 13-variant sweep (~30 min wall, ~$3-5 compute)
    RUNPOD_API_KEY=... HF_TOKEN=... python runpod_fanout.py

    # smoke: one variant only
    python runpod_fanout.py --only fixed-mn-replicate

    # dry-run (show plan, no launches)
    python runpod_fanout.py --dry-run

Prerequisites: pip install runpod huggingface_hub
Env: RUNPOD_API_KEY, HF_TOKEN
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path


DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_HF_REPO = "iamtrask/abcGPT-nano-2"


# The 13-variant sweep we agreed on. Each entry is:
#   (variant_name, train_args_string)
# train_args_string is passed directly to train.py (variant_name is set separately).
SWEEP_DEFAULT = [
    # --- baselines (no gate / fixed gate) -----------------------------------
    ("ungated-10k",
     "--variant ungated --n-iters 10000"),

    ("fixed-mn-replicate",
     "--variant fixed-mn --n-iters 10000 --span 1.0 --alpha-dist beta_half"),

    ("fixed-mn-span0.3",
     "--variant fixed-mn --n-iters 10000 --span 0.3 --alpha-dist beta_half"),

    ("fixed-mn-span0.7",
     "--variant fixed-mn --n-iters 10000 --span 0.7 --alpha-dist beta_half"),

    ("fixed-mn-bimodal-alpha",
     "--variant fixed-mn --n-iters 10000 --span 1.0 --alpha-dist bimodal_endpoints"),

    # --- trainable m_n: LR ratio sweep --------------------------------------
    ("trainable-mn-lrratio1e-2",
     "--variant trainable-mn --n-iters 10000 --span 1.0 --mask-lr-ratio 1e-2 --lambda-var 0.1 --warmup-mn 0"),

    ("trainable-mn-lrratio1e-3",
     "--variant trainable-mn --n-iters 10000 --span 1.0 --mask-lr-ratio 1e-3 --lambda-var 0.1 --warmup-mn 0"),

    ("trainable-mn-lrratio1e-4",
     "--variant trainable-mn --n-iters 10000 --span 1.0 --mask-lr-ratio 1e-4 --lambda-var 0.1 --warmup-mn 0"),

    # --- trainable m_n: lambda_var sweep at moderate LR ratio ---------------
    ("trainable-mn-lambdavar0.01",
     "--variant trainable-mn --n-iters 10000 --span 1.0 --mask-lr-ratio 1e-3 --lambda-var 0.01 --warmup-mn 0"),

    ("trainable-mn-lambdavar1.0",
     "--variant trainable-mn --n-iters 10000 --span 1.0 --mask-lr-ratio 1e-3 --lambda-var 1.0 --warmup-mn 0"),

    # --- trainable m_n: warmup ---------------------------------------------
    ("trainable-mn-warmup1000",
     "--variant trainable-mn --n-iters 10000 --span 1.0 --mask-lr-ratio 1e-3 --lambda-var 0.1 --warmup-mn 1000"),

    # --- trainable m_n: span variations ------------------------------------
    ("trainable-mn-span0.3",
     "--variant trainable-mn --n-iters 10000 --span 0.3 --mask-lr-ratio 1e-3 --lambda-var 0.1 --warmup-mn 0"),

    ("trainable-mn-span0.7",
     "--variant trainable-mn --n-iters 10000 --span 0.7 --mask-lr-ratio 1e-3 --lambda-var 0.1 --warmup-mn 0"),
]


def launch_pod(runpod_module, name, gpu_type, image, env, dry_run):
    """Create a single RunPod pod for one variant."""
    if dry_run:
        print(f"  [DRY-RUN] would create pod '{name}' on {gpu_type}")
        return None
    pod = runpod_module.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type,
        cloud_type="SECURE",  # nano-1 sweep confirmed SECURE has capacity when COMMUNITY doesn't.
        gpu_count=1,
        volume_in_gb=0,
        container_disk_in_gb=20,
        env=env,
        docker_args="bash -c 'curl -fsSL https://raw.githubusercontent.com/iamtrask/abcGPT/main/experiments/nano-2/runpod/on_box.sh | bash; sleep 60'",
    )
    return pod["id"]


def run_until_done(
    runpod_module,
    pending_queue,
    gpu_type,
    image,
    hf_token,
    hf_repo,
    git_sha,
    poll_interval=30,
    timeout=7200,
    max_retries_per_chunk=30,
):
    """Drive sweep to completion: initial launches, retries, monitoring, termination.

    Same continuous-saturation loop as nano-1's runpod_fanout — adapted to
    track variant_names instead of source_names.
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
            return launch_pod(runpod_module, f"nano-2-{variant_name[:30]}", gpu_type, image, env, dry_run=False)
        except Exception:
            return None

    print(f"\nDriving {total} variants to completion...")
    print(f"(poll every {poll_interval}s, timeout {timeout/60:.0f} min, max {max_retries_per_chunk} retries per variant)")

    while (pending_queue or tracked) and time.time() - start < timeout:
        # 1. Poll HF for completed variants — each variant uploads a log.jsonl
        completed = set()
        try:
            for f in api.list_repo_tree(hf_repo, repo_type="model", recursive=True):
                path = getattr(f, "path", None)
                if path and "/" in path and path.endswith("/log.jsonl"):
                    completed.add(path.split("/")[0])
        except Exception as exc:
            print(f"  HF query failed: {exc}; will retry")
            time.sleep(poll_interval)
            continue

        # 2. Terminate pods whose variant has uploaded
        for pid in list(tracked.keys()):
            if tracked[pid] in completed:
                print(f"  ✓ {tracked[pid]} uploaded to HF")
                try:
                    runpod_module.terminate_pod(pid)
                    print(f"  ✓ pod {pid} terminated")
                except Exception as exc:
                    print(f"  ⚠ terminate failed for {pid}: {exc}")
                del tracked[pid]

        # 3. Try launching from pending queue
        new_queue = []
        launched = 0
        skipped = 0
        for idx, (variant_name, train_args) in pending_queue:
            if variant_name in completed:
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
            print(f"  [{elapsed_min:.1f}min] tracked: {len(tracked)} pods, pending: {len(pending_queue)}, HF has {len(completed)} variants")
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
    p.add_argument("--only", default="", help="Comma-separated variant names to run (default: full sweep)")
    p.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    p.add_argument("--git-sha", default="main")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sweep = SWEEP_DEFAULT
    if args.only:
        wanted = set(s.strip() for s in args.only.split(",") if s.strip())
        sweep = [(n, a) for (n, a) in sweep if n in wanted]
        if not sweep:
            sys.exit(f"error: no variants in default sweep matched --only={args.only}")

    print(f"Plan:")
    print(f"  Variants: {len(sweep)}")
    for n, a in sweep:
        print(f"    {n:<32} {a}")
    print(f"  GPU: {args.gpu_type}")
    print(f"  HF repo: {args.hf_repo}")
    print(f"  Git SHA: {args.git_sha}")

    # Cost estimate: each variant ~3-13 min training on 4090 + 5-10 min boot/setup
    per_pod_hr = 0.4  # 0.5 hr per variant total wall-clock cap (training + setup)
    cost_per_hr = 0.69
    print(f"  Est cost per variant: ~${per_pod_hr * cost_per_hr:.2f}")
    print(f"  Est total cost: ~${len(sweep) * per_pod_hr * cost_per_hr:.2f}")
    print(f"  Wall-clock: ~30-40 min (bounded by slowest cold-start pod)")
    print()

    if not args.dry_run:
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
    else:
        runpod = None
        hf_token = os.environ.get("HF_TOKEN", "<dry-run>")

    if args.dry_run:
        print("--dry-run; no launches.")
        return

    pending = [(i, item) for i, item in enumerate(sweep)]
    run_until_done(
        runpod, pending,
        gpu_type=args.gpu_type, image=args.image,
        hf_token=hf_token, hf_repo=args.hf_repo,
        git_sha=args.git_sha,
    )

    print(f"\nResults on HF: https://huggingface.co/{args.hf_repo}")
    print(f"Pull locally with:")
    print(f"  huggingface-cli download {args.hf_repo} --local-dir experiments/nano-2/results --repo-type model")


if __name__ == "__main__":
    main()
