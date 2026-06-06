#!/usr/bin/env python3
"""runpod_fanout.py — orchestrate nano-1 training across N RunPod pods.

Spins up N RunPod pods (default 5), chunks the source list across them, runs
on_box.sh on each, and monitors until they finish. Each pod uploads its own
results to the HF repo on completion and self-terminates.

Why RunPod vs Modal: ~4× cheaper per GPU-hour (RunPod 4090 spot @ $0.34/hr vs
Modal T4 @ $0.59/hr), no workspace-level concurrency cap.

Usage:
    # train all 100 sources across 5 boxes (default)
    RUNPOD_API_KEY=... HF_TOKEN=... python runpod_fanout.py

    # train just the sources Modal didn't finish
    python runpod_fanout.py --sources 050_xxx,051_yyy,052_zzz

    # use a different number of boxes
    python runpod_fanout.py --n-boxes 10

    # dry-run (show plan, don't launch)
    python runpod_fanout.py --dry-run

Prerequisites:
    pip install runpod
    Set RUNPOD_API_KEY (https://www.runpod.io/console/user/settings)
    Set HF_TOKEN (https://huggingface.co/settings/tokens, write scope)
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path


DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_HF_REPO = "iamtrask/abcGPT-nano-1-baselines"
DEFAULT_N_BOXES = 5


def list_all_sources():
    """Find the 100 source names from the local repo."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent.parent
    pattern = str(repo_root / "data/100_simple_voices/sources/source_*.txt")
    files = sorted(glob.glob(pattern))
    return [Path(f).stem.replace("source_", "") for f in files]


def chunk_sources(sources: list[str], n_chunks: int) -> list[list[str]]:
    """Split sources roughly evenly across n_chunks boxes."""
    chunks = [[] for _ in range(n_chunks)]
    for i, src in enumerate(sources):
        chunks[i % n_chunks].append(src)
    return [c for c in chunks if c]  # drop empty


def build_on_box_invocation(on_box_script: str, sources_csv: str, hf_token: str, hf_repo: str, git_sha: str) -> str:
    """Build the shell command RunPod runs on pod startup.

    We can't directly upload a file to the pod via the API (well, we could,
    but it's simpler to inline). Instead we curl the on_box.sh script from
    GitHub at runtime (since the abcGPT repo is public, this works).
    """
    on_box_url = "https://raw.githubusercontent.com/iamtrask/abcGPT/main/experiments/nano-1/runpod/on_box.sh"
    return (
        f"export SOURCES='{sources_csv}' "
        f"HF_TOKEN='{hf_token}' "
        f"HF_REPO='{hf_repo}' "
        f"GIT_SHA='{git_sha}' && "
        f"curl -fsSL {on_box_url} | bash"
    )


def launch_pod(runpod_module, name: str, gpu_type: str, image: str, env: dict, dry_run: bool):
    """Create a single RunPod pod and return its ID."""
    if dry_run:
        print(f"  [DRY-RUN] would create pod '{name}' on {gpu_type}")
        return None
    pod = runpod_module.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type,
        cloud_type="SECURE",  # SECURE = on-demand; COMMUNITY = cheaper spot
        gpu_count=1,
        volume_in_gb=20,
        container_disk_in_gb=30,
        env=env,
        # When the container's startup command exits, the pod stops (saves $).
        # The startup command is: run on_box.sh, then exit.
        docker_args="bash -c 'curl -fsSL https://raw.githubusercontent.com/iamtrask/abcGPT/main/experiments/nano-1/runpod/on_box.sh | bash; sleep 60'",
        # NB: sleep 60 at end so logs flush before pod terminates.
    )
    return pod["id"]


def wait_for_pods(runpod_module, pod_ids: list[str], poll_interval: int = 60):
    """Wait until all pods have exited (status != RUNNING). Print progress."""
    pending = set(pod_ids)
    print(f"\nMonitoring {len(pending)} pods (poll every {poll_interval}s)...")
    while pending:
        for pid in list(pending):
            try:
                pod = runpod_module.get_pod(pid)
                status = pod.get("desiredStatus", "?")
                runtime = pod.get("runtime", {})
                if status != "RUNNING" or (runtime and runtime.get("uptimeInSeconds") and not runtime.get("ports")):
                    pending.discard(pid)
                    print(f"  pod {pid}: {status} (done)")
            except Exception as exc:
                print(f"  pod {pid}: query failed ({exc}); will retry")
        if pending:
            print(f"  {len(pending)} pods still running...")
            time.sleep(poll_interval)
    print("\nAll pods terminated.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boxes", type=int, default=DEFAULT_N_BOXES,
                   help=f"Number of RunPod pods to spin up (default: {DEFAULT_N_BOXES})")
    p.add_argument("--sources", default="",
                   help="Comma-separated source names to train. Default: all 100 from local corpus.")
    p.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE,
                   help=f"RunPod GPU type ID (default: '{DEFAULT_GPU_TYPE}')")
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help=f"Docker image (default: {DEFAULT_IMAGE})")
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                   help=f"Target HF repo for results (default: {DEFAULT_HF_REPO})")
    p.add_argument("--git-sha", default="main",
                   help="abcGPT commit to check out on each pod (default: main)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the plan without actually launching pods")
    p.add_argument("--no-wait", action="store_true",
                   help="Launch pods and exit immediately (don't monitor for completion)")
    args = p.parse_args()

    # Source list
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        sources = list_all_sources()
    if not sources:
        sys.exit("error: no sources found (pass --sources or run from a repo with 100_simple_voices)")

    chunks = chunk_sources(sources, args.n_boxes)
    actual_n_boxes = len(chunks)

    # Cost estimate
    # 4090 spot ≈ $0.34/hr; 4090 trains 5000-iter shakespeare_char in ~25 min;
    # 100/N sources per box × 25 min = wall-clock per box; cost = boxes × wall × $/hr
    per_source_min = 25
    sources_per_box_max = max(len(c) for c in chunks)
    wall_clock_hr = sources_per_box_max * per_source_min / 60
    cost_per_box_hr = 0.34  # 4090 spot
    total_cost = actual_n_boxes * wall_clock_hr * cost_per_box_hr

    print(f"Plan:")
    print(f"  Sources: {len(sources)} total")
    print(f"  Boxes: {actual_n_boxes} (requested {args.n_boxes})")
    print(f"  Per box: {[len(c) for c in chunks]} sources")
    print(f"  GPU: {args.gpu_type}")
    print(f"  Image: {args.image}")
    print(f"  HF repo: {args.hf_repo}")
    print(f"  Git SHA: {args.git_sha}")
    print(f"  Estimated wall-clock: ~{wall_clock_hr:.1f} hr")
    print(f"  Estimated cost: ~${total_cost:.2f}")
    print()

    # Auth checks
    if not args.dry_run:
        api_key = os.environ.get("RUNPOD_API_KEY")
        hf_token = os.environ.get("HF_TOKEN")
        if not api_key:
            sys.exit("error: RUNPOD_API_KEY not set. Get one from https://www.runpod.io/console/user/settings")
        if not hf_token:
            sys.exit("error: HF_TOKEN not set. Get one from https://huggingface.co/settings/tokens (write scope)")

        try:
            import runpod
        except ImportError:
            sys.exit("error: 'runpod' package not installed. Run: pip install runpod")
        runpod.api_key = api_key
    else:
        runpod = None
        hf_token = os.environ.get("HF_TOKEN", "<dry-run-no-token>")

    # Launch pods. Individual launch failures (e.g., RunPod out of capacity for
    # this GPU type) are logged but don't kill the whole batch — the sources on
    # failed boxes just stay missing locally, so re-running runpod-resume picks
    # them up next time.
    pod_ids = []
    failed_chunks = []
    for i, chunk in enumerate(chunks):
        sources_csv = ",".join(chunk)
        name = f"nano-1-box-{i:02d}"
        env = {
            "SOURCES": sources_csv,
            "HF_TOKEN": hf_token,
            "HF_REPO": args.hf_repo,
            "GIT_SHA": args.git_sha,
        }
        print(f"Launching box {i:02d} ({len(chunk)} sources: {sources_csv[:60]}{'...' if len(sources_csv) > 60 else ''})")
        try:
            pid = launch_pod(runpod, name, args.gpu_type, args.image, env, args.dry_run)
            if pid:
                pod_ids.append(pid)
                print(f"  pod id: {pid}")
        except Exception as exc:
            print(f"  ⚠ launch failed: {exc}")
            failed_chunks.append(chunk)

    if failed_chunks:
        failed_sources = [s for c in failed_chunks for s in c]
        print(f"\n⚠ {len(failed_chunks)} pods failed to launch ({len(failed_sources)} sources affected).")
        print(f"Sources to retry: {','.join(failed_sources[:10])}{'...' if len(failed_sources) > 10 else ''}")
        print(f"Re-run `./run.sh runpod-resume` after current pods finish; the missing sources will be picked up automatically.")

    if args.dry_run:
        print("\n--dry-run set; no pods launched.")
        return

    print(f"\nLaunched {len(pod_ids)} pods.")
    if args.no_wait:
        print("(--no-wait set; exiting. Monitor at https://www.runpod.io/console/pods)")
        return

    wait_for_pods(runpod, pod_ids)
    print("\nAll boxes done. Results should be on HF Hub.")
    print(f"Pull with: huggingface-cli download {args.hf_repo} --local-dir results-from-hf")


if __name__ == "__main__":
    main()
