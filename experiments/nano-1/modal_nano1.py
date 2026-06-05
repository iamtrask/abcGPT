"""nano-1 baselines on Modal.

Trains one char-level baseline per source in 100_simple_voices, fanned out
across multiple T4 GPUs in parallel. Karpathy's train_shakespeare_char
config byte-for-byte (only --dataset and --out_dir overridden per run).

Pricing math:
  T4 = ~$0.59/hr. ~12-15 min per baseline. 100 baselines = ~20-25 GPU-hours.
  Total spend: ~$12-15 (well within the $30/mo free credit).
  With parallelism=10, wall-clock = ~2-2.5 hr instead of ~25 hr sequential.

Usage (from the abcGPT repo root):

    # 1. Upload the corpus to the Modal volume (one-time, ~88 MB)
    modal run experiments/nano-1/modal_nano1.py::upload_corpus

    # 2. Smoke-test with one source (~12 min, ~$0.12)
    modal run experiments/nano-1/modal_nano1.py::smoke_test

    # 3. Run all 100 baselines in parallel (~2.5 hr wall-clock, ~$12-15)
    modal run experiments/nano-1/modal_nano1.py::main

    # 4. Download results back to laptop
    modal run experiments/nano-1/modal_nano1.py::download_results
"""

import modal

APP_NAME = "nano-1-baselines"
VOL_NAME = "nano-1"

app = modal.App(APP_NAME)

# Shared volume for corpus + results. Persists across runs.
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# Image: torch w/ CUDA, plus nanoGPT cloned in. Pinned PyTorch CUDA index.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "numpy>=1.26,<2",
        "tiktoken==0.7.0",
        "tqdm",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/iamtrask/abcGPT.git /opt/abcGPT"
    )
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    timeout=60 * 60,  # 1 hr per baseline cap (should finish in ~12-15 min)
)
def train_one(source_name: str) -> dict:
    """Train one baseline on a single source. Returns summary dict."""
    import json
    import os
    import subprocess
    import time

    REPO = "/opt/abcGPT"
    src_file = f"/vol/sources/source_{source_name}.txt"
    data_dir = f"{REPO}/data/100_simple_voices/baselines/data/{source_name}"
    out_dir = f"{REPO}/data/100_simple_voices/baselines/out/{source_name}"
    result_dir = f"/vol/results/{source_name}"

    # If we already have a result on the volume, skip (idempotent reruns)
    if os.path.exists(f"{result_dir}/best_val_loss.txt"):
        with open(f"{result_dir}/best_val_loss.txt") as f:
            bv = float(f.read().strip())
        return {"source": source_name, "best_val_loss": bv, "status": "skipped (already done)"}

    if not os.path.exists(src_file):
        return {"source": source_name, "status": "missing source file"}

    # 1. Char-level prep
    subprocess.run(
        [
            "python",
            f"{REPO}/data/100_simple_voices/baselines/prepare_char.py",
            "--source", src_file,
            "--out", data_dir,
        ],
        cwd=REPO,
        check=True,
    )

    # 2. Train. Override only dataset + out_dir; rest is Karpathy's config.
    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/training.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        subprocess.run(
            [
                "python", "train.py", "config/train_shakespeare_char.py",
                f"--dataset=100_simple_voices/baselines/data/{source_name}",
                f"--out_dir={out_dir}",
                "--wandb_log=False",
                "--always_save_checkpoint=False",
                "--compile=False",  # speeds up cold start on T4
            ],
            cwd=REPO,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=True,
        )
    wall_s = time.time() - t0

    # 3. Extract result to volume
    os.makedirs(result_dir, exist_ok=True)
    subprocess.run(
        [
            "python",
            f"{REPO}/data/100_simple_voices/baselines/extract_result.py",
            "--out-dir", out_dir,
            "--data-dir", data_dir,
            "--result-dir", result_dir,
            "--source-name", source_name,
            "--wall-clock-s", str(wall_s),
        ],
        cwd=REPO,
        check=True,
    )

    # Persist volume writes
    vol.commit()

    # Read the result for the return value
    with open(f"{result_dir}/best_val_loss.txt") as f:
        bv = float(f.read().strip())

    return {
        "source": source_name,
        "best_val_loss": bv,
        "wall_s": round(wall_s, 1),
        "status": "trained",
    }


def _list_local_sources() -> list[str]:
    """Discover 100 source names from local repo."""
    import glob
    import os
    files = sorted(glob.glob("data/100_simple_voices/sources/source_*.txt"))
    return [os.path.basename(f).replace("source_", "").replace(".txt", "") for f in files]


@app.local_entrypoint()
def upload_corpus():
    """Upload data/100_simple_voices/sources/ to the Modal volume.

    Run once before training. ~88 MB upload.
    """
    import subprocess
    print(f"Uploading data/100_simple_voices/sources/ to volume '{VOL_NAME}' at /sources/ ...")
    subprocess.run(
        [
            "modal", "volume", "put", VOL_NAME,
            "data/100_simple_voices/sources/",
            "/sources/",
            "--force",
        ],
        check=True,
    )
    print("Upload complete.")


@app.local_entrypoint()
def smoke_test():
    """Train one source as a smoke test before the big fan-out."""
    src = "077_nba-play-by-play"
    print(f"Smoke test: training {src} on Modal T4 ...")
    result = train_one.remote(src)
    print(f"Result: {result}")


@app.local_entrypoint()
def main(parallel: int = 10, only: str = ""):
    """Fan out training across all 100 sources (or a comma-separated subset).

    Usage:
        modal run modal_nano1.py::main                  # all 100, parallel=10
        modal run modal_nano1.py::main --parallel 20    # all 100, parallel=20
        modal run modal_nano1.py::main --only 077_nba-play-by-play,076_retrosheet-baseball
    """
    if only:
        sources = [s.strip() for s in only.split(",") if s.strip()]
    else:
        sources = _list_local_sources()

    print(f"Training {len(sources)} baselines, parallelism={parallel}")
    print()

    n_trained = n_skipped = n_failed = 0
    for r in train_one.map(sources, max_concurrent_inputs=parallel):
        status = r.get("status", "?")
        bv = r.get("best_val_loss")
        wall = r.get("wall_s")
        bv_str = f"val={bv:.4f}" if bv is not None else "val=?"
        wall_str = f"({wall:.0f}s)" if wall else ""
        print(f"  [{status:>22}] {r['source']:<36} {bv_str} {wall_str}")
        if status == "trained":
            n_trained += 1
        elif "skipped" in status:
            n_skipped += 1
        else:
            n_failed += 1

    print()
    print(f"Summary: {n_trained} trained, {n_skipped} skipped, {n_failed} failed")
    print(f"Results saved to Modal volume '{VOL_NAME}' under /results/")
    print(f"Run 'modal run experiments/nano-1/modal_nano1.py::download_results' to fetch.")


@app.local_entrypoint()
def download_results():
    """Download /results/ from the Modal volume back to local laptop."""
    import subprocess
    local_dir = "data/100_simple_voices/baselines/results/"
    print(f"Downloading /results/ from volume '{VOL_NAME}' to {local_dir} ...")
    subprocess.run(
        [
            "modal", "volume", "get", VOL_NAME,
            "/results/",
            local_dir,
            "--force",
        ],
        check=True,
    )
    print("Download complete.")
    print()
    print("Aggregating ...")
    subprocess.run(
        ["python", "data/100_simple_voices/baselines/aggregate.py"],
        check=True,
    )
