"""nano-1 baselines on Modal.

Trains one char-level baseline per source in 100_simple_voices, fanned out
across multiple T4 GPUs in parallel. Karpathy's train_shakespeare_char
config byte-for-byte (only --dataset and --out_dir overridden per run).

Pricing math:
  T4 = ~$0.59/hr. ~12-15 min per baseline. 100 baselines = ~20-25 GPU-hours.
  Total spend: ~$12-15 (well within the $30/mo free credit).
  With parallelism=10, wall-clock = ~2-2.5 hr instead of ~25 hr sequential.

Usage (from the abcGPT repo root):

    # 1. Smoke-test with one source (~12 min, ~$0.12)
    modal run experiments/nano-1/modal_nano1.py::smoke_test

    # 2. Run all 100 baselines in parallel (~2.5 hr wall-clock, ~$12-15)
    modal run experiments/nano-1/modal_nano1.py::main

    # 3. Download results back to laptop
    modal run experiments/nano-1/modal_nano1.py::download_results
"""

import modal

APP_NAME = "nano-1-baselines"
VOL_NAME = "nano-1"

# Pin the abcGPT commit so the image build is reproducible AND so a
# re-run picks up new commits when we bump this. Bumping the SHA also
# invalidates Modal's image cache and forces a rebuild.
ABCGPT_COMMIT = "54a607d"  # add 100_simple_voices + nano-1 infra

app = modal.App(APP_NAME)

# Volume for results only (corpus + scripts now live in the git clone).
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# Image: torch w/ CUDA, plus the abcGPT repo cloned to a pinned commit.
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
        f"git clone https://github.com/iamtrask/abcGPT.git /opt/abcGPT && "
        f"cd /opt/abcGPT && git checkout {ABCGPT_COMMIT}"
    )
)


def _stream_subprocess(cmd, cwd, log_path, prefix=""):
    """Run a subprocess, streaming its stdout/stderr to BOTH the container's
    stdout (so Modal's dashboard sees it live) AND a local log file. Raises
    CalledProcessError on non-zero exit.
    """
    import subprocess
    with open(log_path, "w") as logf:
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        for line in p.stdout:
            print(f"{prefix}{line}", end="", flush=True)
            logf.write(line)
            logf.flush()
        p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    timeout=90 * 60,  # 90-min cap. Karpathy's 5000-iter recipe needs ~60 min on T4; 50% buffer for noise.
    max_containers=100,  # one wave of all 100 sources in parallel; ~13 min total wall-clock instead of 10 waves × 13 min.
)
def train_one(source_name: str) -> dict:
    """Train one baseline on a single source. Returns summary dict."""
    import os
    import shutil
    import time

    REPO = "/opt/abcGPT"
    src_file = f"{REPO}/data/100_simple_voices/sources/source_{source_name}.txt"
    data_dir = f"{REPO}/data/100_simple_voices/baselines/data/{source_name}"
    out_dir = f"{REPO}/data/100_simple_voices/baselines/out/{source_name}"
    result_dir = f"/vol/results/{source_name}"
    tag = f"[{source_name}] "

    # Idempotency: skip only if we have BOTH the val_loss number AND the model.
    # A prior run that left only the number behind needs to re-train so the
    # model checkpoint gets persisted to the volume too.
    if (
        os.path.exists(f"{result_dir}/best_val_loss.txt")
        and os.path.exists(f"{result_dir}/ckpt.pt")
    ):
        with open(f"{result_dir}/best_val_loss.txt") as f:
            bv = float(f.read().strip())
        print(f"{tag}already trained (val_loss={bv:.4f}, ckpt present), skipping", flush=True)
        return {"source": source_name, "best_val_loss": bv, "status": "skipped (already done)"}

    if not os.path.exists(src_file):
        print(f"{tag}ERROR: missing source file {src_file}", flush=True)
        return {"source": source_name, "status": "missing source file"}

    # 1. Char-level prep
    print(f"{tag}=== STAGE 1: PREP ===", flush=True)
    _stream_subprocess(
        cmd=[
            "python", "-u",
            f"{REPO}/data/100_simple_voices/baselines/prepare_char.py",
            "--source", src_file,
            "--out", data_dir,
        ],
        cwd=REPO,
        log_path=f"/tmp/prep_{source_name}.log",
        prefix=tag,
    )

    # 2. Train. Override only dataset + out_dir; rest is Karpathy's config.
    print(f"{tag}=== STAGE 2: TRAIN (5000 iters of Karpathy's shakespeare_char config) ===", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    _stream_subprocess(
        cmd=[
            "python", "-u", "train.py", "config/train_shakespeare_char.py",
            f"--dataset=100_simple_voices/baselines/data/{source_name}",
            f"--out_dir={out_dir}",
            "--wandb_log=False",
            "--always_save_checkpoint=False",
            "--compile=False",  # faster cold start on T4
        ],
        cwd=REPO,
        log_path=f"{out_dir}/training.log",
        prefix=tag,
    )
    wall_s = time.time() - t0
    print(f"{tag}training done in {wall_s:.1f}s", flush=True)

    # 3. Extract result to volume
    print(f"{tag}=== STAGE 3: EXTRACT ===", flush=True)
    os.makedirs(result_dir, exist_ok=True)
    _stream_subprocess(
        cmd=[
            "python", "-u",
            f"{REPO}/data/100_simple_voices/baselines/extract_result.py",
            "--out-dir", out_dir,
            "--data-dir", data_dir,
            "--result-dir", result_dir,
            "--source-name", source_name,
            "--wall-clock-s", str(wall_s),
        ],
        cwd=REPO,
        log_path=f"/tmp/extract_{source_name}.log",
        prefix=tag,
    )

    # Persist the trained model to the volume so it survives container death.
    # Without this we throw away ~$0.12 of compute every time a container exits.
    ckpt_src = f"{out_dir}/ckpt.pt"
    if os.path.exists(ckpt_src):
        shutil.copy(ckpt_src, f"{result_dir}/ckpt.pt")
        print(f"{tag}saved model checkpoint to volume", flush=True)
    else:
        print(f"{tag}WARNING: no ckpt.pt at {ckpt_src} — val never improved?", flush=True)

    # Persist volume writes
    vol.commit()

    # Read the result for the return value
    with open(f"{result_dir}/best_val_loss.txt") as f:
        bv = float(f.read().strip())
    print(f"{tag}DONE: val_loss={bv:.4f}, wall={wall_s:.1f}s", flush=True)

    return {
        "source": source_name,
        "best_val_loss": bv,
        "wall_s": round(wall_s, 1),
        "status": "trained",
    }


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    timeout=90 * 60,  # same buffer as train_one; learned the hard way on 2026-06-05 when run died at iter 4990/5000.
)
def train_karpathy_shakespeare() -> dict:
    """Karpathy parity check: train his vanilla tinyshakespeare baseline.

    Uses Karpathy's data/shakespeare_char/prepare.py to fetch tinyshakespeare
    and his config/train_shakespeare_char.py unmodified. Target val_loss is
    his published 1.4697. If our number is within ~0.05 nats, our T4 pipeline
    is producing the same result Karpathy did on his A100 — direct evidence
    the rest of nano-1's numbers are Karpathy-comparable.
    """
    import os
    import time
    import torch

    REPO = "/opt/abcGPT"
    out_dir = f"{REPO}/out-shakespeare-char"
    result_dir = "/vol/results/_karpathy-shakespeare-baseline"
    tag = "[karpathy-shake] "

    # Idempotency: skip only if BOTH the val_loss number AND the model are present.
    if (
        os.path.exists(f"{result_dir}/best_val_loss.txt")
        and os.path.exists(f"{result_dir}/ckpt.pt")
    ):
        with open(f"{result_dir}/best_val_loss.txt") as f:
            bv = float(f.read().strip())
        print(f"{tag}already trained (val_loss={bv:.4f}, ckpt present), skipping", flush=True)
        return {
            "source": "karpathy-shakespeare",
            "best_val_loss": bv,
            "karpathy_target": 1.4697,
            "delta": round(bv - 1.4697, 4),
            "wall_s": 0,
            "status": "skipped (already done)",
        }

    # 1. Karpathy's prep (downloads tinyshakespeare from char-rnn)
    print(f"{tag}=== STAGE 1: PREP (downloads tinyshakespeare) ===", flush=True)
    _stream_subprocess(
        cmd=["python", "-u", "data/shakespeare_char/prepare.py"],
        cwd=REPO,
        log_path="/tmp/karpathy_prep.log",
        prefix=tag,
    )

    # 2. Karpathy's exact training command (NO overrides — vanilla config)
    print(f"{tag}=== STAGE 2: TRAIN (5000 iters of Karpathy's exact config) ===", flush=True)
    t0 = time.time()
    _stream_subprocess(
        cmd=[
            "python", "-u", "train.py", "config/train_shakespeare_char.py",
            "--wandb_log=False",
            "--compile=False",  # faster cold start on T4
        ],
        cwd=REPO,
        log_path=f"{out_dir}/training.log" if os.path.isdir(out_dir) else "/tmp/karpathy_train.log",
        prefix=tag,
    )
    wall_s = time.time() - t0

    # 3. Read best_val_loss from checkpoint
    ckpt = torch.load(f"{out_dir}/ckpt.pt", map_location="cpu", weights_only=False)
    best_val = float(ckpt.get("best_val_loss", float("inf")))

    # 4. Save to volume
    import shutil
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/best_val_loss.txt", "w") as f:
        f.write(f"{best_val:.6f}\n")
    # Move training.log to volume so it's downloadable
    if os.path.exists(f"{out_dir}/training.log"):
        shutil.copy(f"{out_dir}/training.log", f"{result_dir}/training.log")
    # Persist the trained model so we can do sampling / comparison work later
    ckpt_path = f"{out_dir}/ckpt.pt"
    if os.path.exists(ckpt_path):
        shutil.copy(ckpt_path, f"{result_dir}/ckpt.pt")
        print(f"{tag}saved model checkpoint to volume", flush=True)
    else:
        print(f"{tag}WARNING: no ckpt.pt at {ckpt_path}", flush=True)
    vol.commit()

    delta = best_val - 1.4697
    print(f"{tag}DONE: val_loss={best_val:.4f}, delta-from-Karpathy={delta:+.4f}, wall={wall_s:.1f}s", flush=True)

    return {
        "source": "karpathy-shakespeare",
        "best_val_loss": best_val,
        "karpathy_target": 1.4697,
        "delta": round(delta, 4),
        "wall_s": round(wall_s, 1),
        "status": "trained",
    }


def _list_local_sources() -> list[str]:
    """Discover 100 source names from local repo. Robust to cwd."""
    import glob
    import os
    # Resolve relative to this file so cwd doesn't matter.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    pattern = os.path.join(repo_root, "data/100_simple_voices/sources/source_*.txt")
    files = sorted(glob.glob(pattern))
    return [os.path.basename(f).replace("source_", "").replace(".txt", "") for f in files]


@app.local_entrypoint()
def smoke_test():
    """Train one source as a smoke test before the big fan-out."""
    src = "077_nba-play-by-play"
    print(f"Smoke test: training {src} on Modal T4 ...")
    result = train_one.remote(src)
    print(f"Result: {result}")


def _print_karpathy_verdict(result: dict) -> None:
    """Pretty-print Karpathy parity verdict. Shared by karpathy_baseline and main."""
    print()
    print("=" * 64)
    print(f"  KARPATHY-PARITY RESULT")
    print(f"  ----")
    print(f"  our val_loss:      {result['best_val_loss']:.4f}")
    print(f"  Karpathy target:   {result['karpathy_target']:.4f}")
    print(f"  delta:             {result['delta']:+.4f} nats")
    print(f"  wall_clock:        {result.get('wall_s', 0):.1f}s")
    if abs(result['delta']) < 0.05:
        print(f"  -> MATCH (within 0.05 nats)")
    elif abs(result['delta']) < 0.15:
        print(f"  -> CLOSE (within 0.15 nats — likely seed variance)")
    else:
        print(f"  -> MISMATCH (>0.15 nats) — investigate before trusting nano-1 numbers")
    print("=" * 64)


@app.local_entrypoint()
def karpathy_baseline():
    """Karpathy parity check: train his exact tinyshakespeare baseline.

    Validates our T4 pipeline produces the same val_loss Karpathy did on A100
    (his published 1.4697). If our number is within ~0.05 nats, the rest of
    nano-1's per-source numbers are directly comparable to Karpathy-style
    publications.
    """
    print("Karpathy parity check: training tinyshakespeare baseline on Modal T4 ...")
    result = train_karpathy_shakespeare.remote()
    _print_karpathy_verdict(result)


@app.local_entrypoint()
def main(only: str = "", skip_karpathy: bool = False):
    """Fan out training across all 100 sources + the Karpathy parity baseline.

    Karpathy's tinyshakespeare baseline is kicked off in parallel with the
    fan-out so it doesn't extend wall-clock (it finishes well before the 100
    sources do). Pass --skip-karpathy if you only want the per-source models.

    Parallelism is controlled by `max_containers=10` on the train_one decorator
    (Modal 1.4.x moved this off of .map() onto the function definition).

    Usage:
        modal run modal_nano1.py::main                  # all 100 + karpathy
        modal run modal_nano1.py::main --only 077_nba-play-by-play,076_retrosheet-baseball
        modal run modal_nano1.py::main --skip-karpathy  # just the per-source baselines
    """
    if only:
        sources = [s.strip() for s in only.split(",") if s.strip()]
    else:
        sources = _list_local_sources()

    print(f"Training {len(sources)} per-source baselines"
          f"{'' if skip_karpathy else ' + Karpathy parity'} (max 100 parallel containers)")
    print()

    # Kick off Karpathy parity check in parallel with the per-source fan-out.
    # It uses one extra container slot (separate from train_one's max_containers
    # budget) and finishes in ~12 min — well before the 100-source fan-out does — so
    # it adds nothing to critical-path wall-clock.
    karpathy_handle = None
    if not skip_karpathy:
        print("Spawning Karpathy parity check (parallel with per-source fan-out)...")
        karpathy_handle = train_karpathy_shakespeare.spawn()
        print()

    n_trained = n_skipped = n_failed = 0
    for r in train_one.map(sources):
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
    print(f"Per-source summary: {n_trained} trained, {n_skipped} skipped, {n_failed} failed")
    print(f"Results saved to Modal volume '{VOL_NAME}' under /results/")

    # Now collect the Karpathy verdict (almost always already done by this point).
    if karpathy_handle is not None:
        print()
        print("Collecting Karpathy parity result ...")
        try:
            karpathy_result = karpathy_handle.get()
            _print_karpathy_verdict(karpathy_result)
        except Exception as exc:
            print(f"WARNING: Karpathy parity check failed: {exc}")
            print(f"  (per-source baselines are unaffected; rerun karpathy_baseline to retry)")

    print()
    print(f"Run 'modal run experiments/nano-1/modal_nano1.py::download_results' to fetch.")


@app.local_entrypoint()
def download_results():
    """Download /results/ from the Modal volume back to local laptop."""
    import subprocess
    local_dir = "experiments/nano-1/results/"
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
        ["python3", "data/100_simple_voices/baselines/aggregate.py"],
        check=True,
    )
