"""nano-2: GatedGPT on Shakespeare+TinyStories, with rich per-iter / per-eval logs.

Wraps gated_gpt_tent.{GatedGPT, GatedGPTConfig} with:
  - per-iter JSONL log: iter, alpha, cohort, train_loss, lr
  - per-eval JSONL log: full val-loss matrix (α ∈ {0, 0.5, 1} × {shake, ts}),
                       m_n distribution snapshot (per-layer shake/halfsies/ts counts),
                       m_n stats (mean, std), m_n drift from init
  - --variant {ungated, fixed-mn, trainable-mn}   (trainable-mn is a stub for now;
                                                    enabled in step 2)
  - --variant-name <str>    (output subdir under results/)
  - Standard hyperparams from the original notebook + new sweepable knobs:
      --span (tent narrowness), --alpha-dist {beta_half, uniform},
      --n-iters, --lr, --warmup, --eval-interval, --log-interval, --batch-size
  - --smoke   (tiny config to validate on laptop CPU in ~30s)

Run locally to validate:
    python experiments/nano-2/train.py --variant fixed-mn --variant-name fixed-mn-smoke --smoke

Run a real variant on a 4090:
    python experiments/nano-2/train.py --variant fixed-mn --variant-name fixed-mn-replicate \\
        --n-iters 10000 --device cuda
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable so we can pick up gated_gpt_tent.py
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gated_gpt_tent import GatedGPT, GatedGPTConfig, sample_beta_half_mask  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading: shake+TS combined corpus, split via '\n\n===\n\n' separator
# ---------------------------------------------------------------------------
SEP_STR = '\n\n===\n\n'


def build_batchers(data_dir, block_size, batch_size, device, val_frac=0.1):
    """Returns (get_shake_train, get_ts_train, get_shake_val, get_ts_val, vocab_size).

    Matches the notebook's split strategy: use train.bin only (val.bin doesn't
    contain Shakespeare data — the prep does a 90/10 split on the concatenated
    string AFTER concat, leaving val.bin as the tail-end of TinyStories only).
    Instead we use train.bin and carve out the last `val_frac` of EACH cohort's
    range as that cohort's validation set.
    """
    arr = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
    with open(data_dir / 'meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']

    sep_tokens = np.array([stoi[c] for c in SEP_STR], dtype=np.uint16)
    candidates = np.where(arr[:len(arr) - len(sep_tokens) + 1] == sep_tokens[0])[0]
    sep_pos = None
    for c in candidates:
        if np.array_equal(arr[c:c + len(sep_tokens)], sep_tokens):
            sep_pos = int(c)
            break
    if sep_pos is None:
        raise RuntimeError(f"separator {SEP_STR!r} not found in train.bin")
    shake_lo, shake_hi = 0, sep_pos
    ts_lo, ts_hi = sep_pos + len(sep_tokens), len(arr)

    # Per-cohort 90/10 split: val is the tail of each range
    shake_val_size = int((shake_hi - shake_lo) * val_frac)
    ts_val_size = int((ts_hi - ts_lo) * val_frac)
    shake_train_hi = shake_hi - shake_val_size
    shake_val_lo = shake_train_hi
    ts_train_hi = ts_hi - ts_val_size
    ts_val_lo = ts_train_hi

    def make(lo, hi):
        def get():
            ix = torch.randint(hi - lo - block_size, (batch_size,)) + lo
            x = torch.stack([torch.from_numpy(arr[i:i + block_size].astype(np.int64)) for i in ix.tolist()])
            y = torch.stack([torch.from_numpy(arr[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix.tolist()])
            return x.to(device), y.to(device)
        return get

    print(f"  shake train: [{shake_lo:,}, {shake_train_hi:,})  val: [{shake_val_lo:,}, {shake_hi:,})  "
          f"({shake_train_hi - shake_lo:,} train, {shake_hi - shake_val_lo:,} val tokens)")
    print(f"  ts train:    [{ts_lo:,}, {ts_train_hi:,})  val: [{ts_val_lo:,}, {ts_hi:,})  "
          f"({ts_train_hi - ts_lo:,} train, {ts_hi - ts_val_lo:,} val tokens)")

    return (
        make(shake_lo, shake_train_hi),
        make(ts_lo, ts_train_hi),
        make(shake_val_lo, shake_hi),
        make(ts_val_lo, ts_hi),
        meta['vocab_size'],
    )


# ---------------------------------------------------------------------------
# Rich logging: collect m_n distribution snapshots + write JSONL
# ---------------------------------------------------------------------------
def snapshot_m_n_distribution(model, m_n_init_snapshot=None):
    """Return per-layer m_n bucket counts + global stats."""
    BUCKETS = [
        ('ts_spec', 0.0, 0.3),       # m_n < 0.3 → TS-specialized (since α=0 → TS)
        ('halfsies', 0.3, 0.7),      # 0.3 ≤ m_n ≤ 0.7 → shared
        ('shake_spec', 0.7, 1.0001),  # m_n > 0.7 → Shake-specialized
    ]

    def bucket_counts(M):
        counts = {}
        for name, lo, hi in BUCKETS:
            counts[name] = int(((M >= lo) & (M < hi)).sum())
        return counts

    # Use the model's accessor methods so this works for both fixed (buffer)
    # and trainable (sigmoid(logits)) modes uniformly.
    with torch.no_grad():
        M_embd = model._M_embd()
        layers = {'M_embd': {
            'bucket_counts': bucket_counts(M_embd),
            'mean': float(M_embd.mean()),
            'std': float(M_embd.std()),
            'n_total': int(M_embd.numel()),
        }}
        for i, block in enumerate(model.transformer.h):
            M_head = block.attn._M_head()
            M_inner = block.mlp._M_inner()
            layers[f'layer_{i}_attn_heads'] = {
                'bucket_counts': bucket_counts(M_head),
                'mean': float(M_head.mean()),
                'std': float(M_head.std()),
                'n_total': int(M_head.numel()),
            }
            layers[f'layer_{i}_mlp_inner'] = {
                'bucket_counts': bucket_counts(M_inner),
                'mean': float(M_inner.mean()),
                'std': float(M_inner.std()),
                'n_total': int(M_inner.numel()),
            }
        snap = {'layers': layers}

        if m_n_init_snapshot is not None:
            init_M_embd = torch.tensor(m_n_init_snapshot['M_embd'], device=M_embd.device)
            snap['M_embd_drift_l2'] = float(((M_embd - init_M_embd) ** 2).sum().sqrt())
    return snap


def capture_m_n_init(model):
    with torch.no_grad():
        return {'M_embd': model._M_embd().cpu().tolist()}


# ---------------------------------------------------------------------------
# Training loop (adapted from gated_gpt_tent.train_gated with logger hooks)
# ---------------------------------------------------------------------------
def train_with_logging(
    model,
    get_shake_batch,
    get_ts_batch,
    n_iters,
    log_path,
    lr=1e-3,
    warmup=100,
    lr_decay_iters=None,
    min_lr=1e-4,
    beta2=0.99,
    weight_decay=0.1,
    grad_clip=1.0,
    alpha_dist='beta_half',
    log_interval=100,
    eval_interval=250,
    eval_iters=200,
    get_shake_val=None,
    get_ts_val=None,
    device='cuda',
    amp_dtype=torch.bfloat16,
    ungated=False,
    # trainable-mn knobs (ignored unless model.trainable_masks=True)
    mask_lr_ratio=1e-2,
    lambda_var=0.1,
    var_gamma=0.25,
    warmup_mn=0,
    balance_penalty=False,
    lambda_balance=1.0,
    boundary_migration=False,
    migration_interval=100,
    migration_step=0.01,
    migration_threshold=0.02,
    migration_warmup=500,
    ema_alpha=0.99,
    migration_step_decay=False,
    migration_boundary_cap=0.5,
):
    """Same logic as gated_gpt_tent.train_gated, but writes a structured JSONL log.

    Each line of log_path is one JSON object with a 'type' field. Types:
      'config'    — initial config snapshot (also captures m_n_init)
      'iter'      — per-iter training step (iter, alpha, cohort, loss, lr)
      'eval'      — per-eval-interval val_loss matrix + m_n distribution snapshot
      'done'      — final summary
    """
    if lr_decay_iters is None:
        lr_decay_iters = n_iters

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, 'w')

    def emit(record):
        log_fp.write(json.dumps(record) + '\n')
        log_fp.flush()

    # Capture m_n init snapshot for drift tracking
    m_n_init = capture_m_n_init(model) if not ungated else None

    emit({
        'type': 'config',
        'n_iters': n_iters,
        'lr': lr,
        'warmup': warmup,
        'lr_decay_iters': lr_decay_iters,
        'min_lr': min_lr,
        'alpha_dist': alpha_dist,
        'eval_interval': eval_interval,
        'log_interval': log_interval,
        'eval_iters': eval_iters,
        'tent_narrowness': model.config.tent_narrowness if hasattr(model.config, 'tent_narrowness') else None,
        'n_layer': model.config.n_layer,
        'n_head': model.config.n_head,
        'n_embd': model.config.n_embd,
        'block_size': model.config.block_size,
        'ungated': ungated,
        't_start': time.time(),
        'm_n_init_snapshot': (snapshot_m_n_distribution(model) if not ungated else None),
    })

    # Identify mask logit params separately so they get a different LR + no weight decay.
    trainable_masks = (not ungated) and getattr(model, 'trainable_masks', False)
    mask_logit_param_ids = set()
    if trainable_masks:
        for p in model.mask_logit_params():
            mask_logit_param_ids.add(id(p))

    decay_params, nodecay_params, mask_params = [], [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in mask_logit_param_ids:
            mask_params.append(p)
        elif p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr, 'name': 'weights_decay'},
        {'params': nodecay_params, 'weight_decay': 0.0, 'lr': lr, 'name': 'weights_nodecay'},
    ]
    if mask_params:
        param_groups.append({
            'params': mask_params, 'weight_decay': 0.0,
            'lr': lr * mask_lr_ratio, 'name': 'mask_logits',
        })
    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, beta2), fused=(device == 'cuda'),
    )

    def get_lr(it):
        if it < warmup:
            return lr * (it + 1) / warmup
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup) / (lr_decay_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    def apply_lr(it):
        """Set the LR for all param groups. Mask-logit group gets lr * mask_lr_ratio."""
        cur = get_lr(it)
        for pg in optimizer.param_groups:
            if pg.get('name') == 'mask_logits':
                pg['lr'] = cur * mask_lr_ratio
            else:
                pg['lr'] = cur
        return cur

    def sample_alpha():
        if alpha_dist == 'beta_half':
            u = torch.rand(1).item()
            return math.sin(math.pi / 2 * u) ** 2
        elif alpha_dist == 'uniform':
            return torch.rand(1).item()
        elif alpha_dist == 'bimodal_endpoints':
            # 80% at endpoints, 20% uniform in middle
            r = torch.rand(1).item()
            if r < 0.4:
                return 0.0
            elif r < 0.8:
                return 1.0
            else:
                return torch.rand(1).item()
        else:
            raise ValueError(f'unknown alpha_dist: {alpha_dist}')

    @torch.no_grad()
    def estimate_val(get_batch_fn, alpha):
        if get_batch_fn is None:
            return None
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn()
            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                if ungated:
                    _, loss = model(X, Y)
                else:
                    _, loss = model(X, alpha, Y, corpus=None)
            losses[k] = loss.item()
        model.train()
        return float(losses.mean())

    n_shake = 0
    n_ts = 0
    # Per-cohort loss EMAs for boundary-migration mode
    loss_ema_shake = None  # set on first shake batch
    loss_ema_ts = None
    n_migrations = 0       # diagnostic counter
    model.train()
    t_start = time.time()
    t_log = t_start

    for it in range(n_iters):
        cur_lr = apply_lr(it)

        balance_pen = None  # populated only in balance-penalty mode
        var_pen = None      # populated only in trainable-mn mode

        if balance_penalty:
            # Dual-batch step: train on shake AT alpha=1 AND ts AT alpha=0 in
            # the same backward, with an extra term penalizing |loss_shake - loss_ts|.
            # This puts gradient pressure DIRECTLY on the imbalance — the optimizer
            # can satisfy it by adapting weights, m_n, or both.
            X_sh, Y_sh = get_shake_batch()
            X_ts, Y_ts = get_ts_batch()
            n_shake += 1
            n_ts += 1
            cohort = 'both'
            alpha = -1  # sentinel — we used two different alphas
            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                _, loss_sh = model(X_sh, 1.0, Y_sh)
                _, loss_ts = model(X_ts, 0.0, Y_ts)
                ce_loss = 0.5 * (loss_sh + loss_ts)
                balance_pen = (loss_sh - loss_ts).abs()
                loss = ce_loss + lambda_balance * balance_pen
        else:
            alpha = sample_alpha()
            use_shake = torch.rand(1).item() < alpha
            if use_shake:
                X, Y = get_shake_batch()
                n_shake += 1
                cohort = 'shake'
            else:
                X, Y = get_ts_batch()
                n_ts += 1
                cohort = 'ts'

            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                if ungated:
                    _, ce_loss = model(X, Y)
                else:
                    _, ce_loss = model(X, alpha, Y)
                # Variance regularizer (trainable_masks=True only; zero otherwise)
                if trainable_masks and lambda_var > 0:
                    var_pen = model.mask_variance_regularizer(gamma=var_gamma)
                    loss = ce_loss + lambda_var * var_pen
                else:
                    loss = ce_loss
                    var_pen = None
        loss.backward()
        # Warmup: zero mask gradients for first warmup_mn iters so weights find
        # their initial specialization before m_n starts drifting.
        if trainable_masks and warmup_mn > 0 and it < warmup_mn:
            for p in mask_params:
                if p.grad is not None:
                    p.grad.zero_()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ------------------------------------------------------------------
        # Boundary-migration: control-loop step (outside gradient flow).
        # Updates per-cohort training-loss EMAs; periodically nudges the
        # boundary toward giving capacity to the higher-loss cohort.
        # ------------------------------------------------------------------
        if boundary_migration and not balance_penalty:
            train_loss_val = float(ce_loss.item())
            if cohort == 'shake':
                loss_ema_shake = train_loss_val if loss_ema_shake is None else \
                                 ema_alpha * loss_ema_shake + (1 - ema_alpha) * train_loss_val
            elif cohort == 'ts':
                loss_ema_ts = train_loss_val if loss_ema_ts is None else \
                              ema_alpha * loss_ema_ts + (1 - ema_alpha) * train_loss_val

            if (it + 1) % migration_interval == 0 and (it + 1) > migration_warmup \
               and loss_ema_shake is not None and loss_ema_ts is not None:
                delta = loss_ema_shake - loss_ema_ts
                if abs(delta) > migration_threshold:
                    # Effective step: linearly decay from migration_step → 0 if --migration-step-decay
                    if migration_step_decay:
                        progress = (it + 1) / n_iters  # 0 → 1
                        cur_step = migration_step * max(0.0, 1.0 - progress)
                    else:
                        cur_step = migration_step
                    # Direction: shake harder → boundary DOWN, ts harder → boundary UP
                    shift_sign = -1.0 if delta > 0 else +1.0
                    proposed_shift = shift_sign * cur_step
                    # Cap check: don't move past abs(boundary - 0.5) > migration_boundary_cap
                    cur_boundary = float(model.M_embd_boundary.item())
                    new_boundary = cur_boundary + proposed_shift
                    if abs(new_boundary - 0.5) > migration_boundary_cap:
                        # Clip to the cap rather than reject entirely
                        if new_boundary > 0.5:
                            new_boundary = 0.5 + migration_boundary_cap
                        else:
                            new_boundary = 0.5 - migration_boundary_cap
                        proposed_shift = new_boundary - cur_boundary
                    if abs(proposed_shift) > 1e-7:
                        model.shift_all_boundaries(proposed_shift)
                        n_migrations += 1

        if (it + 1) % log_interval == 0:
            dt = time.time() - t_log
            train_loss = float(ce_loss.item())
            rec = {
                'type': 'iter',
                'iter': it + 1,
                'alpha': alpha,
                'cohort': cohort,
                'train_loss': train_loss,
                'lr': cur_lr,
                'dt_s': dt,
                't_elapsed_s': time.time() - t_start,
                'n_shake': n_shake,
                'n_ts': n_ts,
            }
            if var_pen is not None:
                rec['var_penalty'] = float(var_pen.item())
            if balance_penalty and balance_pen is not None:
                rec['balance_penalty'] = float(balance_pen.item())
                rec['loss_shake'] = float(loss_sh.item())
                rec['loss_ts'] = float(loss_ts.item())
            emit(rec)
            extras = ''
            if var_pen is not None: extras += f" var_pen={var_pen.item():.4f}"
            if balance_pen is not None:
                extras += f" bal_pen={balance_pen.item():.4f} sh={loss_sh.item():.3f} ts={loss_ts.item():.3f}"
            print(f"iter {it+1:5d} | alpha {alpha:>5} | {cohort:>5s} | "
                  f"loss {train_loss:.4f} | lr {cur_lr:.6f} | dt {dt:.1f}s{extras}", flush=True)
            t_log = time.time()

        if (it + 1) % eval_interval == 0:
            v_sh_1 = estimate_val(get_shake_val, alpha=1.0)
            v_sh_5 = estimate_val(get_shake_val, alpha=0.5)
            v_sh_0 = estimate_val(get_shake_val, alpha=0.0)
            v_ts_1 = estimate_val(get_ts_val, alpha=1.0)
            v_ts_5 = estimate_val(get_ts_val, alpha=0.5)
            v_ts_0 = estimate_val(get_ts_val, alpha=0.0)
            mn_snap = snapshot_m_n_distribution(model, m_n_init_snapshot=m_n_init) if not ungated else None
            eval_rec = {
                'type': 'eval',
                'iter': it + 1,
                't_elapsed_s': time.time() - t_start,
                'val_loss_matrix': {
                    'shake@a=0.0': v_sh_0, 'shake@a=0.5': v_sh_5, 'shake@a=1.0': v_sh_1,
                    'ts@a=0.0':    v_ts_0, 'ts@a=0.5':    v_ts_5, 'ts@a=1.0':    v_ts_1,
                },
                'm_n_snapshot': mn_snap,
                'n_shake': n_shake,
                'n_ts': n_ts,
            }
            if boundary_migration:
                eval_rec['boundaries'] = model.get_boundaries()
                eval_rec['loss_ema_shake'] = loss_ema_shake
                eval_rec['loss_ema_ts'] = loss_ema_ts
                eval_rec['n_migrations'] = n_migrations
            emit(eval_rec)
            print(f"  >>> step {it+1}: shake@a=1.0={v_sh_1:.3f}  ts@a=0.0={v_ts_0:.3f}  "
                  f"shake@a=0.5={v_sh_5:.3f}  ts@a=0.5={v_ts_5:.3f}  "
                  f"corpus split so far: {n_shake} shake / {n_ts} ts", flush=True)

    total_s = time.time() - t_start
    emit({
        'type': 'done',
        'iter': n_iters,
        'total_s': total_s,
        'n_shake': n_shake,
        'n_ts': n_ts,
    })
    print(f"total training time: {total_s:.1f}s  final split: {n_shake} shake / {n_ts} ts", flush=True)
    log_fp.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant',
                   choices=['ungated', 'fixed-mn', 'trainable-mn', 'balance-penalty', 'boundary-migration'],
                   required=True,
                   help='ungated = vanilla GPT; fixed-mn = original nano-2; trainable-mn = sigmoid-logits masks; '
                        'balance-penalty = fixed-mn + dual-batch + |loss_shake - loss_ts| penalty; '
                        'boundary-migration = each neuron has a permanent rank, single boundary scalar moves '
                        'based on per-cohort loss EMA (control-system, no gradient on masks)')
    p.add_argument('--variant-name', required=True,
                   help='subdir name under experiments/nano-2/results/<variant-name>/')
    p.add_argument('--data-dir', default=str(REPO_ROOT / 'data/shakespeare_tinystories_char'))
    p.add_argument('--results-root', default=str(REPO_ROOT / 'experiments/nano-2/results'))
    # Model arch (Karpathy shakespeare_char spec by default)
    p.add_argument('--n-layer', type=int, default=6)
    p.add_argument('--n-head', type=int, default=6)
    p.add_argument('--n-embd', type=int, default=384)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--span', type=float, default=1.0, help='tent_narrowness for the gate (1.0 = full width)')
    p.add_argument('--mask-seed', type=int, default=1337)
    # Training
    p.add_argument('--n-iters', type=int, default=10000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--min-lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--beta2', type=float, default=0.99)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--alpha-dist', default='beta_half', choices=['beta_half', 'uniform', 'bimodal_endpoints'])
    # Match the notebook defaults exactly: log every 250 iters, eval every 500.
    p.add_argument('--log-interval', type=int, default=250)
    p.add_argument('--eval-interval', type=int, default=500)
    p.add_argument('--eval-iters', type=int, default=200)
    p.add_argument('--seed', type=int, default=1337,
                   help='torch.manual_seed value, set before model init. Default 1337 matches the original nano-2 notebook.')
    # Trainable-m_n params (stubs for step 2)
    p.add_argument('--mask-lr-ratio', type=float, default=1e-2,
                   help='(trainable-mn only) m_n LR = lr * mask_lr_ratio. 1e-2 = "moderate"')
    p.add_argument('--lambda-var', type=float, default=0.1,
                   help='(trainable-mn only) variance-floor regularizer strength')
    p.add_argument('--warmup-mn', type=int, default=0,
                   help='(trainable-mn only) freeze m_n for first N iters')
    p.add_argument('--lambda-balance', type=float, default=1.0,
                   help='(balance-penalty only) weight on the |loss_shake - loss_ts| term in the combined loss')
    # boundary-migration knobs
    p.add_argument('--boundary-sharpness', type=float, default=10.0,
                   help='(boundary-migration) sharpness of sigmoid transition from rank-space to m_n. higher = sharper')
    p.add_argument('--migration-interval', type=int, default=100,
                   help='(boundary-migration) check imbalance + maybe shift boundary every N iters')
    p.add_argument('--migration-step', type=float, default=0.01,
                   help='(boundary-migration) boundary shift magnitude per migration event')
    p.add_argument('--migration-threshold', type=float, default=0.02,
                   help='(boundary-migration) only shift if |ema_shake - ema_ts| > this (nats)')
    p.add_argument('--migration-warmup', type=int, default=500,
                   help='(boundary-migration) skip migration for the first N iters (let weights settle)')
    p.add_argument('--ema-alpha', type=float, default=0.99,
                   help='(boundary-migration) exponential moving average factor for per-cohort loss tracking')
    # Damping knobs added after the first boundary-migration run overshot
    # (boundary went from 0.5 → 0.065, TS collapsed). These stop runaway migration.
    p.add_argument('--migration-step-decay', action='store_true',
                   help='(boundary-migration) linearly decay migration step from --migration-step to 0 over n_iters')
    p.add_argument('--migration-boundary-cap', type=float, default=0.5,
                   help='(boundary-migration) max |boundary - 0.5| allowed (default 0.5 = no cap). e.g. 0.15 = boundary stays in [0.35, 0.65]')
    p.add_argument('--initial-boundary', type=float, default=0.5,
                   help='(boundary-migration) starting value for the boundary scalar. '
                        'Use 0.5 for the default (no shift). Use e.g. 0.35 to start with the '
                        'post-migration allocation baked in (phase-2 retrain protocol). '
                        'Combine with --migration-step 0 to freeze the boundary for the whole run.')
    # Device + smoke
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp-dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    p.add_argument('--smoke', action='store_true', help='tiny config to validate driver works in <1 min on CPU')
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
        if args.device == 'cuda' and not torch.cuda.is_available():
            args.device = 'cpu'

    # trainable-mn now supported (step 2 done).

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"error: data dir missing: {data_dir}")

    # Seed the global RNG to match the original notebook's reproducibility.
    # The notebook calls torch.manual_seed(1337) before model init; this affects
    # weight init AND every torch.rand/randint call during training (alpha
    # sampling, batch index sampling, etc.). Without this, our runs diverge
    # from the notebook by ~0.10-0.15 nats just from seed variance.
    torch.manual_seed(args.seed)

    out_dir = Path(args.results_root) / args.variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'log.jsonl'
    print(f"variant: {args.variant}  | name: {args.variant_name}  | log: {log_path}  | seed: {args.seed}")

    amp_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[args.amp_dtype]

    sh_train, ts_train, sh_val, ts_val, vocab_size = build_batchers(
        data_dir, args.block_size, args.batch_size, args.device,
    )
    print(f"data: vocab_size={vocab_size}, block_size={args.block_size}, batch_size={args.batch_size}, device={args.device}")

    # Build model
    if args.variant == 'ungated':
        # Vanilla nanoGPT — load Karpathy's GPT from the abcGPT repo root
        from model import GPT, GPTConfig
        cfg = GPTConfig(
            vocab_size=vocab_size, n_layer=args.n_layer, n_head=args.n_head,
            n_embd=args.n_embd, block_size=args.block_size, dropout=args.dropout, bias=False,
        )
        model = GPT(cfg).to(args.device)
        ungated = True
    else:
        # fixed-mn / trainable-mn / balance-penalty all use GatedGPT.
        # balance-penalty uses FIXED masks by default — the imbalance penalty
        # shapes the weights (and m_n if trainable_masks is also True).
        cfg = GatedGPTConfig(
            vocab_size=vocab_size, n_layer=args.n_layer, n_head=args.n_head,
            n_embd=args.n_embd, block_size=args.block_size, dropout=args.dropout,
            mask_seed=args.mask_seed, tent_narrowness=args.span,
            trainable_masks=(args.variant == 'trainable-mn'),
            boundary_mode=(args.variant == 'boundary-migration'),
            boundary_sharpness=args.boundary_sharpness,
        )
        model = GatedGPT(cfg).to(args.device)
        ungated = False

        if args.variant == 'boundary-migration' and abs(args.initial_boundary - 0.5) > 1e-7:
            # Phase-2 protocol: bake a post-migration boundary into iter-0 init,
            # so the model trains from scratch with the target allocation already
            # in place. No unlearn-then-relearn transient.
            delta = args.initial_boundary - 0.5
            model.shift_all_boundaries(delta)
            print(f"  initial boundary set to {args.initial_boundary:.4f} (shifted by {delta:+.4f})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params  ({args.variant})")

    train_with_logging(
        model,
        get_shake_batch=sh_train, get_ts_batch=ts_train,
        n_iters=args.n_iters,
        log_path=log_path,
        lr=args.lr, warmup=args.warmup, min_lr=args.min_lr,
        beta2=args.beta2, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        alpha_dist=args.alpha_dist,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        get_shake_val=sh_val, get_ts_val=ts_val,
        device=args.device, amp_dtype=amp_dtype,
        ungated=ungated,
        mask_lr_ratio=args.mask_lr_ratio,
        lambda_var=args.lambda_var,
        warmup_mn=args.warmup_mn,
        balance_penalty=(args.variant == 'balance-penalty'),
        lambda_balance=args.lambda_balance,
        boundary_migration=(args.variant == 'boundary-migration'),
        migration_interval=args.migration_interval,
        migration_step=args.migration_step,
        migration_threshold=args.migration_threshold,
        migration_warmup=args.migration_warmup,
        ema_alpha=args.ema_alpha,
        migration_step_decay=args.migration_step_decay,
        migration_boundary_cap=args.migration_boundary_cap,
    )

    # Save final summary + final m_n snapshot if applicable
    summary = {'variant': args.variant, 'variant_name': args.variant_name, 'n_params': n_params}
    if not ungated:
        summary['final_m_n_snapshot'] = snapshot_m_n_distribution(model)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"DONE. results -> {out_dir}")


if __name__ == '__main__':
    main()
