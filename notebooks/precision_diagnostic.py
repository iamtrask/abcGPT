# === abcGPT precision diagnostic — paste into one Colab cell (or run locally) ===
#
# Captures every environment knob that could plausibly affect numerics, plus
# deterministic init + forward checks so we can compare bit-for-bit against
# the RunPod runs. Paste the OUTPUT back so we can diff.

import subprocess
subprocess.run(["pip", "install", "-q", "torch", "numpy"], check=True)
subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/iamtrask/abcGPT.git"], check=False)

import sys, os
sys.path.insert(0, "abcGPT")

import hashlib, platform
import torch
import numpy as np
import gated_gpt_tent as ggt

# ── 1. ENVIRONMENT ─────────────────────────────────────────────────────
print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print(f"Python:           {sys.version.split()[0]}")
print(f"Platform:         {platform.platform()}")
print(f"PyTorch:          {torch.__version__}")
print(f"NumPy:            {np.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:     {torch.version.cuda}")
    print(f"cuDNN version:    {torch.backends.cudnn.version()}")
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute cap:      sm_{cap[0]}{cap[1]}")
    print(f"bf16 supported:   {torch.cuda.is_bf16_supported()}")

# ── 2. NUMERICS / KERNEL DEFAULTS ─────────────────────────────────────
print()
print("=" * 70)
print("NUMERICS / KERNEL DEFAULTS")
print("=" * 70)
print(f"TF32 matmul allowed:     {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cudnn allowed:      {torch.backends.cudnn.allow_tf32}")
print(f"cuDNN deterministic:     {torch.backends.cudnn.deterministic}")
print(f"cuDNN benchmark:         {torch.backends.cudnn.benchmark}")
print(f"matmul precision:        {torch.get_float32_matmul_precision()}")
print(f"default dtype:           {torch.get_default_dtype()}")
print(f"OMP_NUM_THREADS env:     {os.environ.get('OMP_NUM_THREADS', '<unset>')}")
print(f"torch num_threads:       {torch.get_num_threads()}")

# ── 3. MODEL CONSTRUCTION (seeded) ────────────────────────────────────
print()
print("=" * 70)
print("MODEL CONSTRUCTION — same config as nano-2 fixed-mn")
print("=" * 70)
torch.manual_seed(1337)
np.random.seed(1337)
cfg = ggt.GatedGPTConfig(
    vocab_size=75, n_layer=6, n_head=6, n_embd=384,
    block_size=256, dropout=0.2, bias=False,
    mask_seed=1337, tent_narrowness=1.0,
)
m = ggt.GatedGPT(cfg)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = m.to(device)

# Param SHA1 — same value across platforms = bit-identical init
h = hashlib.sha1()
for name, p in sorted(m.named_parameters()):
    h.update(name.encode())
    h.update(p.detach().cpu().to(torch.float32).contiguous().numpy().tobytes())
for name, b in sorted(m.named_buffers()):
    h.update(name.encode())
    h.update(b.detach().cpu().to(torch.float32).contiguous().numpy().tobytes())
print(f"model param SHA1 (first 16 hex): {h.hexdigest()[:16]}")
print(f"total params: {sum(p.numel() for p in m.parameters()) / 1e6:.4f}M")

# Sample a few specific param values for direct comparison
sample_param = next(p for n, p in m.named_parameters() if 'wte' in n)
print(f"wte.weight[0, :5]: {sample_param[0, :5].tolist()}")

# ── 4. DETERMINISTIC FORWARD + BACKWARD (fp32) ────────────────────────
print()
print("=" * 70)
print("DETERMINISTIC FORWARD+BACKWARD (no autocast, fp32)")
print("=" * 70)
torch.manual_seed(1337)
X = torch.randint(0, 75, (4, 32), device=device)
Y = torch.randint(0, 75, (4, 32), device=device)
m.train()
_, loss_fp32 = m(X, alpha=0.5, targets=Y)
loss_fp32.backward()
grad_norm_fp32 = sum(p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
print(f"fp32 forward loss:  {loss_fp32.item():.8f}")
print(f"fp32 grad norm:     {grad_norm_fp32:.6f}")

# ── 5. SAME FORWARD UNDER BF16 AUTOCAST (only if CUDA) ────────────────
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    print()
    print("=" * 70)
    print("DETERMINISTIC FORWARD+BACKWARD (bf16 autocast — what cloud uses)")
    print("=" * 70)
    m.zero_grad()
    torch.manual_seed(1337)
    X = torch.randint(0, 75, (4, 32), device=device)
    Y = torch.randint(0, 75, (4, 32), device=device)
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, loss_bf16 = m(X, alpha=0.5, targets=Y)
    loss_bf16.backward()
    grad_norm_bf16 = sum(p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
    print(f"bf16 forward loss:  {loss_bf16.item():.8f}")
    print(f"bf16 grad norm:     {grad_norm_bf16:.6f}")
    print(f"Δ vs fp32:          {abs(loss_bf16.item() - loss_fp32.item()):.6f}")

print()
print("=" * 70)
print("DONE — paste this output back so we can diff against RunPod")
print("=" * 70)
