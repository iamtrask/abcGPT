"""nano-3 N-cohort GatedGPT with three variants.

Same transformer arch as nano-2 (n_layer=6, n_head=6, n_embd=384, block_size=256)
but generalized to N cohorts via α ∈ simplex^N.

Variants:
  - 'ungated':    vanilla GPT; α is ignored. Capacity ceiling.
  - 'per_weight': W_eff = W * (Σ_c α_c · scale_c)  with per-cohort scales
                  (out × in) on each FFN projection. Stratified-init recommended.
                  Memory: O(L · N · 4 · n_embd²). Infeasible past N ≈ 100.
  - 'hypernet':   W_eff = W * sigmoid(scale_bias + U(e_α) @ V(e_α)^T) * 2
                  where e_α = α @ cohort_embeddings.
                  Memory: O(L · (out + in) · r · d_embed) + O(N · d_embed).
                  Scales to N >> 1M with d_embed=64, r=16.

Attention QKV is shared across cohorts (slider only modulates FFN). This matches
the tiny-3-source design that validated on real text and the
[[abcgpt-dual-source-training]] / [[hypernet-validated]] toy results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoGPTConfig:
    vocab_size: int = 103
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False
    n_cohorts: int = 3
    variant: str = "ungated"          # 'ungated' | 'per_weight' | 'hypernet'
    d_embed: int = 8                  # cohort-embedding dim (hypernet)
    rank: int = 16                    # rank of the factorized hypernet
    cohort_names: List[str] = field(default_factory=lambda: ["shake", "ts", "code"])


# ---------------------------------------------------------------------------
# Attention (shared across cohorts; α not consumed)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# ---------------------------------------------------------------------------
# FFN variants
# ---------------------------------------------------------------------------
class UngatedFFN(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, alpha=None, cohort_embeddings=None):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class PerWeightGatedLinear(nn.Module):
    """W_eff = W * mix where mix = (α.view(-1,1,1) * scales).sum(0).

    `scales` is (n_cohorts, out, in). Default init is ones; the trainer will
    overwrite with stratified-init samples.
    """
    def __init__(self, in_features, out_features, n_cohorts, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_cohorts = n_cohorts
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scales = nn.Parameter(torch.ones(n_cohorts, out_features, in_features))

    def forward(self, x, alpha):
        mix = (alpha.view(-1, 1, 1) * self.scales).sum(dim=0)
        W_eff = self.weight * mix
        return F.linear(x, W_eff, self.bias)


class PerWeightFFN(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.c_fc = PerWeightGatedLinear(cfg.n_embd, 4 * cfg.n_embd, cfg.n_cohorts, bias=cfg.bias)
        self.c_proj = PerWeightGatedLinear(4 * cfg.n_embd, cfg.n_embd, cfg.n_cohorts, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, alpha, cohort_embeddings=None):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x, alpha)), alpha))


class FactorizedHypernet(nn.Module):
    """g(e) = sigmoid(scale_bias + U(e) @ V(e)^T) * 2  bounded in [0, 2]."""

    def __init__(self, d_embed, out_features, in_features, rank):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.U = nn.Linear(d_embed, out_features * rank, bias=False)
        self.V = nn.Linear(d_embed, in_features * rank, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

    def forward(self, e):
        if e.dim() == 1:
            U = self.U(e).view(self.out_features, self.rank)
            V = self.V(e).view(self.in_features, self.rank)
            raw = self.scale_bias + U @ V.T
        else:
            B = e.shape[0]
            U = self.U(e).view(B, self.out_features, self.rank)
            V = self.V(e).view(B, self.in_features, self.rank)
            raw = self.scale_bias.unsqueeze(0) + torch.bmm(U, V.transpose(1, 2))
        return torch.sigmoid(raw) * 2.0


class HypernetGatedLinear(nn.Module):
    def __init__(self, in_features, out_features, n_cohorts, d_embed, rank, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_cohorts = n_cohorts
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.hypernet = FactorizedHypernet(d_embed, out_features, in_features, rank)

    def forward(self, x, alpha, cohort_embeddings):
        e_alpha = alpha @ cohort_embeddings  # (d_embed,)
        mix = self.hypernet(e_alpha)
        W_eff = self.weight * mix
        return F.linear(x, W_eff, self.bias)


class HypernetFFN(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.c_fc = HypernetGatedLinear(cfg.n_embd, 4 * cfg.n_embd, cfg.n_cohorts,
                                          cfg.d_embed, cfg.rank, bias=cfg.bias)
        self.c_proj = HypernetGatedLinear(4 * cfg.n_embd, cfg.n_embd, cfg.n_cohorts,
                                            cfg.d_embed, cfg.rank, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, alpha, cohort_embeddings):
        h = self.c_fc(x, alpha, cohort_embeddings)
        return self.dropout(self.c_proj(F.gelu(h), alpha, cohort_embeddings))


# ---------------------------------------------------------------------------
# Block + Model
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.variant = cfg.variant
        if cfg.variant == "ungated":
            self.ffn = UngatedFFN(cfg)
        elif cfg.variant == "per_weight":
            self.ffn = PerWeightFFN(cfg)
        elif cfg.variant == "hypernet":
            self.ffn = HypernetFFN(cfg)
        else:
            raise ValueError(f"unknown variant: {cfg.variant}")

    def forward(self, x, alpha=None, cohort_embeddings=None):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        if self.variant == "ungated":
            return x + self.ffn(h)
        elif self.variant == "per_weight":
            return x + self.ffn(h, alpha)
        else:  # hypernet
            return x + self.ffn(h, alpha, cohort_embeddings)


class NanoGPT(nn.Module):
    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # tied
        if cfg.variant == "hypernet":
            self.cohort_embeddings = nn.Parameter(torch.randn(cfg.n_cohorts, cfg.d_embed) * 0.1)
        else:
            self.cohort_embeddings = None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if not isinstance(module, (PerWeightGatedLinear, HypernetGatedLinear)) \
               and not (hasattr(module, 'weight') and module.weight.dim() == 4):
                # Standard linear init only on plain Linear layers (kaiming already
                # set on the gated ones; hypernet U/V already normal_(0, 0.01))
                if not hasattr(module, '_skip_default_init'):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, alpha=None, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        for block in self.blocks:
            x = block(x, alpha, self.cohort_embeddings)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
        )
        return logits, loss

    def gated_layers(self):
        """Return list of (PerWeightGatedLinear | HypernetGatedLinear) for init/anchor."""
        out = []
        for block in self.blocks:
            if isinstance(block.ffn, (PerWeightFFN, HypernetFFN)):
                out.append(block.ffn.c_fc)
                out.append(block.ffn.c_proj)
        return out

    def num_params(self, exclude_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.wpe.weight.numel()
        return n
