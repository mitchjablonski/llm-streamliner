"""Pluggable replacement modules.

Every replacement maps a sequence of hidden states to a sequence of the same
shape: ``(batch, seq_len, hidden)`` -> ``(batch, seq_len, hidden)``.

All modules share the signature::

    forward(hidden_states, attention_mask=None) -> hidden_states

``attention_mask`` is the *extended* additive mask used by HF attention
(``0`` for keep, large-negative for pad), or ``None``. Modules that do not
attend across tokens (MLP/CNN variants) simply ignore it, except where a
padding mask is needed to zero out pad positions.

Register new modules with ``@register("key")`` so they show up in the CLI.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register(key: str):
    def deco(cls):
        REGISTRY[key] = cls
        cls.key = key
        return cls

    return deco


def build_replacement(key: str, hidden: int, **kwargs) -> nn.Module:
    """Factory: instantiate a registered replacement by key."""
    if key not in REGISTRY:
        raise KeyError(f"Unknown replacement '{key}'. Options: {sorted(REGISTRY)}")
    return REGISTRY[key](hidden=hidden, **kwargs)


def _bert_layer_config(config):
    """Build a ``BertConfig`` mirroring any host encoder's geometry.

    The ``encoder``/``hybrid`` replacements use a ``BertLayer`` for their
    attention branch, but the host backbone may be DistilBERT/RoBERTa/etc. whose
    configs use different field names (``dim`` vs ``hidden_size``) and lack
    BERT-only attributes (``is_decoder``). We read the host's geometry via
    several known aliases and synthesize a clean post-LN ``BertConfig`` so the
    spliced block matches the residual-stream dims regardless of backbone.
    """
    from transformers.models.bert.configuration_bert import BertConfig

    def pick(*names, default=None):
        for n in names:
            if hasattr(config, n) and getattr(config, n) is not None:
                return getattr(config, n)
        return default

    hidden = pick("hidden_size", "dim", "d_model")
    heads = pick("num_attention_heads", "n_heads", "encoder_attention_heads", default=max(1, hidden // 64))
    inter = pick("intermediate_size", "hidden_dim", "ffn_dim", "encoder_ffn_dim", default=4 * hidden)
    act = pick("hidden_act", "activation", "activation_function", default="gelu")
    drop = pick("hidden_dropout_prob", "dropout", default=0.1)
    attn_drop = pick("attention_probs_dropout_prob", "attention_dropout", default=0.1)
    eps = pick("layer_norm_eps", default=1e-12)
    return BertConfig(
        hidden_size=hidden,
        num_attention_heads=heads,
        intermediate_size=inter,
        hidden_act=act if isinstance(act, str) else "gelu",
        hidden_dropout_prob=drop,
        attention_probs_dropout_prob=attn_drop,
        layer_norm_eps=eps,
        is_decoder=False,
        add_cross_attention=False,
    )


def _zero_(layer: nn.Module) -> nn.Module:
    """Zero a layer's weight and bias -> the module starts as identity (with residual).

    The removed span is low-importance (input ~= output), so starting the
    replacement at ``f(x) = x`` is a strong initialization: Stage-1 distillation
    then only has to learn the small delta. See research notes Section 5.
    """
    nn.init.zeros_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)
    return layer


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
class _Residual(nn.Module):
    """Wrap a token-wise transform with pre-LayerNorm + residual.

    Keeps replacements drop-in compatible with the residual stream the kept
    encoder blocks operate on.
    """

    def __init__(self, hidden: int, body: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.body = body
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):  # noqa: D401
        return x + self.drop(self.body(self.norm(x)))


# --------------------------------------------------------------------------- #
# Replacement modules
# --------------------------------------------------------------------------- #
@register("encoder")
class EncoderBlockReplacement(nn.Module):
    """A single standard transformer encoder block (our proposed baseline).

    Lazily mirrors the kept blocks' config so it matches their interface. The
    actual ``nn.Module`` is constructed in :func:`surgery.splice_replacement`
    where the source model's config is available; here we keep a thin stub so
    the registry is uniform.
    """

    def __init__(self, hidden: int, config=None, **_):
        super().__init__()
        if config is None:
            raise ValueError(
                "encoder replacement requires the source model `config`; "
                "surgery.splice_replacement passes it in."
            )
        from transformers.models.bert.modeling_bert import BertLayer

        self.layer = BertLayer(_bert_layer_config(config))

    def forward(self, x, attention_mask=None):
        out = self.layer(x, attention_mask=attention_mask)
        return out[0] if isinstance(out, tuple) else out  # BertLayer returns a tensor (transformers >= 5)


@register("mlp")
class VanillaMLP(nn.Module):
    """Per-token 2-layer MLP with GELU, pre-norm + residual."""

    def __init__(self, hidden: int, expansion: float = 4.0, dropout: float = 0.1, **_):
        super().__init__()
        inner = int(hidden * expansion)
        body = nn.Sequential(
            nn.Linear(hidden, inner), nn.GELU(), _zero_(nn.Linear(inner, hidden))
        )
        self.block = _Residual(hidden, body, dropout)

    def forward(self, x, attention_mask=None):
        return self.block(x, attention_mask)


@register("glu_mlp")
class GLUMLP(nn.Module):
    """Gated MLP (SwiGLU-style): (W1 x) * SiLU(W2 x) -> W3."""

    def __init__(self, hidden: int, expansion: float = 4.0, dropout: float = 0.1, **_):
        super().__init__()
        inner = int(hidden * expansion)
        self.norm = nn.LayerNorm(hidden)
        self.w_val = nn.Linear(hidden, inner)
        self.w_gate = nn.Linear(hidden, inner)
        self.w_out = _zero_(nn.Linear(inner, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        h = self.norm(x)
        h = self.w_val(h) * F.silu(self.w_gate(h))
        return x + self.drop(self.w_out(h))


@register("bottleneck")
class BottleneckMLP(nn.Module):
    """Low-rank bottleneck MLP (down-project -> up-project). Cheap params."""

    def __init__(self, hidden: int, rank: int = 64, dropout: float = 0.1, **_):
        super().__init__()
        body = nn.Sequential(
            nn.Linear(hidden, rank), nn.GELU(), _zero_(nn.Linear(rank, hidden))
        )
        self.block = _Residual(hidden, body, dropout)

    def forward(self, x, attention_mask=None):
        return self.block(x, attention_mask)


@register("mixer")
class MixerBlock(nn.Module):
    """MLP-Mixer style: token-mixing then channel-mixing.

    Token-mixing operates over the sequence axis, so ``max_seq_len`` must be
    fixed. Padded positions are not masked in token-mixing (a known Mixer
    limitation); keep sequences padded to ``max_seq_len`` consistently.
    """

    def __init__(
        self,
        hidden: int,
        max_seq_len: int = 128,
        token_expansion: float = 0.5,
        channel_expansion: float = 4.0,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        tok_inner = int(max_seq_len * token_expansion)
        ch_inner = int(hidden * channel_expansion)
        self.norm_tok = nn.LayerNorm(hidden)
        self.tok_mix = nn.Sequential(
            nn.Linear(max_seq_len, tok_inner), nn.GELU(), nn.Linear(tok_inner, max_seq_len)
        )
        self.norm_ch = nn.LayerNorm(hidden)
        self.ch_mix = nn.Sequential(
            nn.Linear(hidden, ch_inner), nn.GELU(), _zero_(nn.Linear(ch_inner, hidden))
        )
        # zero the token-mix output too so the whole block starts as identity
        _zero_(self.tok_mix[-1])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        b, n, d = x.shape
        if n != self.max_seq_len:
            raise ValueError(
                f"mixer expects seq_len={self.max_seq_len}, got {n}. "
                "Pad/truncate batches to a fixed length for this replacement."
            )
        # token mixing: transpose to mix over sequence
        h = self.norm_tok(x).transpose(1, 2)  # (b, d, n)
        x = x + self.drop(self.tok_mix(h).transpose(1, 2))
        # channel mixing
        x = x + self.drop(self.ch_mix(self.norm_ch(x)))
        return x


@register("cnn")
class CNNBlock(nn.Module):
    """1D conv over the sequence: depthwise (mixes tokens) + pointwise (mixes channels)."""

    def __init__(self, hidden: int, kernel_size: int = 5, dropout: float = 0.1, **_):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size, padding=pad, groups=hidden)
        self.pointwise = _zero_(nn.Conv1d(hidden, hidden, 1))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        h = self.norm(x).transpose(1, 2)  # (b, d, n)
        h = self.pointwise(self.act(self.depthwise(h))).transpose(1, 2)
        return x + self.drop(h)


@register("hybrid")
class HybridConcat(nn.Module):
    """concat([mlp_branch(x), encoder_branch(x)]) -> tiny fusion MLP -> residual.

    Lets a cheap token-wise branch and an attention branch each contribute; a
    small fusion layer learns how to combine them.
    """

    def __init__(self, hidden: int, config=None, expansion: float = 2.0, dropout: float = 0.1, **_):
        super().__init__()
        if config is None:
            raise ValueError("hybrid replacement requires the source model `config`.")
        from transformers.models.bert.modeling_bert import BertLayer

        inner = int(hidden * expansion)
        self.norm = nn.LayerNorm(hidden)
        self.mlp_branch = nn.Sequential(
            nn.Linear(hidden, inner), nn.GELU(), nn.Linear(inner, hidden)
        )
        self.encoder_branch = BertLayer(_bert_layer_config(config))
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), _zero_(nn.Linear(hidden, hidden))
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        h = self.norm(x)
        mlp_out = self.mlp_branch(h)
        enc_out = self.encoder_branch(h, attention_mask=attention_mask)
        enc_out = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        fused = self.fuse(torch.cat([mlp_out, enc_out], dim=-1))
        return x + self.drop(fused)


def _make_subblock(kind: str, hidden: int, config, expansion: float, rank: int, dropout: float,
                   inter: int | None = None) -> nn.Module:
    """Build one sub-block for a :class:`BlockStack` of the requested ``kind``.

    ``inter`` overrides the encoder sub-block's FFN width so a deeper stack can be
    held to the same parameter budget as a single full-width block (a thin-FFN
    encoder block: attention is unchanged, only the FFN shrinks).
    """
    if kind == "encoder":
        from transformers.models.bert.modeling_bert import BertLayer

        cfg = _bert_layer_config(config)
        if inter is not None:
            cfg.intermediate_size = int(inter)
        return BertLayer(cfg)
    inner = int(hidden * expansion)
    if kind == "mlp":
        body = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), _zero_(nn.Linear(inner, hidden)))
        return _Residual(hidden, body, dropout)
    if kind == "bottleneck":
        body = nn.Sequential(nn.Linear(hidden, rank), nn.GELU(), _zero_(nn.Linear(rank, hidden)))
        return _Residual(hidden, body, dropout)
    if kind == "glu":
        return _GLUResidual(hidden, inner, dropout)
    raise KeyError(f"Unknown stack sub-block kind '{kind}' (encoder|mlp|bottleneck|glu).")


class _GLUResidual(nn.Module):
    """SwiGLU body wrapped in pre-norm + residual, zero-init output (identity start)."""

    def __init__(self, hidden: int, inner: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.w_val = nn.Linear(hidden, inner)
        self.w_gate = nn.Linear(hidden, inner)
        self.w_out = _zero_(nn.Linear(inner, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        h = self.norm(x)
        return x + self.drop(self.w_out(self.w_val(h) * F.silu(self.w_gate(h))))


@register("stack")
class BlockStack(nn.Module):
    """A short stack of ``depth`` sub-blocks with **intermediate anchoring**.

    Replaces a long removed span with ``depth`` cheap sub-blocks. The key feature
    is :attr:`anchors`: after a forward pass it holds the output of *each*
    sub-block, so Stage-1 distillation can supervise every sub-block against an
    evenly-spaced teacher hidden state (TinyBERT-style layer-wise matching,
    applied to span replacement). This decomposes the hard ``span -> 1`` hidden
    reconstruction into ``depth`` easy sub-steps instead of one big jump.

    ``kind`` picks the sub-block family (encoder|mlp|bottleneck|glu). With
    ``kind='encoder'`` and ``depth`` warm-started from removed blocks (see
    ``surgery.splice_replacement``) this is a strong, faithful replacement; with
    cheaper kinds it trades capacity for params.
    """

    def __init__(
        self,
        hidden: int,
        config=None,
        depth: int = 2,
        kind: str = "encoder",
        expansion: float = 4.0,
        rank: int = 256,
        inter: int | None = None,
        dropout: float = 0.1,
        skip: bool = False,
        **_,
    ):
        super().__init__()
        if kind == "encoder" and config is None:
            raise ValueError("stack(kind='encoder') requires the source model `config`.")
        self.depth = depth
        self.kind = kind
        self.skip = skip
        self.blocks = nn.ModuleList(
            _make_subblock(kind, hidden, config, expansion, rank, dropout, inter)
            for _ in range(depth)
        )
        # gated input-skip from the stack input x0 into each sub-block (init 0 => no-op
        # at start, so it can only help). Lets later sub-blocks re-access the span
        # input directly when reconstructing their (absolute) anchored teacher target.
        self.skip_gate = nn.Parameter(torch.zeros(depth)) if skip else None
        self.anchors: list = []

    def forward(self, x, attention_mask=None):
        x0, anchors = x, []
        for i, blk in enumerate(self.blocks):
            inp = x + self.skip_gate[i] * x0 if self.skip else x
            out = blk(inp, attention_mask=attention_mask)
            x = out[0] if isinstance(out, tuple) else out
            anchors.append(x)
        self.anchors = anchors  # exposed for anchored Stage-1 distillation
        return x


@register("fnet")
class FNetReplacement(nn.Module):
    """Fourier token mixing (FNet, arXiv 2105.03824) + FFN. ZERO mixing params.

    Replaces attention with a parameter-free 2D FFT over (sequence, hidden); the
    token mixing is O(L log L) and natively bidirectional. A learnable scalar gate
    (init 0) makes the module start as identity so Stage-1 distillation learns how
    much Fourier mixing to inject. The only learnable params are the FFN + gate, so
    this is a much cheaper, faster span replacement than an attention block.
    """

    def __init__(self, hidden: int, expansion: float = 4.0, dropout: float = 0.1, **_):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        inner = int(hidden * expansion)
        self.ffn = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), _zero_(nn.Linear(inner, hidden)))
        self.gate = nn.Parameter(torch.zeros(1))  # identity start
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        h = self.norm1(x)
        mixed = torch.fft.fft2(h.float(), dim=(-2, -1)).real.to(x.dtype)
        x = x + self.gate * mixed
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


@register("fusion")
class FusionReplacement(nn.Module):
    """concat[transformer-block branch, dense branch] -> linear fuse -> final block.

    Implements the "combine a (thin) transformer block's output with dense-layer
    outputs, mix through a linear layer, then pass a final block before output"
    idea. Two thin transformer blocks (configurable FFN width via ``thin_inter``)
    plus a dense branch and a 2d->d->d fusion give a more expressive span
    reconstruction than a single block, while staying compact enough to pair with
    aggressive kept-block pruning. Identity-initialized (zero-init output
    projection) so it starts as f(x)=x and distillation learns the delta.
    """

    def __init__(self, hidden: int, config=None, dense_expansion: float = 2.0,
                 thin_inter: int | None = 768, dropout: float = 0.1, **_):
        super().__init__()
        if config is None:
            raise ValueError("fusion replacement requires the source model `config`.")
        from transformers.models.bert.modeling_bert import BertLayer

        cfg = _bert_layer_config(config)
        if thin_inter is not None:
            cfg.intermediate_size = int(thin_inter)
        self.norm = nn.LayerNorm(hidden)
        self.block1 = BertLayer(cfg)
        inner = int(hidden * dense_expansion)
        self.dense = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), nn.Linear(inner, hidden))
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.final = BertLayer(cfg)
        self.out_proj = _zero_(nn.Linear(hidden, hidden))  # identity start
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        h = self.norm(x)
        a = self.block1(h, attention_mask=attention_mask)
        a = a[0] if isinstance(a, tuple) else a
        b = self.dense(h)
        f = self.fuse(torch.cat([a, b], dim=-1)) + a  # fused refinement of the block branch
        o = self.final(f, attention_mask=attention_mask)
        o = o[0] if isinstance(o, tuple) else o
        return x + self.drop(self.out_proj(o))


def _keep_mask(attention_mask, b, n, device, dtype):
    """Turn the extended additive mask (b,1,1,n; 0 keep / -inf pad) into (b,n,1) keep weights."""
    if attention_mask is None:
        return torch.ones(b, n, 1, device=device, dtype=dtype)
    m = attention_mask.reshape(b, -1)[:, :n]  # (b,n)
    return (m > -1.0).to(dtype).unsqueeze(-1)


@register("linattn")
class LinearAttention(nn.Module):
    """Linear (kernelized) attention — O(L) novel attention + FFN, identity-init.

    Replaces softmax O(L^2) attention with the elu+1 feature map and the
    associativity trick phi(Q)(phi(K)^T V), so cost is linear in sequence length.
    Padded keys/values are zeroed via the mask. Output projection and FFN are
    zero-initialized so the block starts as identity for clean span distillation.
    """

    def __init__(self, hidden: int, heads: int | None = None, expansion: float = 4.0, dropout: float = 0.1, **_):
        super().__init__()
        if heads is None or hidden % heads != 0:
            heads = max(1, hidden // 64)  # standard head dim 64
        self.h, self.dh = heads, hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden); self.k = nn.Linear(hidden, hidden); self.v = nn.Linear(hidden, hidden)
        self.o = _zero_(nn.Linear(hidden, hidden))
        inner = int(hidden * expansion)
        self.ffn = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), _zero_(nn.Linear(inner, hidden)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        b, n, d = x.shape
        hn = self.norm1(x)
        q = (F.elu(self.q(hn).view(b, n, self.h, self.dh)) + 1)
        k = (F.elu(self.k(hn).view(b, n, self.h, self.dh)) + 1)
        v = self.v(hn).view(b, n, self.h, self.dh)
        keep = _keep_mask(attention_mask, b, n, x.device, x.dtype).unsqueeze(-1)  # (b,n,1,1)
        k = k * keep; v = v * keep
        kv = torch.einsum("bnhd,bnhe->bhde", k, v)          # (b,h,dh,dh)
        ksum = k.sum(dim=1)                                  # (b,h,dh)
        z = 1.0 / (torch.einsum("bnhd,bhd->bnh", q, ksum) + 1e-6)
        out = torch.einsum("bnhd,bhde->bnhe", q, kv) * z.unsqueeze(-1)
        x = x + self.drop(self.o(out.reshape(b, n, d)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


@register("fusion2")
class MultiBranchFusion(nn.Module):
    """Multi-branch DOUBLE fusion: concat[transformer, conv, mlp] -> fuse -> fuse.

    Combines three complementary token/channel mixers — a (thin) attention block
    (global mixing), a depthwise conv (local mixing), and a per-token MLP (channel
    mixing) — concatenates their outputs, then passes through TWO fusion MLPs before
    a zero-init residual. This is the "fusion + fusion" design: a learned two-stage
    combiner over diverse mechanisms. Identity-initialized.
    """

    def __init__(self, hidden: int, config=None, thin_inter: int | None = 768, kernel_size: int = 7,
                 expansion: float = 2.0, dropout: float = 0.1, **_):
        super().__init__()
        if config is None:
            raise ValueError("fusion2 requires the source model `config`.")
        from transformers.models.bert.modeling_bert import BertLayer
        cfg = _bert_layer_config(config)
        if thin_inter is not None:
            cfg.intermediate_size = int(thin_inter)
        self.norm = nn.LayerNorm(hidden)
        self.attn = BertLayer(cfg)
        pad = kernel_size // 2
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, padding=pad, groups=hidden)
        inner = int(hidden * expansion)
        self.mlp = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), nn.Linear(inner, hidden))
        self.fuse1 = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU())
        self.fuse2 = _zero_(nn.Linear(hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        h = self.norm(x)
        a = self.attn(h, attention_mask=attention_mask)
        a = a[0] if isinstance(a, tuple) else a
        c = self.conv(h.transpose(1, 2)).transpose(1, 2)
        m = self.mlp(h)
        fused = self.fuse2(self.fuse1(torch.cat([a, c, m], dim=-1)))
        return x + self.drop(fused)


@register("recur")
class RecurrentBlock(nn.Module):
    """Weight-tied recurrent replacement: ONE sub-block applied ``steps`` times.

    Gets the *depth* benefit of a K-block anchored stack (which beats the single
    block) at the *parameter cost of ONE block* — the params are shared across
    steps. Exposes per-step outputs (``anchors``) so anchored Stage-1 distillation
    can match each unrolled step to an evenly-spaced teacher hidden state. Targets
    "more performance, small parameter count": e.g. a thin-FFN encoder block
    (inter=768, 3.5M) unrolled 3× gives depth-3 reconstruction at 3.5M, vs a
    depth-3 stack's ~10M or the single block's 7M.
    """

    def __init__(self, hidden: int, config=None, steps: int = 3, kind: str = "encoder",
                 inter: int | None = None, expansion: float = 4.0, rank: int = 256,
                 dropout: float = 0.1, **_):
        super().__init__()
        if kind == "encoder" and config is None:
            raise ValueError("recur(kind='encoder') requires the source model `config`.")
        self.depth = steps            # consumed by the anchoring logic in the training loop
        self.steps = steps
        self.kind = kind
        self.block = _make_subblock(kind, hidden, config, expansion, rank, dropout, inter)
        self.anchors: list = []

    def forward(self, x, attention_mask=None):
        anchors = []
        for _ in range(self.steps):
            out = self.block(x, attention_mask=attention_mask)
            x = out[0] if isinstance(out, tuple) else out
            anchors.append(x)
        self.anchors = anchors
        return x


@register("encfuse")
class EncoderPlusFusion(nn.Module):
    """Full encoder block (main path) + a GATED additive fusion correction.

    Fixes the flaw in concat->fuse designs (`hybrid`, `fusion2`), which replace the
    attention path with a fused mixture and came out *worse* than a bare encoder
    block. Here the encoder block is the main path and the fusion — a learned
    combination of [encoder-out, depthwise-conv, per-token MLP] — is added through a
    scalar gate initialized to 0 and a zero-init projection. So the module **starts
    exactly as the pure encoder block** (the strongest single replacement) and the
    fusion can only add value; it cannot dilute the attention path.
    """

    def __init__(self, hidden: int, config=None, kernel_size: int = 7, expansion: float = 2.0,
                 dropout: float = 0.1, **_):
        super().__init__()
        if config is None:
            raise ValueError("encfuse requires the source model `config`.")
        from transformers.models.bert.modeling_bert import BertLayer
        self.layer = BertLayer(_bert_layer_config(config))  # warm-startable main path
        self.norm = nn.LayerNorm(hidden)
        pad = kernel_size // 2
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, padding=pad, groups=hidden)
        inner = int(hidden * expansion)
        self.mlp = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), nn.Linear(inner, hidden))
        self.fuse = _zero_(nn.Linear(3 * hidden, hidden))  # correction starts at 0
        self.gate = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        eo = self.layer(x, attention_mask=attention_mask)
        eo = eo[0] if isinstance(eo, tuple) else eo   # full encoder-block output (its own residual)
        h = self.norm(x)
        c = self.conv(h.transpose(1, 2)).transpose(1, 2)
        m = self.mlp(h)
        corr = self.fuse(torch.cat([eo, c, m], dim=-1))
        return eo + self.gate * self.drop(corr)        # gate=0 at init -> exactly the encoder block


@register("linseed")
class LinSeed(nn.Module):
    """Closed-form-seeded linear span-map + optional nonlinear correction.

    ``out = x + (W x + b) + body(LayerNorm(x))``

    The linear path ``W,b`` is seeded (via :meth:`seed_linear`) to the
    least-squares-optimal map of the removed span's input->output **delta**
    ``h_out - h_in`` on calibration data -- the span is approximately linear in
    the residual stream ("your transformer is secretly linear"), so this gives
    the module the best linear approximation *for free*. The nonlinear ``body``
    (zero-initialized, so it starts as identity) only has to learn the residual
    correction. ``body_kind``: ``none`` (pure linear, ~0.6M), ``bottleneck``
    (low-rank correction), or ``mlp``.
    """

    def __init__(self, hidden: int, body_kind: str = "bottleneck", expansion: float = 4.0,
                 rank: int = 64, dropout: float = 0.1, **_):
        super().__init__()
        self.skip = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)  # default: zero delta == identity span map
        self.norm = nn.LayerNorm(hidden)
        if body_kind == "none":
            self.body = None
        elif body_kind == "mlp":
            inner = int(hidden * expansion)
            self.body = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), _zero_(nn.Linear(inner, hidden)))
        elif body_kind == "bottleneck":
            self.body = nn.Sequential(nn.Linear(hidden, rank), nn.GELU(), _zero_(nn.Linear(rank, hidden)))
        else:
            raise KeyError(f"linseed body_kind '{body_kind}' (none|bottleneck|mlp)")
        self.drop = nn.Dropout(dropout)

    @torch.no_grad()
    def seed_linear(self, W, b):
        self.skip.weight.copy_(W.to(self.skip.weight.dtype))
        self.skip.bias.copy_(b.to(self.skip.bias.dtype))

    def forward(self, x, attention_mask=None):
        out = x + self.skip(x)
        if self.body is not None:
            out = out + self.drop(self.body(self.norm(x)))
        return out


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
