"""Encoder block surgery.

Given a HuggingFace BERT-style encoder, remove a contiguous span of blocks and
splice a single replacement module in its place. The span to remove is chosen
by one of two strategies (``SurgeryPlan.selection``):

* ``"fixed"``      -- keep the first ``keep_front`` and last ``keep_back`` blocks;
                      remove everything in between (our proposed baseline).
* ``"importance"`` -- the LLM-Streamline strategy (Chen et al., ICLR 2025): score
                      each layer by the cosine similarity between its input and
                      output hidden states (high similarity = low importance =
                      small transformation), then remove the ``num_pruned``-long
                      contiguous span whose endpoints are most similar, i.e.
                      ``argmax_l cos(x(l), x(l + num_pruned))`` (their Eq. 3).

Both strategies reduce to "remove blocks ``[start : start + count]``, splice one
module there", so they share the same splice machinery and only differ in how
``(start, count)`` is determined.

The kept blocks and the inserted replacement share the residual-stream
interface ``(hidden_states, attention_mask) -> hidden_states``, so the spliced
encoder is a drop-in for the original ``BertEncoder.layer`` ModuleList.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .replacements import build_replacement


@dataclass
class SurgeryPlan:
    replacement: str = "encoder"
    selection: str = "fixed"  # "fixed" | "importance"
    # fixed mode:
    keep_front: int = 2
    keep_back: int = 2
    # importance mode (LLM-Streamline): how many contiguous layers to remove.
    num_pruned: int = 4
    # width+depth co-compression: structured FFN-width pruning of the KEPT blocks.
    # None = off; else the target intermediate size each kept block's FFN is pruned
    # down to (BERT default 3072). Composed with depth removal for higher total
    # compression than depth alone.
    prune_ffn: int | None = None
    # width+depth: number of attention heads to KEEP in each kept block (None = off).
    prune_heads: int | None = None
    # replace the KEPT blocks' self-attention with parameter-free Fourier mixing
    # (FNet-style, LoLCATs-inspired linearization of a frozen pretrained model).
    fourier_kept: bool = False

    def removal_span(self, n_layers: int, model=None, calib_batches=None) -> tuple[int, int]:
        """Return ``(start, count)``: the contiguous span of blocks to remove."""
        if self.selection == "fixed":
            count = n_layers - self.keep_front - self.keep_back
            return self.keep_front, count
        if self.selection == "importance":
            if model is None or calib_batches is None:
                raise ValueError(
                    "importance selection needs `model` and `calib_batches` "
                    "(a few input batches) to score layers."
                )
            start = select_prune_start(model, calib_batches, self.num_pruned)
            return start, self.num_pruned
        raise ValueError(f"Unknown selection '{self.selection}' (fixed|importance).")

    def describe(self, n_layers: int, start: int | None = None, count: int | None = None) -> str:
        if start is None or count is None:
            start, count = self.removal_span(n_layers) if self.selection == "fixed" else (None, self.num_pruned)
        if start is None:
            return (
                f"{n_layers} blocks -> [importance] remove {count} contiguous "
                f"(start chosen at runtime by cosine similarity) + <{self.replacement}>"
            )
        return (
            f"{n_layers} blocks -> keep[0:{start}] + <{self.replacement}> + "
            f"keep[{start + count}:{n_layers}] (removed blocks [{start}:{start + count}])"
        )


class _ReplacementLayer(nn.Module):
    """Adapts a replacement module to the ``BertLayer`` call signature.

    HF ``BertEncoder`` calls each layer as
    ``layer(hidden_states, attention_mask, ...)`` and (transformers >= 5)
    assigns the bare-tensor return straight back to ``hidden_states``. Our
    replacements already return a bare tensor, so this adapter just forwards
    the call and normalizes any legacy tuple return to a tensor.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        out = self.module(hidden_states, attention_mask=attention_mask)
        return out[0] if isinstance(out, tuple) else out


def get_encoder_layers(model) -> nn.ModuleList:
    """Locate the encoder block ModuleList across common HF architectures."""
    # BERT / RoBERTa / ELECTRA
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta"):
        return model.roberta.encoder.layer
    # DistilBERT
    if hasattr(model, "distilbert"):
        return model.distilbert.transformer.layer
    raise AttributeError(
        "Could not locate encoder layers; add this architecture to "
        "get_encoder_layers()."
    )


def set_encoder_layers(model, layers: nn.ModuleList) -> None:
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        model.encoder.layer = layers
    elif hasattr(model, "bert"):
        model.bert.encoder.layer = layers
    elif hasattr(model, "roberta"):
        model.roberta.encoder.layer = layers
    elif hasattr(model, "distilbert"):
        model.distilbert.transformer.layer = layers
    else:
        raise AttributeError("Unsupported architecture for set_encoder_layers().")


@torch.no_grad()
def layer_importance(model, calib_batches, num_pruned: int) -> torch.Tensor:
    """Cosine similarity ``cos(x(l), x(l+num_pruned))`` for each valid start ``l``.

    Implements LLM-Streamline Eq. 2/3: run the model with
    ``output_hidden_states=True`` over a few calibration batches, average the
    per-token cosine similarity between the hidden state entering layer ``l`` and
    the hidden state leaving layer ``l+num_pruned-1``. Higher = the span [l, l+n)
    transforms the residual stream least = the best span to prune.

    Returns a tensor of length ``n_layers - num_pruned + 1`` indexed by start ``l``.
    """
    model.eval()
    device = next(model.parameters()).device
    n_layers = len(get_encoder_layers(model))
    n_starts = n_layers - num_pruned + 1
    if n_starts < 1:
        raise ValueError(f"num_pruned({num_pruned}) too large for {n_layers} layers.")

    sims = torch.zeros(n_starts)
    counts = 0
    for batch in calib_batches:
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple len n_layers+1; hs[l] enters layer l, hs[l+1] leaves it.
        hs = out.hidden_states
        for l in range(n_starts):
            a, b = hs[l], hs[l + num_pruned]  # x(l) and x(l+num_pruned)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # (B, L)
            sims[l] += cos.mean().item()
        counts += 1
    return sims / max(counts, 1)


def select_prune_start(model, calib_batches, num_pruned: int) -> int:
    """argmax over start layers of the cosine-importance score (LLM-Streamline Eq. 3)."""
    sims = layer_importance(model, calib_batches, num_pruned)
    return int(torch.argmax(sims).item())


def _ffn_linears(layer):
    """``(in_linear, out_linear, setter)`` for a block's FFN across architectures.

    BERT/RoBERTa/ELECTRA: ``intermediate.dense`` (d->m), ``output.dense`` (m->d).
    DistilBERT: ``ffn.lin1`` (d->m), ``ffn.lin2`` (m->d).
    """
    inter = getattr(getattr(layer, "intermediate", None), "dense", None)
    out = getattr(getattr(layer, "output", None), "dense", None)
    if inter is not None and out is not None:
        def setter(a, b):
            layer.intermediate.dense, layer.output.dense = a, b
        return inter, out, setter
    ffn = getattr(layer, "ffn", None)
    if ffn is not None and hasattr(ffn, "lin1") and hasattr(ffn, "lin2"):
        def setter(a, b):
            layer.ffn.lin1, layer.ffn.lin2 = a, b
        return ffn.lin1, ffn.lin2, setter
    return None, None, None


@torch.no_grad()
def prune_bertlayer_ffn(layer, keep_k: int) -> bool:
    """Structured FFN-width pruning of one encoder block, in place (arch-agnostic).

    Keeps the top-``keep_k`` intermediate neurons by ``||W_in[j]||*||W_out[:,j]||``
    and resizes the FFN in/out linears. Works for BERT/RoBERTa/ELECTRA and
    DistilBERT. Returns True if it pruned.
    """
    inter, out, setter = _ffn_linears(layer)
    if inter is None:
        return False
    m = inter.weight.shape[0]
    if keep_k >= m:
        return False
    score = inter.weight.norm(dim=1) * out.weight.norm(dim=0)
    idx = torch.topk(score, keep_k).indices.sort().values
    new_inter = nn.Linear(inter.in_features, keep_k).to(inter.weight.device)
    new_inter.weight.copy_(inter.weight[idx]); new_inter.bias.copy_(inter.bias[idx])
    new_out = nn.Linear(keep_k, out.out_features).to(out.weight.device)
    new_out.weight.copy_(out.weight[:, idx]); new_out.bias.copy_(out.bias)
    setter(new_inter, new_out)
    return True


@torch.no_grad()
def prune_bertlayer_heads(layer, keep_heads: int) -> bool:
    """Structured attention-head pruning of one BERT encoder block, in place.

    Keeps the top-``keep_heads`` self-attention heads by importance (combined
    Q/K/V/O magnitude per head), shrinking query/key/value (d->h*hs) and
    attention.output.dense (h*hs->d). Composed with FFN-width and depth pruning,
    this unlocks the >63% total-compression range (FFN pruning alone saturates
    there). Returns True if it pruned.
    """
    attn = getattr(layer, "attention", None)
    self_attn = getattr(attn, "self", None) if attn is not None else None
    out_dense = getattr(getattr(attn, "output", None), "dense", None) if attn is not None else None
    if self_attn is not None and out_dense is not None:
        # BERT / RoBERTa / ELECTRA
        H, hs = self_attn.num_attention_heads, self_attn.attention_head_size
        q, k, v = self_attn.query, self_attn.key, self_attn.value
        def assign(nq, nk, nv, no):
            self_attn.query, self_attn.key, self_attn.value = nq, nk, nv
            layer.attention.output.dense = no
            self_attn.num_attention_heads = keep_heads
            self_attn.all_head_size = keep_heads * hs
    elif attn is not None and all(hasattr(attn, a) for a in ("q_lin", "k_lin", "v_lin", "out_lin")):
        # DistilBERT MultiHeadSelfAttention
        H = attn.n_heads
        hs = attn.dim // attn.n_heads
        q, k, v, out_dense = attn.q_lin, attn.k_lin, attn.v_lin, attn.out_lin
        def assign(nq, nk, nv, no):
            attn.q_lin, attn.k_lin, attn.v_lin, attn.out_lin = nq, nk, nv, no
            attn.n_heads = keep_heads
            attn.dim = keep_heads * hs
    else:
        return False
    if keep_heads >= H:
        return False

    def head_score(lin, by_row=True):
        W = lin.weight
        s = W.norm(dim=1) if by_row else W.norm(dim=0)
        return s.view(H, hs).sum(dim=1)
    score = head_score(q) + head_score(k) + head_score(v) + head_score(out_dense, by_row=False)
    heads = torch.topk(score, keep_heads).indices.sort().values
    idx = torch.cat([torch.arange(h * hs, h * hs + hs, device=q.weight.device) for h in heads])
    new_dim = keep_heads * hs
    def shrink_out(lin):
        nl = nn.Linear(lin.in_features, new_dim).to(lin.weight.device)
        nl.weight.copy_(lin.weight[idx]); nl.bias.copy_(lin.bias[idx]); return nl
    nq, nk, nv = shrink_out(q), shrink_out(k), shrink_out(v)
    no = nn.Linear(new_dim, out_dense.out_features).to(out_dense.weight.device)
    no.weight.copy_(out_dense.weight[:, idx]); no.bias.copy_(out_dense.bias)
    assign(nq, nk, nv, no)
    return True


class _FourierAttention(nn.Module):
    """Parameter-free Fourier token mixing, drop-in for a block's attention sublayer.

    Mimics ``BertAttention``'s call signature and post-LN residual, but replaces
    the O(L^2) softmax attention (4*d^2 params) with a 2D FFT over (sequence,
    hidden) — zero mixing params, O(L log L). Distillation/healing then adapts the
    surviving FFN to the Fourier-mixed stream. Inspired by FNet (2105.03824) and
    the LoLCATs/Hedgehog finding that attention can be linearized in a *frozen*
    pretrained model.
    """

    def __init__(self, hidden: int, eps: float = 1e-12):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, eps=eps)
        self.gate = nn.Parameter(torch.ones(1))  # init 1; healing can rescale

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        mixed = torch.fft.fft2(hidden_states.float(), dim=(-2, -1)).real.to(hidden_states.dtype)
        out = self.norm(hidden_states + self.gate * mixed)  # post-LN residual (BERT)
        return out, None  # (attn_output, attn_weights) interface


@torch.no_grad()
def fourierize_attention(layer) -> bool:
    """Replace one BERT/RoBERTa block's attention sublayer with Fourier mixing."""
    if not hasattr(layer, "attention"):
        return False
    hidden = layer.attention.output.dense.out_features
    eps = getattr(layer.attention.output.LayerNorm, "eps", 1e-12)
    layer.attention = _FourierAttention(hidden, eps).to(layer.attention.output.dense.weight.device)
    return True


def _partial_load(module: nn.Module, src_state: dict) -> int:
    """Copy only the source params whose name+shape match the target.

    Lets a thin-FFN encoder sub-block inherit a removed block's attention weights
    (which match in shape) while leaving its differently-shaped FFN randomly
    initialized. Returns the number of tensors copied.
    """
    tgt = module.state_dict()
    matched = {k: v for k, v in src_state.items() if k in tgt and tgt[k].shape == v.shape}
    module.load_state_dict(matched, strict=False)
    return len(matched)


def splice_replacement(
    model,
    plan: SurgeryPlan,
    calib_batches=None,
    **replacement_kwargs,
) -> nn.Module:
    """Mutate ``model`` in place: remove a contiguous span, splice a replacement.

    The span is chosen by ``plan.selection`` (``"fixed"`` or ``"importance"``).
    For ``"importance"``, pass ``calib_batches`` (an iterable of input batches)
    so the cosine-similarity search can run.

    Returns the inserted replacement module (so the caller can target it for
    training / param counting). The replacement's output aligns to the teacher's
    hidden state at index ``start + count`` -- use that for Stage-1 distillation.
    """
    layers = get_encoder_layers(model)
    n = len(layers)
    start, count = plan.removal_span(n, model=model, calib_batches=calib_batches)
    if count < 1:
        raise ValueError(f"Removal span is empty (start={start}, count={count}); nothing to remove.")
    if start < 0 or start + count > n:
        raise ValueError(f"Span [{start}:{start + count}] out of range for {n} layers.")

    hidden = model.config.hidden_size
    removed = list(layers[start : start + count])
    replacement = build_replacement(
        plan.replacement,
        hidden=hidden,
        config=model.config,
        **replacement_kwargs,
    )

    # Warm-start an `encoder` replacement from a removed middle block (the median
    # one) instead of random init -- it already lives in the right residual space.
    if plan.replacement in ("encoder", "encfuse") and removed and hasattr(replacement, "layer"):
        mid = removed[len(removed) // 2]
        try:
            _partial_load(replacement.layer, mid.state_dict())  # transfer matching weights
        except Exception:  # noqa: BLE001 -- shape mismatch across architectures; fall back to random
            pass

    # Warm-start a `recur` (kind='encoder') block from the median removed block.
    if plan.replacement == "recur" and removed and getattr(replacement, "kind", None) == "encoder":
        _partial_load(replacement.block, removed[len(removed) // 2].state_dict())

    # Warm-start a `stack` (kind='encoder') sub-block-wise: sub-block j inherits an
    # evenly-spaced removed block, so the K-block stack starts as a subsampling of
    # the original span (a strong prior). We copy only matching-shape params, so a
    # thin-FFN sub-block still inherits the (expensive, important) attention
    # weights from its source block even when the FFN width differs.
    if plan.replacement == "stack" and removed and getattr(replacement, "kind", None) == "encoder":
        k = len(replacement.blocks)
        for j, blk in enumerate(replacement.blocks):
            src = removed[min(len(removed) - 1, round((j + 0.5) * len(removed) / k))]
            _partial_load(blk, src.state_dict())

    front = list(layers[:start])
    back = list(layers[start + count :])

    # width+depth co-compression: prune the KEPT blocks' FFN width and/or heads.
    if plan.prune_ffn is not None:
        pruned = sum(prune_bertlayer_ffn(l, plan.prune_ffn) for l in front + back)
        print(f"[surgery] width-pruned FFN of {pruned} kept blocks to intermediate={plan.prune_ffn}")
    if plan.prune_heads is not None:
        ph = sum(prune_bertlayer_heads(l, plan.prune_heads) for l in front + back)
        print(f"[surgery] head-pruned {ph} kept blocks to {plan.prune_heads} heads")
    if plan.fourier_kept:
        fk = sum(fourierize_attention(l) for l in front + back)
        print(f"[surgery] replaced attention with Fourier mixing in {fk} kept blocks")

    new_layers = nn.ModuleList(front + [_ReplacementLayer(replacement)] + back)
    set_encoder_layers(model, new_layers)

    # Record where the splice sits so the trainer can align hidden states.
    model.config.num_hidden_layers = len(new_layers)
    replacement._splice_teacher_index = start + count  # teacher hs index to distill toward
    replacement._splice_start = start
    return replacement


def freeze_kept_blocks(model, plan: SurgeryPlan) -> None:
    """Freeze everything except the inserted replacement (the ``_ReplacementLayer``).

    Useful for the distillation warmup phase: train only the new middle module
    to mimic the removed span, leaving the kept blocks intact.
    """
    layers = get_encoder_layers(model)
    for layer in layers:
        requires = isinstance(layer, _ReplacementLayer)
        for p in layer.parameters():
            p.requires_grad = requires
