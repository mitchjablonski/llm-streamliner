"""Training the spliced model.

Two-stage recipe (per the research notes):

Stage 1 -- *hidden-state distillation* of the inserted module only.
    Freeze the kept blocks. Run the **teacher** (original, unmodified model) and
    the **student** (spliced model) on the same inputs. Match the student's
    hidden state at the splice exit to the teacher's hidden state at the
    corresponding layer (MSE + cosine). This teaches the new middle module to
    reproduce what the removed span produced.

Stage 2 -- *end-to-end fine-tune* on the task, optionally with KD logit loss
    against the teacher, optionally unfreezing the kept blocks.

This module provides the loss helpers and a thin training loop. Wire it up from
``scripts/run_generative.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    stage1_epochs: int = 2
    stage1_lr: float = 1e-3  # the small module trains from near-scratch -> higher LR
    stage1_max_batches: int | None = None  # cap batches/epoch for fast iteration
    stage2_epochs: int = 3
    stage2_max_batches: int | None = None
    lr: float = 5e-5  # stage-2 (end-to-end heal) LR -- BERT fine-tune range
    weight_decay: float = 0.01
    grad_accum: int = 1
    amp: bool = True  # mixed precision -- important on 8GB
    freeze_kept_in_stage2: bool = False  # if True, only the spliced module + head heal
    kd_alpha: float = 0.5  # weight on KD logit loss in stage 2 (0 = pure CE)
    kd_temp: float = 2.0
    hidden_mse_weight: float = 1.0
    hidden_cos_weight: float = 1.0
    warmup_ratio: float = 0.06  # linear warmup fraction for the Stage-2 LR schedule
    stage2_hidden_weight: float = 0.0  # >0 adds full feature distillation in Stage 2


def hidden_state_loss(student_h, teacher_h, mse_w=1.0, cos_w=1.0):
    """MSE + (1 - cosine) between two hidden-state tensors (b, n, d)."""
    mse = F.mse_loss(student_h, teacher_h)
    cos = 1.0 - F.cosine_similarity(student_h, teacher_h, dim=-1).mean()
    return mse_w * mse + cos_w * cos


def kd_logit_loss(student_logits, teacher_logits, temp=2.0):
    """Soft-target KL divergence (Hinton KD)."""
    s = F.log_softmax(student_logits / temp, dim=-1)
    t = F.softmax(teacher_logits / temp, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temp ** 2)


def make_optimizer(model, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


# --------------------------------------------------------------------------- #
# Stage 1: hidden-state distillation of the inserted module
# --------------------------------------------------------------------------- #
def _extended_attention_mask(mask, dtype):
    """Build the HF additive attention mask: 0 to keep, large-negative for pad.

    Shape ``(B, 1, 1, L)`` so it broadcasts over heads/queries inside the
    replacement's attention (used by the `encoder` / `hybrid` modules; ignored
    by the MLP/CNN variants).
    """
    if mask is None:
        return None
    ext = mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - ext) * torch.finfo(dtype).min


def _batch_inputs(batch, device):
    keys = ("input_ids", "attention_mask", "token_type_ids")
    return {k: batch[k].to(device) for k in keys if k in batch}


@torch.no_grad()
def compute_linear_seed(teacher, batches, start: int, teacher_index: int, device, ridge: float = 1.0):
    """Least-squares-optimal linear map of the span's input->output **delta**.

    Fits ``(h_out - h_in) ~= W h_in + b`` by ridge regression over calibration
    tokens (the span is approximately linear in the residual stream). Accumulates
    the normal-equation matrices so memory is O(d^2), not O(N). Returns ``(W, b)``
    with ``W`` shape (d, d) ready for ``nn.Linear.weight`` (i.e. ``y = W x + b``).
    """
    teacher.eval()
    d = teacher.config.hidden_size
    XtX = torch.zeros(d + 1, d + 1, device=device, dtype=torch.float64)
    XtY = torch.zeros(d + 1, d, device=device, dtype=torch.float64)
    for batch in batches:
        inputs = _batch_inputs(batch, device)
        hs = teacher(**inputs, output_hidden_states=True).hidden_states
        h_in = hs[start].reshape(-1, d).double()
        delta = (hs[teacher_index].reshape(-1, d) - hs[start].reshape(-1, d)).double()
        ones = torch.ones(h_in.shape[0], 1, device=device, dtype=torch.float64)
        X = torch.cat([h_in, ones], dim=1)  # (N, d+1)
        XtX += X.t() @ X
        XtY += X.t() @ delta
    reg = ridge * torch.eye(d + 1, device=device, dtype=torch.float64)
    reg[d, d] = 0.0  # don't regularize the bias row
    sol = torch.linalg.solve(XtX + reg, XtY)  # (d+1, d): rows = [W^T ; b]
    W = sol[:d, :].t().contiguous().float()   # (d, d) so that y = W x
    b = sol[d, :].contiguous().float()         # (d,)
    return W, b


def anchor_indices(start: int, teacher_index: int, depth: int) -> list[int]:
    """Evenly-spaced teacher hidden-state indices to anchor a ``depth``-block stack.

    For a span removed over ``[start, teacher_index)`` replaced by ``depth``
    sub-blocks, sub-block ``j`` (1-indexed) is supervised against teacher hidden
    state ``start + round(j * span / depth)``. The last anchor is exactly
    ``teacher_index`` (the span exit), so a depth-1 stack reduces to the standard
    single-target distillation.
    """
    span = teacher_index - start
    return [start + round(j * span / depth) for j in range(1, depth + 1)]


def run_stage1(
    replacement,
    teacher,
    dataloader,
    start: int,
    teacher_index: int,
    cfg: TrainConfig,
    device,
    log_every: int = 50,
    anchors: list[int] | None = None,
):
    """Train ``replacement`` to map the removed span's input -> its output.

    The teacher (the *original*, unmodified model) is run with
    ``output_hidden_states=True``; ``hidden_states[start]`` is the residual
    stream entering the removed span and ``hidden_states[teacher_index]`` is what
    it produced. Because the student's kept front blocks are identical to the
    teacher's, ``hidden_states[start]`` is exactly what the spliced model will
    feed into the replacement -- so we can train the module standalone
    (LLM-Streamline Eq. 4), no student forward or task labels required.

    **Anchored mode (our contribution).** If ``anchors`` is given *and* the
    replacement exposes per-sub-block outputs (``replacement.anchors``, e.g. a
    :class:`~src.replacements.BlockStack`), each sub-block output is matched to
    its own intermediate teacher hidden state rather than only the span exit.
    This turns one hard ``span -> 1`` reconstruction into ``depth`` easy
    sub-steps (TinyBERT-style layer-wise distillation for span replacement).

    Returns the history of per-step losses.
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    replacement.to(device).train()

    opt = torch.optim.AdamW(
        (p for p in replacement.parameters() if p.requires_grad),
        lr=cfg.stage1_lr,
        weight_decay=cfg.weight_decay,
    )

    history = []
    step = 0
    for epoch in range(cfg.stage1_epochs):
        for i, batch in enumerate(dataloader):
            if cfg.stage1_max_batches is not None and i >= cfg.stage1_max_batches:
                break
            inputs = _batch_inputs(batch, device)
            with torch.no_grad():
                hs = teacher(**inputs, output_hidden_states=True).hidden_states
            h_in = hs[start]
            ext = _extended_attention_mask(inputs.get("attention_mask"), h_in.dtype)

            pred = replacement(h_in, attention_mask=ext)
            sub_outs = getattr(replacement, "anchors", None)
            if anchors and sub_outs:
                # anchored: each sub-block output matches its own teacher depth
                loss = sum(
                    hidden_state_loss(o, hs[idx], cfg.hidden_mse_weight, cfg.hidden_cos_weight)
                    for idx, o in zip(anchors, sub_outs)
                ) / len(sub_outs)
            else:
                loss = hidden_state_loss(
                    pred, hs[teacher_index], cfg.hidden_mse_weight, cfg.hidden_cos_weight
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

            history.append(loss.item())
            if step % log_every == 0:
                print(f"  [stage1] epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1
    return history


# --------------------------------------------------------------------------- #
# Stage 2: end-to-end self-healing of the spliced student
# --------------------------------------------------------------------------- #
def set_stage2_trainable(student, freeze_kept: bool) -> int:
    """Configure which student params heal in Stage 2; return trainable count.

    ``freeze_kept=True`` keeps the inherited encoder blocks frozen and heals only
    the inserted module + classifier head (cheapest, cleanest attribution).
    ``freeze_kept=False`` lets the whole student adapt around the new module.
    """
    from .surgery import _ReplacementLayer, get_encoder_layers

    for p in student.parameters():
        p.requires_grad = True
    if freeze_kept:
        for layer in get_encoder_layers(student):
            if not isinstance(layer, _ReplacementLayer):
                for p in layer.parameters():
                    p.requires_grad = False
    return sum(p.numel() for p in student.parameters() if p.requires_grad)


def run_stage2(student, teacher, dataloader, cfg: TrainConfig, device, log_every: int = 50,
               hidden_map: list | None = None):
    """End-to-end fine-tune the spliced student on the task to *heal* the splice.

    Loss = ``(1 - kd_alpha) * CE(student, y) + kd_alpha * KD(student, teacher)``
    (DistilBERT-style soft-target distillation against the full model's logits).

    **Full feature distillation (our addition).** If ``cfg.stage2_hidden_weight>0``
    and ``hidden_map`` is given, we additionally match the student's hidden states
    to the teacher's at aligned layers (TinyBERT/MiniLM-style representation
    transfer). ``hidden_map[j]`` is the teacher ``hidden_states`` index that
    student ``hidden_states[j]`` should mimic. This transfers far more signal than
    logits alone and is the key to recovering quality at high compression.

    Uses autocast/GradScaler on CUDA so it stays within the 8 GB budget.
    Returns the per-step loss history.
    """
    student.to(device).train()
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    set_stage2_trainable(student, cfg.freeze_kept_in_stage2)
    opt = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    use_amp = bool(cfg.amp) and getattr(device, "type", None) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Linear warmup + linear decay LR schedule (standard BERT fine-tune; stabilizes
    # the heal and reduces seed variance vs. a constant LR).
    per_epoch = len(dataloader) if cfg.stage2_max_batches is None else min(len(dataloader), cfg.stage2_max_batches)
    total_steps = max(1, cfg.stage2_epochs * per_epoch)
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def _lr_lambda(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    history = []
    step = 0
    for epoch in range(cfg.stage2_epochs):
        for i, batch in enumerate(dataloader):
            if cfg.stage2_max_batches is not None and i >= cfg.stage2_max_batches:
                break
            inputs = _batch_inputs(batch, device)
            labels = batch["label"].to(device)

            want_hidden = cfg.stage2_hidden_weight > 0 and hidden_map is not None
            with torch.autocast("cuda", enabled=use_amp):
                s_out = student(**inputs, output_hidden_states=want_hidden)
                logits = s_out.logits
                loss = F.cross_entropy(logits, labels)
                if cfg.kd_alpha > 0 or want_hidden:
                    with torch.no_grad():
                        t_out = teacher(**inputs, output_hidden_states=want_hidden)
                    if cfg.kd_alpha > 0:
                        kd = kd_logit_loss(logits, t_out.logits, cfg.kd_temp)
                        loss = (1.0 - cfg.kd_alpha) * loss + cfg.kd_alpha * kd
                    if want_hidden:
                        s_hs, t_hs = s_out.hidden_states, t_out.hidden_states
                        # robust pairing: anchor from the final hidden state (most
                        # important) so a module that emits one fewer hidden state
                        # than expected can't cause an index error.
                        k = min(len(s_hs), len(hidden_map))
                        pairs = [(s_hs[-(i + 1)], t_hs[hidden_map[-(i + 1)]]) for i in range(k)
                                 if hidden_map[-(i + 1)] < len(t_hs)]
                        hloss = sum(hidden_state_loss(s, t, cfg.hidden_mse_weight, cfg.hidden_cos_weight)
                                    for s, t in pairs) / max(1, len(pairs))
                        loss = loss + cfg.stage2_hidden_weight * hloss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            history.append(loss.item())
            if step % log_every == 0:
                print(f"  [stage2] epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1
    return history
