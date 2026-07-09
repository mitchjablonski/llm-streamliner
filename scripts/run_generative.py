"""Generative-LLM co-compression: does the width+depth principle hold on a decoder
LM measured by perplexity (not just GLUE classification)?

Pipeline on GPT-2 (12 layers, d=768): remove a middle span of decoder blocks,
splice ONE GPT2Block (warm-started from a removed block), optionally prune the
KEPT blocks' FFN width (Conv1D), then heal on WikiText-2 with LM loss + logit-KD
vs the frozen teacher. Reports WikiText-2 perplexity and retention
(teacher_ppl / compressed_ppl, in (0,1]; higher = better).

    uv run python scripts/run_generative.py --tag jun17 --remove 8 --prune_ffn 1024
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F
from src import report
from src.arch import get_arch, reassign_layer_idx
from src.utils import count_total, get_device, set_seed


def _kd(logits, t_logits, T: float = 2.0, topk: int = 0, weight=None, reverse=False, standardize=False, focus=0.0):
    """Per-token KD (flatten to tokens x vocab). Options:
    - topk>0: TOP-K KD (drop the noisy ~50k-logit tail).
    - weight: per-token hard-example weights.
    - reverse: REVERSE/mode-seeking KL = KL(student||teacher) (MiniLLM 2306.08543) — student commits to
      the teacher's major modes instead of spreading over the tail. Better for generation/calibration.
    - standardize: z-score logits per token before KL (2403.01427) — match the teacher's RELATIONAL
      structure, not magnitude; helps most at large teacher-student capacity gaps (our regime).
    """
    V = logits.size(-1)
    sl = logits.float().reshape(-1, V)
    tl = t_logits.float().reshape(-1, V)
    if standardize:
        sl = (sl - sl.mean(-1, keepdim=True)) / (sl.std(-1, keepdim=True) + 1e-6)
        tl = (tl - tl.mean(-1, keepdim=True)) / (tl.std(-1, keepdim=True) + 1e-6)
    sl = sl / T; tl = tl / T
    if reverse:                                              # KL(student || teacher), mode-seeking
        ls = F.log_softmax(sl, -1); ps = ls.exp(); lt = F.log_softmax(tl, -1)
        per_tok = (ps * (ls - lt)).sum(-1)
        if weight is not None:
            w = weight.reshape(-1).to(per_tok.dtype); return (per_tok * w).sum() / (w.sum() + 1e-8) * (T * T)
        return per_tok.mean() * (T * T)
    if topk and topk < V:
        tv, ti = tl.topk(topk, dim=-1)                       # teacher's top-K (per token) — memory-light
        t_top = F.softmax(tv, dim=-1)                        # renormalized over top-K
        s_top = torch.gather(F.log_softmax(sl, dim=-1), 1, ti)
        per_tok = (t_top * (t_top.clamp_min(1e-8).log() - s_top)).sum(-1)
    elif weight is not None or focus > 0:                    # full-vocab, per-token (hard-example wts / focus)
        per_tok = F.kl_div(F.log_softmax(sl, -1), F.softmax(tl, -1), reduction="none").sum(-1)
    else:                                                    # full-vocab, fused/efficient (default path)
        return F.kl_div(F.log_softmax(sl, -1), F.softmax(tl, -1), reduction="batchmean") * (T * T)
    if focus > 0:                                            # FOCUSED HEAL: spend the gradient budget on the top-F
        k = max(1, int(focus * per_tok.numel()))             # most-divergent tokens (damage), not the already-correct ones
        return per_tok.topk(k).values.mean() * (T * T)
    if weight is not None:
        w = weight.reshape(-1).to(per_tok.dtype)
        return (per_tok * w).sum() / (w.sum() + 1e-8) * (T * T)
    return per_tok.mean() * (T * T)


@torch.no_grad()
def _disagreement(logits, t_logits):
    """Per-token disagreement (teacher-student symmetric divergence proxy) -> hard-example weights.
    High where pruning BROKE the function; ~0 where the student already matches the teacher. Normalized
    to mean 1 so it reweights (not rescales) the loss. This puts cheap data's gradient where it's needed."""
    V = logits.size(-1)
    s = F.softmax(logits.float().reshape(-1, V), -1)
    t = F.softmax(t_logits.float().reshape(-1, V), -1)
    d = 0.5 * ((t * (t.clamp_min(1e-8).log() - s.clamp_min(1e-8).log())).sum(-1)
               + (s * (s.clamp_min(1e-8).log() - t.clamp_min(1e-8).log())).sum(-1))
    return (d / (d.mean() + 1e-8)).clamp(0.1, 10.0)          # mean-1, clipped


def _feat(a, b):
    """Scale-robust hidden-state matching: per-token unit-norm then MSE (~cosine distance)."""
    a, b = a.float(), b.float()
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
    return F.mse_loss(a, b)


def _dkd(s_logits, t_logits, labels, alpha=1.0, beta=2.0, T=2.0):
    """Decoupled KD (Zhao 2203.08679): split KD into TARGET-class KD (binary {target, rest}) + NON-target
    KD (KL over the other classes). Standard KD couples them so NCKD (the 'dark knowledge') is suppressed
    by teacher confidence; DKD decouples with weight beta to boost it. y=-100 tokens ignored."""
    V = s_logits.size(-1)
    s = s_logits.reshape(-1, V).float(); t = t_logits.reshape(-1, V).float(); y = labels.reshape(-1)
    m = y != -100
    if m.sum() == 0:
        return s.sum() * 0.0
    s, t, y = s[m], t[m], y[m]
    idx = torch.arange(s.size(0), device=s.device)
    ls = F.log_softmax(s / T, -1); lt = F.log_softmax(t / T, -1)
    ps = ls.exp(); pt = lt.exp()
    # TCKD: binary {p_target, 1 - p_target}
    pt_t = pt[idx, y].clamp(1e-6, 1 - 1e-6); ps_t = ps[idx, y].clamp(1e-6, 1 - 1e-6)
    tckd = (pt_t * (pt_t.log() - ps_t.log()) + (1 - pt_t) * ((1 - pt_t).log() - (1 - ps_t).log())).mean()
    # NCKD: KL over non-target classes (mask target, renormalize via log-softmax)
    nan = torch.full_like(s, float("-inf")); mask = torch.zeros_like(s, dtype=torch.bool); mask[idx, y] = True
    ls_nt = F.log_softmax(s.masked_fill(mask, -1e4) / T, -1)
    pt_nt = F.softmax(t.masked_fill(mask, -1e4) / T, -1)
    nckd = (pt_nt * (pt_nt.clamp_min(1e-8).log() - ls_nt)).sum(-1).mean()
    return (alpha * tckd + beta * nckd) * (T * T)


def _jsd(s_logits, t_logits, beta=0.5, T=1.0, weight=None):
    """Generalized Jensen-Shannon divergence (GKD 2306.13649): symmetric, interpolated KL via mixture
    m = beta*p_teacher + (1-beta)*p_student. beta=0.5 = standard JSD; ->1 approaches forward-KL,
    ->0 approaches reverse-KL. Bounded + symmetric -> stable as the on-policy GKD objective."""
    V = s_logits.size(-1)
    sl = (s_logits / T).float().reshape(-1, V); tl = (t_logits / T).float().reshape(-1, V)
    ls = F.log_softmax(sl, -1); lt = F.log_softmax(tl, -1)
    ps, pt = ls.exp(), lt.exp()
    m = (beta * pt + (1.0 - beta) * ps).clamp_min(1e-9); lm = m.log()
    per = beta * (pt * (lt - lm)).sum(-1) + (1.0 - beta) * (ps * (ls - lm)).sum(-1)
    if weight is not None:
        wv = weight.reshape(-1); return (per * wv).sum() / (wv.sum() + 1e-8) * (T * T)
    return per.mean() * (T * T)


def _lillama(a, b):
    """Lillama local-distillation loss (2412.16719 Eq.7): L1 + log-sigmoid-cosine. Beats MSE for local
    feature matching — L1 is outlier-robust, the cosine term aligns direction. For the frozen-block recipe."""
    a, b = a.float(), b.float()
    l1 = (a - b).abs().mean()
    cos = F.cosine_similarity(a, b, dim=-1).mean()
    return l1 - F.logsigmoid(cos)


def _rel(a, b):
    """RELATION-matrix distillation (SP-KD / MiniLMv2): match the token-token similarity matrix instead
    of the hidden state directly. G = rownorm(H Hᵀ) is (seq×seq) — DIMENSION-AGNOSTIC, so it transfers
    the teacher's representation GEOMETRY even when student/teacher widths differ (no projector needed).
    Exactly the signal that survives width factorization, where hidden-MSE needs a fragile projection."""
    a, b = a.float(), b.float()
    Ga = torch.bmm(a, a.transpose(1, 2))                 # (B, L, L) token-token similarity
    Gb = torch.bmm(b, b.transpose(1, 2))
    Ga = F.normalize(Ga, p=2, dim=-1); Gb = F.normalize(Gb, p=2, dim=-1)   # row-normalize (SP-KD)
    return F.mse_loss(Ga, Gb)


def _attn_kl(s_attn, t_attn):
    """KL between student & teacher attention prob maps (batch,heads,q,k): sum over keys, mean else."""
    s = s_attn.float().clamp_min(1e-8).log()
    t = t_attn.float().clamp_min(1e-8)
    return (t * (t.clamp_min(1e-8).log() - s)).sum(-1).mean()


def build_calib(tok, mode, device, max_tokens=2048, teacher=None):
    """Calibration tokens for ALL importance MEASUREMENT (Block Influence, CBA FFN/residual). The
    selection step is input-distribution-limited: prune by prose-only stats and you may delete
    code/math-critical structure (unrecoverable). Modes:
      general  — WikiText only (default; reproduces prior runs)
      teacher  — WikiText anchor + TEACHER-GENERATED text across code/math/general seeds. Captures the
                 teacher's actual capability distribution (what we want to preserve), no gated datasets."""
    from datasets import load_dataset
    if mode == "instruct":
        # INSTRUCTION calibration (ReplaceMe ablation: SlimOrca >> web by up to +9pt on the closed-form fit —
        # the analytic solve is highly calibration-distribution-sensitive). Concatenate SlimOrca conversations.
        ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
        parts, nchar = [], 0
        for ex in ds:
            convo = ex.get("conversations") or []
            parts.append("\n".join(c.get("value", "") for c in convo))
            nchar += len(parts[-1])
            if nchar > 400000:
                break
        text = "\n\n".join(parts)
        ids = tok(text, return_tensors="pt").input_ids[:, :max_tokens].to(device)
        print(f"[calib] mode=instruct (SlimOrca), {ids.size(1)} tokens")
        return ids
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:200])
    n_dom = 1
    if mode == "teacher" and teacher is not None:
        seeds = ["def ", "import numpy as np\n", "class ", "SELECT * FROM",            # code
                 "Question: If a train travels", "Solve for x: 2x + 3 =", "The derivative of",  # math
                 "In summary,", "The history of", "According to recent research"]      # general
        gen = []
        teacher.config.use_cache = True
        for sd in seeds:
            sid = tok(sd, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = teacher.generate(sid, max_new_tokens=160, do_sample=True, temperature=0.9,
                                       top_p=0.95, pad_token_id=tok.eos_token_id or 0)
            gen.append(tok.decode(out[0], skip_special_tokens=True))
        teacher.config.use_cache = False
        text = text[:5000] + "\n\n" + "\n\n".join(gen)                                  # wiki anchor + teacher
        n_dom = 1 + len(seeds)
    ids = tok(text, return_tensors="pt").input_ids[:, :max_tokens].to(device)
    print(f"[calib] mode={mode}, {ids.size(1)} tokens" + (f" (wiki + teacher-gen x{n_dom-1} seeds)" if mode == "teacher" else ""))
    return ids


def perplexity(model, enc, device, stride=512, max_len=1024):
    """Sliding-window perplexity over a long tokenized sequence."""
    model.eval()
    ids = enc.to(device)
    nll, ntok = 0.0, 0
    for begin in range(0, ids.size(1), stride):
        end = min(begin + max_len, ids.size(1))
        inp = ids[:, begin:end]
        tgt = inp.clone()
        trg_len = end - (begin + stride) if begin + max_len < ids.size(1) else end - begin
        tgt[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(inp, labels=tgt, use_cache=False)
        n = (tgt != -100).sum().item()
        nll += out.loss.item() * n; ntok += n
        if end == ids.size(1):
            break
    return math.exp(nll / max(ntok, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gen")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--remove", type=int, default=8)       # blocks to remove
    ap.add_argument("--hetero_frac", type=float, default=None,
                    help="GREEDY HETEROGENEOUS allocation: target fraction of params to remove by greedily picking "
                         "the cheapest reductions across (delete-layer | narrow-FFN) per cost-per-param. Use with --select hetero")
    ap.add_argument("--select", default="middle", choices=["middle", "bi", "bi_dist", "hetero"],
                    help="which layers to remove: 'middle' (geometric span), 'bi' (lowest-BI contiguous "
                         "window), 'bi_dist' (lowest-BI layers DISTRIBUTED anywhere) — all protect outer layers")
    ap.add_argument("--replace_per_gap", type=int, default=1,
                    help="for bi_dist: replacement blocks to splice at EACH removed cluster (0 = pure "
                         "collapse, 1 = one anchored block per gap)")
    ap.add_argument("--protect_edges", type=int, default=1,
                    help="keep this many outermost layers each side off-limits to BI selection "
                         "(0 = let BI decide freely; the splice still bridges any edge removal)")
    ap.add_argument("--max_insert", type=int, default=-1,
                    help="cap TOTAL replacement blocks injected, allocated to the highest-priority "
                         "(largest) gaps; smaller gaps collapse with no replacement. -1 = one per gap. "
                         "Lets you set net reduction = removed - max_insert precisely.")
    ap.add_argument("--n_replace", type=int, default=1,
                    help="K = number of anchored replacement blocks for the removed span (AMBR). "
                         "K=1 is the vanilla single-block splice; K=2-3 adds capacity, each anchored "
                         "to an intermediate teacher hidden state. Net reduction = (remove - K) layers.")
    ap.add_argument("--recur_replace", type=int, default=1,
                    help="WEIGHT-TIED recurrence: replace the span with ONE block applied this many times "
                         "(same weights). Depth-K expressivity at 1-block params — captures the span's "
                         "composition without adding parameters. Net reduction = remove - 1 layer.")
    ap.add_argument("--hybrid_window", type=int, default=128,
                    help="sliding-window size for --cheap_replace hybrid (the exact-attention span)")
    ap.add_argument("--hybrid_kernel", type=int, default=7,
                    help="conv-kernel for the hybrid's global side-path (larger = longer-range cheap mixing)")
    ap.add_argument("--hybrid_global", choices=["conv", "linattn"], default="conv",
                    help="hybrid global side-path: 'conv' (fixed-kernel) or 'linattn' (causal linear attention, "
                         "unbounded range — the cheap causal form of SPECTRE)")
    ap.add_argument("--subfit", action="store_true",
                    help="SubFit (2606.02559): replace each deleted span with a gated-affine + low-rank "
                         "closed-form map (g⊙x+b+(x−μ)AB) — richer than ReplaceMe's linear T. Training-free.")
    ap.add_argument("--subfit_rank", type=int, default=64,
                    help="low-rank bypass rank for --subfit")
    ap.add_argument("--prune_comp", action="store_true",
                    help="Prune&Comp (2507.18212): training-free diagonal magnitude compensation at the gap "
                         "after pure deletion. Restores the residual-stream scale the removed layers carried.")
    ap.add_argument("--obs_ffn", type=float, default=None,
                    help="local-OBS FFN-width pruning to this fraction (SlimGPT): prune intermediate channels "
                         "+ reconstruct down_proj locally. Concentrate-damage width narrowing (vs global SVD).")
    ap.add_argument("--obs_attn", type=float, default=None,
                    help="local-OBS ATTENTION-head pruning to this head fraction: drop low-importance heads + "
                         "reconstruct o_proj locally (gpt_neox). The attention half of direct-transfer width.")
    ap.add_argument("--cheap_replace", default=None, choices=[None, "conv", "attn", "fusion", "hybrid"],
                    help="replace the span with a CHEAP micro-model: 'conv' (depthwise-conv+FFN), 'attn' "
                         "(conv+light attention+FFN), or 'fusion' (FULL attention path PRESERVED + gated "
                         "additive conv/MLP side-pathways — adds capacity WITHOUT diluting attention).")
    ap.add_argument("--kd_focus", type=float, default=0.0,
                    help="FOCUSED HEAL: compute per-token KL and train on only the top-F fraction most-divergent "
                         "tokens (0=off). Same token budget, gradient budget concentrated on the damage.")
    ap.add_argument("--topk_kd", type=int, default=0,
                    help="TOP-K KD: distill only the teacher's K most-confident logits (tail is noise). "
                         "More data-efficient + higher quality. 0 = full distribution.")
    ap.add_argument("--disagree_weight", action="store_true",
                    help="HARD-EXAMPLE healing: weight per-token loss by teacher-student disagreement, "
                         "focusing cheap data's gradient where pruning BROKE the function.")
    ap.add_argument("--auto_lr", action="store_true",
                    help="nanochat-style: derive the healing LR from the COMPRESSED model's geometry "
                         "(lr scaled by √(ref_hidden/hidden) and √(bs/ref_bs)) instead of a fixed value.")
    ap.add_argument("--mid_feat", action="store_true",
                    help="CKA-informed feature distillation: match the ALIGNED MIDDLE layers (where parent "
                         "& small reps agree), NOT the divergent final hidden. Logit-KD carries the output.")
    ap.add_argument("--rel_distill", action="store_true",
                    help="RELATION-matrix distillation (SP-KD/MiniLMv2): match token-token similarity "
                         "matrices — DIMENSION-AGNOSTIC, transfers geometry across width-factorized layers "
                         "with no projector. The KD-research #1 pick for factorized students.")
    ap.add_argument("--rel_weight", type=float, default=1.0, help="weight on the relation-distillation term")
    ap.add_argument("--dkd", action="store_true",
                    help="Decoupled KD (split target/non-target class KL) instead of plain logit-KL — "
                         "boosts dark-knowledge transfer via --dkd_beta. Cheap ablation.")
    ap.add_argument("--dkd_beta", type=float, default=2.0, help="DKD non-target (dark-knowledge) weight")
    ap.add_argument("--jsd", action="store_true",
                    help="use generalized JSD (GKD) as the KD divergence instead of forward/reverse-KL (symmetric, bounded)")
    ap.add_argument("--jsd_beta", type=float, default=0.5,
                    help="JSD mixture weight on the TEACHER (0.5=standard JSD; ->1 forward-KL, ->0 reverse-KL)")
    ap.add_argument("--gkd", action="store_true",
                    help="ON-POLICY GKD (2306.13649): a fraction of steps train on STUDENT-GENERATED sequences "
                         "(teacher-scored) -> fixes exposure bias. Uses JSD on the on-policy batch, no CE")
    ap.add_argument("--gkd_lambda", type=float, default=0.5,
                    help="fraction of heal steps that are on-policy (student-generated). 0.5 = half on-policy/half corpus")
    ap.add_argument("--gkd_prefix", type=int, default=32,
                    help="prompt-prefix length the student continues when generating on-policy sequences")
    ap.add_argument("--reverse_kl", action="store_true",
                    help="REVERSE/mode-seeking KL = KL(student||teacher) (MiniLLM) instead of forward KL.")
    ap.add_argument("--lillama", action="store_true",
                    help="Lillama frozen-block recipe (with --distill_only): local L1+log-sigmoid-cosine "
                         "feature loss + unfrozen RMSNorm. The cheap frozen-regime upgrade (2412.16719).")
    ap.add_argument("--lora_seam", action="store_true",
                    help="frozen regime: add trainable LoRA on the kept blocks adjacent to the gap (every "
                         "conversion paper found fully-frozen hurts). Raises the frozen ceiling cheaply.")
    ap.add_argument("--lora_rank", type=int, default=16, help="LoRA rank for --lora_seam")
    ap.add_argument("--lora_span", type=int, default=1, help="how many kept blocks each side of a gap to LoRA")
    ap.add_argument("--logit_std", action="store_true",
                    help="logit standardization: z-score logits before KL (2403.01427) — match relational "
                         "structure not magnitude; biggest at large capacity gaps.")
    ap.add_argument("--slice_after", action="store_true",
                    help="run the residual reshape (--slice_dim) AFTER deletion+ReplaceMe (native-replication combo: "
                         "delete toward native depth, then narrow toward native width). Composes; per-layer-first would break the gap-fill")
    ap.add_argument("--slice_learn", action="store_true",
                    help="LEARNED PROJECTION: with --slice_perlayer, unfreeze the per-stage re-base adapters so the heal "
                         "optimizes the inter-stage transforms directly (Fisher-initialized, loss-optimized) — the transform endpoint")
    ap.add_argument("--slice_perlayer", action="store_true",
                    help="PER-LAYER SliceGPT: each layer gets its OWN Fisher basis (per-stage subspaces ~140-dim differ) "
                         "+ re-base adapters -> far more aggressive width compression than one global basis")
    ap.add_argument("--slice_solver", default="pca",
                    choices=["pca", "saliency", "fisher", "random", "outlier", "outlier_fisher", "outlier_saliency"],
                    help="SliceGPT rotation solver: pca (variance) | saliency (act*grad) | fisher (loss-gradient eigenbasis) "
                         "| random (control) | outlier (keep massive-activation dims exact + PCA the rest)")
    ap.add_argument("--slice_importance", action="store_true",
                    help="with --slice_dim: rank residual directions by gradient SALIENCY (Flex-KD), not "
                         "pure PCA variance — keeps the discriminative variance-tail (VkD caution).")
    ap.add_argument("--svd_frac", type=float, default=None,
                    help="RANK axis: activation-aware SVD-factorize kept FFN weights to this fraction of full "
                         "rank (e.g. 0.5). PRESERVES the parent's learned directions in fewer params (vs "
                         "deletion). The compression axis we were missing; combine with depth/width pruning.")
    ap.add_argument("--svd_attn", action="store_true",
                    help="also SVD-factorize attention q/k/v/o projections (with --svd_frac), not just FFN.")
    ap.add_argument("--svd_embed", type=float, default=None,
                    help="factorize the input embedding (+ untied lm_head) to this rank fraction — the "
                         "largest matrix in a small model. ALBERT-style; preserves parent token directions.")
    ap.add_argument("--eora_frac", type=float, default=0.0,
                    help="EoRA error-compensation path: add a trainable low-rank repair path (this fraction "
                         "of full rank) initialized from the compression-error eigenspace. The validated "
                         "'learn the delta' move (arXiv 2410.21271) — best paired with an aggressive svd_frac.")
    ap.add_argument("--eora_only", action="store_true",
                    help="FROZEN backbone: train ONLY the EoRA repair paths (+ norms). Cheap-at-scale repair "
                         "of distributed factorization damage; pair with --svd_frac + --eora_frac.")
    ap.add_argument("--teacher_correct", type=int, default=0,
                    help="TEACHER CORRECTION: fine-tune the parent this many steps on the heal corpus BEFORE "
                         "distilling, so the KD target matches our data distribution (near-free, Minitron-style).")
    ap.add_argument("--teacher_correct_lr", type=float, default=5e-6,
                    help="LR for teacher correction (small — light adaptation, not retraining).")
    ap.add_argument("--slice_dim", type=int, default=None,
                    help="SliceGPT-style: rotate residual into activation-PCA basis then slice to this many "
                         "PRINCIPAL dims (Llama/Qwen RMSNorm). Inherits parent's dominant subspace vs raw channels.")
    ap.add_argument("--replaceme", action="store_true",
                    help="ReplaceMe (arXiv 2505.02819): replace each deleted span with a CLOSED-FORM linear "
                         "transform fit from the teacher's pre/post-span activations. ZERO training, stacks on "
                         "deletion. The analytic 'scaffold absorbs the deletion'. Use with --remove; eval at 0 heal.")
    ap.add_argument("--replaceme_cosine", action="store_true",
                    help="refine the ReplaceMe T on the COSINE objective (Adam) vs L2 closed-form (+1.2 acc).")
    ap.add_argument("--replaceme_fuse", action="store_true",
                    help="FUSE ReplaceMe's analytic linear span-map with a trainable block: h@T + gate·(block "
                         "residual). Linear part solved analytically; block learns only the nonlinear residual.")
    ap.add_argument("--attn_res", action="store_true",
                    help="gated ATTENTION-RESIDUAL retrofit (Moonshot-style, init 0 -> identity): each "
                         "layer learns a softmax-weighted pull from earlier layer outputs, healed jointly. "
                         "Rebalances the residual stream that layer-removal disrupted (topology, not content).")
    ap.add_argument("--prune_ffn", type=int, default=None)  # kept-block FFN width (None=off)
    ap.add_argument("--hybrid_ffn_keep", type=int, default=None,
                    help="CAPTURE-not-DROP FFN: keep this many top intermediate channels EXACT (by activation energy) + "
                         "represent the remaining low-rank (--hybrid_ffn_rank). The residual add-channels recipe ported to FFN.")
    ap.add_argument("--hybrid_ffn_rank", type=int, default=256,
                    help="rank for the captured-remaining FFN path (with --hybrid_ffn_keep)")
    ap.add_argument("--hybrid_eora_rank", type=int, default=0, help="EoRA error-compensation rank on the hybrid captured-remaining (added +8pt for SVD-capture)")
    ap.add_argument("--canon_mix", type=int, default=0,
                    help="NET-NEW cheap compute: attach causal depthwise-conv token mixers (kernel K) to every layer, "
                         "zero-init gated (no-op at surgery). ~K*H params/layer; the heal grows them. 0=off.")
    ap.add_argument("--heal_hard", type=float, default=0.0,
                    help="HARD-EXAMPLE heal: stream ~2x C4, keep only the top-F fraction of 512-token chunks by "
                         "TEACHER loss (intrinsically hard text). Same heal_tokens; harder task. 0=off.")
    ap.add_argument("--mtp_horizon", type=int, default=2,
                    help="MTP lookahead horizon: aux CE at offsets 2..H (0.5^k decay) with --mtp_weight. Deeper "
                         "lookahead trains long-context integration (the most-damaged capability).")
    ap.add_argument("--mtp_kd", action="store_true",
                    help="MTP-KD: the t+k aux targets are the TEACHER's soft distributions (its logits at position "
                         "i+k-1) instead of one-hot corpus tokens - denser signal, zero extra compute.")
    ap.add_argument("--mtp_weight", type=float, default=0.0,
                    help="MTP auxiliary heal objective: same head also predicts t+2 (aux CE, this weight). ZERO eval "
                         "params; densifies supervision per token (attacks the information bound).")
    ap.add_argument("--layer_tie", type=int, default=0,
                    help="LAYER TYING (MobileLLM repeat): the N least-important layers SHARE their predecessor's "
                         "params (unique params == deletion) but keep executing - full depth, zero weights deleted.")
    ap.add_argument("--bus_width", type=int, default=None,
                    help="FULL-WIDTH-BUS reallocation: residual stays full H; each layer reads/writes its OWN "
                         "top-W channels (per-layer energy). Same body params as --prune_hidden W, but no channel "
                         "globally deleted. Do NOT combine with --prune_hidden.")
    ap.add_argument("--layer_merge", type=int, default=0,
                    help="LAYER-MERGE (LaCo-style) depth operator: fold this many least-important interior layers "
                         "into their predecessors (parameter interpolation) instead of deleting. Same params as deletion.")
    ap.add_argument("--merge_alpha", type=float, default=0.5,
                    help="interpolation weight of the MERGED layer (0.5=average; lower = bias toward the kept layer)")
    ap.add_argument("--depth_capture_rank", type=int, default=0,
                    help="DEPTH capture-not-drop: replace --depth_capture_n least-important blocks with rank-R trainable "
                         "GhostBlock mini-FFNs (heal-trained) instead of deleting+linear-splicing. Lowers the depth axis cost.")
    ap.add_argument("--depth_capture_n", type=int, default=4, help="how many blocks to ghost (depth capture-not-drop).")
    ap.add_argument("--capture_sequential", action="store_true",
                    help="SEQUENTIAL/GPTQ capture: step layer-by-layer, each captured on the DRIFTED compressed-upstream input "
                         "(cross-layer-aware). Uses --hybrid_ffn_keep/_rank + --hybrid_attn_keep/_rank. Combine with --hybrid_cov.")
    ap.add_argument("--hybrid_cov", action="store_true",
                    help="RMSNorm-aware capture: whiten the HybridFFN/HybridAttn residual-readers (gate/up/q) by the FULL input covariance (SVD-LLM) instead of a diagonal scale -> captures post-RMSNorm-gain correlations.")
    ap.add_argument("--hybrid_out_aware", action="store_true",
                    help="OUTPUT-AWARE (RRR) capture: weight the HybridFFN/HybridAttn residual-writer factorization (down_proj, "
                         "o_proj) by output energy, not just input variance. The residual capture winner, ported to FFN/attn.")
    ap.add_argument("--hybrid_attn_keep", type=int, default=None,
                    help="CAPTURE-not-DROP attention: keep this many top query heads EXACT (wanda) + remaining heads q/o low-rank (--hybrid_attn_rank)")
    ap.add_argument("--hybrid_attn_rank", type=int, default=256, help="rank for captured-remaining attention heads")
    ap.add_argument("--eora_rank", type=int, default=0,
                    help="EoRA-on-top: add trainable rank-R low-rank paths to each down_proj after surgery (init in the "
                         "input-activation eigenspace, no-op at init, heal-trained). Tests low-rank compensation beyond OBS.")
    ap.add_argument("--prune_heads", type=int, default=None,
                    help="prune kept blocks' attention to this many heads (gpt_neox) — orthogonal "
                         "structural axis (KV/attention) that stacks on depth + FFN width")
    ap.add_argument("--heal_teacher", default=None,
                    help="STAGED: distill the full-heal toward THIS model via logit-KD (e.g. the original full model), "
                         "instead of the --load_model phase-1 model — so each stage chases the original, not the prev degraded stage")
    ap.add_argument("--save_model_path", default=None,
                    help="STAGED: after heal, torch.save the (narrowed) model object for a phase-2 resume")
    ap.add_argument("--load_model", default=None,
                    help="STAGED phase-2: resume student+teacher from a torch.saved phase-1 model (use --model only for tokenizer)")
    ap.add_argument("--prune_hidden_after", action="store_true",
                    help="run --prune_hidden AFTER deletion+ReplaceMe (multi-axis): ReplaceMe builds its transform at "
                         "full width first, then the residual is narrowed -> composes (else dim-mismatch teacher 896 vs student 640)")
    ap.add_argument("--prune_hidden", type=int, default=None,
                    help="prune the RESIDUAL hidden_size to this width (Minitron-style 4th axis; "
                         "Llama-family) — reshapes toward a narrow small-model geometry, stacks on all else")
    ap.add_argument("--heads_per_group", type=int, default=None,
                    help="GQA query-head pruning: keep this many query heads per KV group (uniform, valid GQA). "
                         "Trims over-sized attention to match a narrowed residual. Llama-family")
    ap.add_argument("--head_metric", default="wanda", choices=["norm", "energy", "wanda"],
                    help="head importance: norm(|W|, old gqa) | energy(activation) | wanda(act*|W|). Width sweep predicts wanda/energy win")
    ap.add_argument("--addchan_ortho", action="store_true",
                    help="ADD-CHANNELS orthogonal mode: capture the dropped subspace CONDITIONED on kept (Schur complement) "
                         "-> the r channels carry info INDEPENDENT of the kept channels (no redundancy). Use with --add_lowrank")
    ap.add_argument("--addchan_wanda", action="store_true",
                    help="ADD-CHANNELS wanda mode: weight dropped channels by reader USAGE (act x |W|) before PCA -> capture "
                         "directions that are high-variance AND heavily-read (output-aware). Use with --add_lowrank")
    ap.add_argument("--addchan_pls", action="store_true",
                    help="ADD-CHANNELS PLS (teacher-supervised): dropped directions most PREDICTIVE of the output "
                         "(eigvecs of Sxy Sxy^T, Sxy=dropped^T teacher_hidden). Captures full nonlinear downstream usage. Use with --add_lowrank")
    ap.add_argument("--addchan_rrr", action="store_true",
                    help="ADD-CHANNELS RRR/SVD-LLM mode (VERIFIED-OPTIMAL): capture directions by OUTPUT energy = "
                         "un-whiten[top-r eigvecs of C^1/2 G C^1/2], G=reader Gram. Provably minimizes the output error "
                         "the heal will learn. Beats PCA/Wanda (full off-diag Gram + un-whitening). Use with --add_lowrank")
    ap.add_argument("--addchan_renorm", action="store_true",
                    help="ADD-CHANNELS renorm: balance captured-channel residual scale to kept (function-preserving: writer*s, reader/s) "
                         "-> decouple the captured block from the shared RMSNorm denominator. Use with --add_lowrank")
    ap.add_argument("--addchan_combine3", action="store_true",
                    help="ADD-CHANNELS 3-way diversity: PCA(variance) + orthogonal(independent) + OUTLIER(heavy-tail/sink) directions, "
                         "union orthonormalized. Adds the outlier criterion to the combine. Use with --add_lowrank")
    ap.add_argument("--addchan_combine", action="store_true",
                    help="ADD-CHANNELS combine mode: KEEP PCA-of-dropped directions + ADD orthogonal(Schur) directions "
                         "(union, orthonormalized) -> complement PCA rather than replace it. Use with --add_lowrank")
    ap.add_argument("--add_lowrank", type=int, default=0,
                    help="ADD-CHANNELS absorption: keep --prune_hidden channels EXACT (raw-drop, uncorrupted) + ADD this many "
                         "low-rank channels for the dropped subspace. Recovers dropped info WITHOUT corrupting the kept readers "
                         "(unlike fold). New hidden = prune_hidden + add_lowrank. The retention-safe absorption.")
    ap.add_argument("--absorb_sequential", action="store_true",
                    help="SEQUENTIAL (SparseGPT-style) absorption: per-layer fold + iterative re-solve of readers from "
                         "their DRIFTED inputs so narrowing errors don't accumulate down the network. Use with --prune_hidden")
    ap.add_argument("--absorb_seq_iters", type=int, default=2, help="sequential refinement passes (Jacobi iterations)")
    ap.add_argument("--absorb_lossw", action="store_true",
                    help="LOSS-AWARE absorption: weighted-lstsq with per-token weights = teacher CE (loss-critical tokens "
                         "weighted more). Minimizes loss-weighted reconstruction instead of plain MSE. Use with --absorb_perlayer")
    ap.add_argument("--absorb_perlayer", action="store_true",
                    help="PER-LAYER closed-form absorption (the validated winner): each layer folds its OWN dropped->kept "
                         "regression M (attn/MLP separate). Beats raw-drop at ALL compression levels (MSE). Use with --prune_hidden")
    ap.add_argument("--absorb_width", action="store_true",
                    help="CLOSED-FORM ABSORPTION: when width-pruning, fold the dropped channels into the kept ones "
                         "via lstsq (W_kept += W_dropped @ M, M=dropped~kept regression) -> minimize output error at "
                         "compression time, no heal needed (SliceGPT/OBS-style; attacks the destruction at its source)")
    ap.add_argument("--protect_math", type=int, default=0,
                    help="CAPABILITY-AWARE width prune: reserve this many kept channels for those disproportionately "
                         "important to MATH (differential energy on OpenWebMath vs C4), instead of pure C4 energy.")
    ap.add_argument("--prune_sink_pin", type=int, default=0,
                    help="OUTLIER+RAW: force-pin the top-N massive-activation (sink) dims via max-activation, "
                         "fill the rest by energy, NO rotation. Tests outlier's sink-handling + raw's no-rotation")
    ap.add_argument("--add_depth", type=int, default=0,
                    help="ADD this many duplicated trainable layers (toward the deep-narrow optimal aspect "
                         "ratio). Combine with --prune_hidden/--prune_ffn (remove width) to reshape a "
                         "shallow-wide teacher into a deep-narrow small model. Student deeper than teacher "
                         "-> feature-distill auto-off; heal via CE+KD.")
    ap.add_argument("--realloc_adjacent", type=float, default=None,
                    help="NEAR-ZERO-HEAL width->depth: keep this FRACTION of FFN channels in place (exact), "
                         "reallocate --realloc_realloc to a block inserted ADJACENT (right after the layer, "
                         "function-preserving), drop the rest. Conserve-don't-destroy; auto-captures at near-zero heal.")
    ap.add_argument("--realloc_realloc", type=float, default=0.3,
                    help="fraction of FFN channels reallocated to the adjacent depth block (--realloc_adjacent)")
    ap.add_argument("--realloc_ffn_only", action="store_true",
                    help="adjacent realloc blocks carry ONLY the FFN (no dead attention params) — same function, "
                         "fewer params, so deep-narrow-realloc is param-matched to in-place narrowing (clean shape test)")
    ap.add_argument("--realloc_correct", action="store_true",
                    help="apply the lstsq input-shift correction to realloc-adjacent blocks (empirically 12x "
                         "lower zero-heal MSE; re-fits the adjacent down_proj on its actual shifted activation)")
    ap.add_argument("--realloc_depth", type=float, default=None,
                    help="closed-form WIDTH->DEPTH reallocation: narrow each FFN to this keep-fraction (OBS) and "
                         "directly TRANSFER the removed channels' weights into --realloc_n_depth new depth blocks "
                         "(warm-init, not dropped). The aspect-ratio reshape by weight transfer, then heal refines.")
    ap.add_argument("--realloc_n_depth", type=int, default=4,
                    help="number of new depth blocks to reallocate the removed FFN width into (--realloc_depth)")
    ap.add_argument("--add_depth_identity", action="store_true",
                    help="init --add_depth layers as IDENTITY no-ops (LLaMA-Pro zero-init residual) rather than "
                         "duplicates: empty capacity the heal REALLOCATES the removed width's function into.")
    ap.add_argument("--prune_kv_groups", type=int, default=None,
                    help="GQA head pruning (Llama-family): keep this many KV groups (by importance). "
                         "NOTE low-yield axis — attention is a small param fraction vs FFN")
    ap.add_argument("--cba_ffn_energy", type=float, default=None,
                    help="MEASUREMENT-DRIVEN per-layer FFN width: keep each layer's own effective rank "
                         "at this activation-energy threshold (e.g. 0.9) instead of a forced uniform width")
    ap.add_argument("--cba_hidden_energy", type=float, default=None,
                    help="measurement-driven residual width: keep #channels holding this fraction of "
                         "global activation energy (discovers the width instead of forcing it)")
    ap.add_argument("--calib", default="general", choices=["general", "teacher", "instruct"],
                    help="calibration for importance MEASUREMENT (BI/CBA). 'teacher' = WikiText + "
                         "teacher-generated code/math/general, so selection reflects the teacher's full "
                         "capability (a pruning decision is unrecoverable). No gated datasets needed.")
    ap.add_argument("--keep_best", action="store_true",
                    help="replacement-FREE: instead of splicing fresh blocks at a removed cluster, KEEP "
                         "the highest-BI original layers there (uses pretrained weights, nothing new to train)")
    ap.add_argument("--svd_ratio", type=float, default=1.0,
                    help="low-rank factorize kept blocks' Linears to rank=ratio*min(in,out) — a "
                         "SEPARATE parameter-removal axis that STACKS on layer removal (1.0 = off)")
    ap.add_argument("--heal_steps", type=int, default=1500)
    ap.add_argument("--distill_only", action="store_true",
                    help="train ONLY the spliced block via hidden-state distillation "
                         "(frozen backbone, LLM-Streamline-style) — fits large models in 8GB")
    ap.add_argument("--partial_heal", action="store_true",
                    help="heal spliced block + pruned kept blocks + final-norm/head with "
                         "LM+KD loss; freeze embeddings & unpruned blocks (8-bit Adam + "
                         "gradient checkpointing). The recommended regime for billion-param models.")
    ap.add_argument("--full_heal", action="store_true",
                    help="heal the whole compressed student end-to-end (LM+KD).")
    ap.add_argument("--bf16", action="store_true", help="load models in bf16 (large models)")
    ap.add_argument("--fp16", action="store_true", help="load models in fp16 (for large models)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--sdd_seeds", type=int, default=16000,
                    help="number of C4 seed prefixes for self-distill generation (batched+cached; ~4k=15min, "
                         "16k=~1hr). More = more diverse on-policy heal data, generated once and reused.")
    ap.add_argument("--heal_data", default="wikitext2", choices=["wikitext2", "c4", "teacher", "instruct", "mathmix"],
                    help="healing corpus: wikitext2 / streamed c4 / 'teacher' (on-policy generated) / 'instruct' "
                         "(SlimOrca — denser instruction/reasoning signal per token than web; tests the data lever)")
    ap.add_argument("--heal_tokens", type=int, default=20_000_000,
                    help="for --heal_data c4: stream this many tokens for healing")
    ap.add_argument("--drop_distill", action="store_true",
                    help="DROP-SPECIFIC distill (generalized): hidden-distill every CAPTURED component to its teacher counterpart "
                         "- ghost layers to original blocks, OR all hybrid-captured middle layers to teacher (proj for width). "
                         "Explicit local supervision of what was compressed, vs global logit-KD alone.")
    ap.add_argument("--ghost_distill", action="store_true",
                    help="DROP-SPECIFIC distill: locally supervise each GhostBlock to match the teacher's ORIGINAL block "
                         "output at that position (targeted hidden-state distill at the ghosted layers) - directly teaches "
                         "the ghost to mimic the deleted block, vs hoping global logit-KD trains it.")
    ap.add_argument("--feat_distill", action="store_true",
                    help="add hidden-state feature distillation (spliced-exit + final hidden -> teacher)")
    ap.add_argument("--feat_weight", type=float, default=1.0, help="weight on the feature-distill term")
    ap.add_argument("--warm_distill_steps", type=int, default=0,
                    help="two-stage: feature-distill the spliced block (frozen backbone) for up to N steps "
                         "as initialization BEFORE healing (LLM-Streamline/TinyBERT-style)")
    ap.add_argument("--warm_plateau", action="store_true",
                    help="Phase 1 runs until the condenser's feature loss PLATEAUS (converges on the span "
                         "map), not a fixed step count — then Phase 2 (full heal) begins. Your two-phase idea.")
    ap.add_argument("--warm_plateau_patience", type=int, default=300,
                    help="stop Phase 1 after this many steps without EMA-loss improvement")
    ap.add_argument("--warm_plateau_eps", type=float, default=1e-4,
                    help="minimum EMA-loss improvement to count as progress")
    ap.add_argument("--deep_distill", action="store_true",
                    help="match ALL aligned hidden states (teacher every kept+spliced layer), not just "
                         "spliced-exit+final — richer per-token teacher signal, less data-dependent")
    ap.add_argument("--ce_weight", type=float, default=0.5,
                    help="weight on the data CE term; set 0 for PURE teacher distillation (KD+features "
                         "only) to minimize reliance on the healing corpus")
    ap.add_argument("--big_teacher", default=None,
                    help="a LARGER same-family model for logit-KD (e.g. Qwen/Qwen2.5-32B while pruning 7B) "
                         "— pulls the student past its parent. Logit-KD only (shared vocab required).")
    ap.add_argument("--big_teacher_4bit", action="store_true",
                    help="load the big teacher in 4-bit (nf4) to fit large ones (32B/72B) in memory")
    ap.add_argument("--big_kd_weight", type=float, default=0.5,
                    help="fraction of the KD term taken from the big teacher (vs the parent)")
    ap.add_argument("--offload_parent", action="store_true",
                    help="after warm-start + ppl baseline, move the PARENT teacher to CPU and use the "
                         "big teacher alone for KD (frees ~weights GB so a 70B-4bit teacher fits). "
                         "Disables parent feature-distill; the recipe becomes CE + big-teacher logit-KD.")
    ap.add_argument("--kept_lr_mult", type=float, default=1.0,
                    help="LR multiplier for KEPT (pretrained) layers vs the spliced blocks; e.g. 0.1 = "
                         "gently adapt kept layers (avoid forgetting) while the new blocks learn fast")
    ap.add_argument("--batch_size", type=int, default=0,
                    help="sequences per step (0 = auto from model size + GPU memory; H100 fits more)")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (cuts activation memory)")
    ap.add_argument("--attn_distill", action="store_true",
                    help="MiniLM-style: also match the spliced blocks' ATTENTION maps to the teacher's "
                         "anchor-layer attention (KL) — supervises the mechanism, not just hidden states")
    ap.add_argument("--attn_weight", type=float, default=1.0, help="weight on the attention-distill term")
    ap.add_argument("--eval_bs", type=int, default=8,
                    help="lm-eval batch size (lowered from 16; large vocab + post-train memory OOM'd at 16)")
    ap.add_argument("--lm_eval_tasks", default=None,
                    help="comma-sep LM-eval-harness tasks to run on teacher+compressed "
                         "(e.g. lambada_openai,hellaswag,arc_easy,piqa,winogrande)")
    ap.add_argument("--eval_limit", type=int, default=None,
                    help="cap examples/task in lm-eval (e.g. 200 for slow generative tasks like gsm8k). "
                         "Part of the teacher cache key, so limited and full evals cache separately.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    report.reset_peak_vram(device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    dt = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    ld = {"torch_dtype": dt}
    if args.attn_distill:
        ld["attn_implementation"] = "eager"  # needed to return attention weights
    if args.load_model:                          # STAGED phase-2: resume from saved (narrowed) phase-1 model (uniform config round-trips)
        teacher = AutoModelForCausalLM.from_pretrained(args.load_model, **ld).to(device).eval()  # prev-stage = ReplaceMe+heal ref
        model = AutoModelForCausalLM.from_pretrained(args.load_model, **ld).to(device)
        print(f"[staged] loaded phase-1 model from {args.load_model} (hidden={model.config.hidden_size})")
    else:
        teacher = AutoModelForCausalLM.from_pretrained(args.model, **ld).to(device).eval()
        model = AutoModelForCausalLM.from_pretrained(args.model, **ld).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    heal_teacher = None
    if args.heal_teacher:                            # STAGED: a SEPARATE full-width model the heal distills toward (logit-KD)
        heal_teacher = AutoModelForCausalLM.from_pretrained(args.heal_teacher, **ld).to(device).eval()
        for p in heal_teacher.parameters(): p.requires_grad = False
        print(f"[staged] heal_teacher loaded: {args.heal_teacher} (hidden={heal_teacher.config.hidden_size})")
    before = count_total(model)
    arch = get_arch(model)

    # Optional LARGER teacher for logit-KD (decoupled from the parent we warm-start/prune from).
    # Parent provides feature/hidden distillation (dims align); the big teacher pulls the student
    # toward a STRONGER target than its parent -> enables exceeding the parent (Meta did this for
    # Llama-3.2 w/ 8B+70B). Logit-KD only (shared vocab within a family). Optional 4-bit to fit.
    big_teacher = None
    if args.big_teacher:
        bld = {"torch_dtype": torch.bfloat16}
        if args.big_teacher_4bit:
            from transformers import BitsAndBytesConfig
            bld["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            bld["device_map"] = {"": 0}
        big_teacher = AutoModelForCausalLM.from_pretrained(args.big_teacher, **bld).eval()
        if not args.big_teacher_4bit:
            big_teacher = big_teacher.to(device)
        for p in big_teacher.parameters():
            p.requires_grad = False
        if big_teacher.config.vocab_size != model.config.vocab_size:
            # same family, different vocab PADDING (e.g. Qwen2.5-7B 152064 vs 1.5B 151936) — the real
            # token prefix matches, so KD on the shared min-vocab prefix (b_logits sliced at the KD sites).
            assert big_teacher.config.vocab_size >= model.config.vocab_size, \
                "big_teacher vocab smaller than student — not a padding mismatch; check tokenizer family"
            print(f"[big_teacher] vocab {big_teacher.config.vocab_size} vs student {model.config.vocab_size}; "
                  f"KD on shared {model.config.vocab_size}-token prefix")
        print(f"[big-teacher] {args.big_teacher} loaded ({'4bit' if args.big_teacher_4bit else 'bf16'}) "
              f"for logit-KD weight {args.big_kd_weight}")

    # --- residual hidden-width pruning (Minitron-style 4th axis) — BEFORE depth surgery so spliced
    #     blocks are built at the new hidden_size. Reshapes toward a narrow small-model geometry. ---
    model.config._prune_sink_pin = args.prune_sink_pin   # OUTLIER+RAW: sink-pin signal for prune_hidden_width
    if args.protect_math > 0:                            # CAPABILITY-AWARE width selection: reserve channels for math
        _mds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        _mtxt = "\n\n".join(ex["text"] for ex in _mds.take(400))
        model.config._prune_math_calib = tok(_mtxt, return_tensors="pt").input_ids[:, :2048].to(device)
        model.config._prune_math_reserve = args.protect_math
        print(f"[surgery] protect_math: reserving {args.protect_math} width channels by math-differential energy")
    model.config._absorb_width = args.absorb_width       # CLOSED-FORM absorption of dropped channels into kept
    model.config._absorb_lossw = args.absorb_lossw       # LOSS-AWARE: weight the per-layer absorption solve by per-token loss
    model.config._addchan_ortho = args.addchan_ortho     # ADD-CHANNELS: capture dropped subspace orthogonal to kept (Schur)
    model.config._addchan_wanda = args.addchan_wanda     # ADD-CHANNELS: reader-usage-weighted capture
    model.config._addchan_combine = args.addchan_combine # ADD-CHANNELS: PCA + orthogonal complement
    model.config._addchan_combine3 = args.addchan_combine3  # ADD-CHANNELS: PCA + orthogonal + outlier (3-way diversity)
    model.config._addchan_renorm = args.addchan_renorm   # ADD-CHANNELS: balance captured residual scale (decouple shared RMS)
    model.config._addchan_rrr = args.addchan_rrr         # ADD-CHANNELS: RRR/SVD-LLM output-optimal whitened directions
    model.config._hybrid_out_aware = args.hybrid_out_aware  # HYBRID: output-aware (RRR) capture for residual-writers
    model.config._hybrid_cov = args.hybrid_cov            # HYBRID: full-covariance (SVD-LLM, RMSNorm-aware) on readers
    model.config._hybrid_eora_rank = args.hybrid_eora_rank  # HYBRID: EoRA error-compensation on captured-remaining
    model.config._addchan_pls = args.addchan_pls         # ADD-CHANNELS: PLS teacher-supervised predictive directions
    def _do_slice():                                     # SliceGPT residual reshape (global or per-layer Fisher)
        cids = build_calib(tok, args.calib, device, teacher=teacher)
        oldH = model.config.hidden_size
        if args.slice_perlayer:                          # PER-LAYER Fisher basis (per-stage subspaces + re-base adapters)
            from src.arch import slicegpt_rotate_perlayer
            slicegpt_rotate_perlayer(model, cids, device, args.slice_dim, solver=args.slice_solver); model.to(device)
            print(f"[surgery] PER-LAYER SliceGPT rotate+slice residual {oldH} -> {args.slice_dim} (per-stage {args.slice_solver} basis + adapters)")
            if args.slice_learn:                         # LEARNED PROJECTION: heal optimizes the per-stage re-base transforms (Fisher-init)
                nlearn = 0
                for mm in model.modules():
                    if type(mm).__name__ == "_RebaseLayer": mm.adapter.requires_grad_(True); nlearn += 1
                print(f"[surgery] learned-projection: {nlearn} per-stage adapters set TRAINABLE (Fisher-init, loss-optimized by heal)")
        else:
            from src.arch import slicegpt_rotate, prune_hidden_width
            slicegpt_rotate(model, cids, device, importance=args.slice_importance, solver=args.slice_solver)
            if prune_hidden_width(model, arch, args.slice_dim, cids, device):
                model.to(device)
                print(f"[surgery] SliceGPT rotate+slice residual {oldH} -> {args.slice_dim} ({args.slice_solver} subspace)")
    if args.slice_dim is not None and not args.slice_after:   # default: reshape BEFORE deletion (standalone width reshape)
        _do_slice()
    if args.prune_hidden is not None and not args.prune_hidden_after:   # default: narrow BEFORE deletion (standalone)
        from src.arch import prune_hidden_width, prune_hidden_perlayer, prune_hidden_sequential, prune_hidden_hybrid
        cids = build_calib(tok, args.calib, device, teacher=teacher)
        oldH = model.config.hidden_size
        ok = (prune_hidden_hybrid(model, arch, args.prune_hidden, args.add_lowrank, cids, device) if args.add_lowrank > 0
              else prune_hidden_sequential(model, arch, args.prune_hidden, cids, device, iters=args.absorb_seq_iters) if args.absorb_sequential
              else prune_hidden_perlayer(model, arch, args.prune_hidden, cids, device) if args.absorb_perlayer
              else prune_hidden_width(model, arch, args.prune_hidden, cids, device))
        if ok:
            model.to(device)
            print(f"[surgery] pruned residual hidden width {oldH} -> {model.config.hidden_size}")

    # --- surgery: remove layers (geometric / BI-contiguous / BI-distributed), splice block(s) ---
    layers = arch.get_layers(model)
    n = len(layers)
    count = args.remove
    K = max(1, args.n_replace)
    mdtype = next(model.parameters()).dtype  # match the loaded checkpoint dtype (Pythia ships fp16)

    # decide WHICH layers to remove (a set of teacher indices), protecting layer 0 and the last
    if args.select in ("bi", "bi_dist", "hetero"):
        calib = build_calib(tok, args.calib, device, teacher=teacher)
        with torch.no_grad():
            hs = teacher(calib, output_hidden_states=True, use_cache=False).hidden_states
        bi = [(1 - F.cosine_similarity(hs[i], hs[i + 1], dim=-1).mean()).item() for i in range(len(hs) - 1)]
        pe = args.protect_edges  # how many outermost layers each side to keep off-limits (0 = free)
        if args.select == "hetero":  # GREEDY heterogeneous: allocate delete vs narrow-FFN per cost-per-param
            from src.arch import greedy_hetero_allocate
            removed_set, _ = greedy_hetero_allocate(model, arch, layers, bi, calib, device,
                                                    args.hetero_frac, protect_edges=max(1, pe))
            count = len(removed_set)                       # the allocator decided the delete count
        elif args.select == "bi":  # lowest-BI CONTIGUOUS window
            lo, hi = pe, n - count - pe  # window [s,s+count) stays within [pe, n-pe]
            s0 = min(range(lo, hi + 1), key=lambda s: sum(bi[s:s + count]))
            removed_set = set(range(s0, s0 + count))
        else:                    # lowest-BI DISTRIBUTED individual layers
            cand = sorted(range(pe, n - pe), key=lambda i: bi[i])[:count]
            removed_set = set(cand)
        print(f"[select] {args.select}: remove layers {sorted(removed_set)} "
              f"BI={[round(bi[i],3) for i in sorted(removed_set)]}")
    else:  # geometric middle (one contiguous span)
        s0 = (n - count) // 2
        removed_set = set(range(s0, s0 + count))

    # find clusters (maximal runs of removed layers); allocate a REPLACEMENT BUDGET to the
    # highest-priority gaps (biggest clusters = most removed function). max_insert caps total
    # injected blocks so net reduction = removed - injected is controllable; small/low-priority
    # gaps get pure collapse (no replacement). This decouples "how much to remove" from "how
    # many blocks to add back".
    clusters = []
    s = None
    for i in range(n + 1):
        if i < n and i in removed_set:
            s = i if s is None else s
        elif s is not None:
            clusters.append((s, i)); s = None
    per_gap = K if args.select in ("middle", "bi") else max(1, args.replace_per_gap)
    if args.max_insert >= 0:
        # rank clusters by size (gap magnitude); top-budget clusters get per_gap blocks, rest collapse
        order = sorted(range(len(clusters)), key=lambda c: -(clusters[c][1] - clusters[c][0]))
        budget, get_repl = args.max_insert, {}
        for c in order:
            k_here = min(per_gap, max(0, budget))
            get_repl[c] = k_here; budget -= k_here
    else:
        get_repl = {c: per_gap for c in range(len(clusters))}  # one-per-gap (default)
    cl_at = {cs: (ci, ce) for ci, (cs, ce) in enumerate(clusters)}

    new_layers, repl_blocks, spliced_anchors, feat_pairs = [], [], [], []
    replaceme_spans = []
    i = 0
    while i < n:
        if i not in removed_set:
            new_layers.append(layers[i])
            feat_pairs.append((len(new_layers), i + 1))   # kept layer output -> teacher th[i+1]
            i += 1
        else:
            ci, j = cl_at[i]  # i is always a cluster start here; ci=index, j=cluster end
            nb = get_repl.get(ci, per_gap)
            if args.replaceme:
                # ReplaceMe: pure-collapse the span here; T is fit + merged into the preceding block's
                # down_proj AFTER the stack is rebuilt (the correct subset-objective method, zero params).
                replaceme_spans.append((i, j - i))
                i = j; continue
            if args.subfit:
                # SubFit (2606.02559): collapse + insert a gated-affine + low-rank closed-form map (richer
                # than ReplaceMe's linear T). Training-free, drop-in.
                from src.arch import subfit_fit
                _scal = build_calib(tok, args.calib, device, max_tokens=8192, teacher=teacher)
                sl = subfit_fit(teacher, arch, i, j - i, _scal, device, rank=args.subfit_rank, dtype=mdtype)
                new_layers.append(sl.to(device))
                print(f"[surgery] SubFit gated-affine+low-rank(r={args.subfit_rank}) at span [{i},{j})")
                i = j; continue
            if args.prune_comp:
                # Prune&Comp (2507.18212): collapse + insert a diagonal MAGNITUDE-compensation scale at
                # the gap (training-free, restores the post-span residual magnitude profile).
                from src.arch import TransformLayer, prune_comp_fit
                _pcal = build_calib(tok, args.calib, device, teacher=teacher)
                Tpc, bpc = prune_comp_fit(teacher, i, j - i, _pcal, device)
                new_layers.append(TransformLayer(Tpc.to(mdtype), bpc.to(mdtype)).to(device))
                print(f"[surgery] Prune&Comp: diagonal magnitude + OFFSET correction at span [{i},{j})")
                i = j; continue
            if args.keep_best and nb > 0:
                # replacement-FREE: keep the nb highest-BI ORIGINAL layers of the cluster (in place),
                # remove the rest. No fresh block to train — uses existing pretrained weights.
                keep_idx = sorted(sorted(range(i, j), key=lambda x: -bi[x])[:nb])
                for x in keep_idx:
                    new_layers.append(layers[x])
                    feat_pairs.append((len(new_layers), x + 1))  # original layer -> its teacher hidden
            elif args.recur_replace > 1 and nb > 0:
                # WEIGHT-TIED RECURRENCE: ONE block applied recur_replace times (same object inserted
                # repeatedly). Depth-K expressivity at 1-block params — captures the span's COMPOSITION
                # of K layers without adding parameters. Anchor each application to an evenly-spaced
                # teacher depth (a recurrent condenser of the span).
                R = args.recur_replace
                src = layers[i + (j - i) // 2]
                b = arch.make_block(model, i).to(device=device, dtype=mdtype)
                b.load_state_dict(src.state_dict())
                repl_blocks.append(b)
                for g in range(R):
                    new_layers.append(b)                       # SAME object -> weight-tied
                    t_anchor = i + int(round((g + 1) * (j - i) / R))
                    spliced_anchors.append((len(new_layers), t_anchor))
                    feat_pairs.append((len(new_layers), t_anchor))
            else:
                for g in range(nb):
                    if args.cheap_replace == "fusion":        # full block (attn preserved) + gated side-paths
                        from src.arch import FusionBlock
                        src = layers[i + (j - i) // 2]
                        base = arch.make_block(model, i + g).to(device=device, dtype=mdtype)
                        base.load_state_dict(src.state_dict())  # warm-start the wrapped block
                        b = FusionBlock(base, model.config.hidden_size, dtype=mdtype).to(device)
                    elif args.cheap_replace == "hybrid":      # sliver-of-exact-attention + cheap side-path
                        from src.arch import HybridBlock
                        src = layers[min(j - 1, i + (j - i) // 2)]
                        inter = getattr(model.config, "intermediate_size", None) or 4 * model.config.hidden_size
                        b = HybridBlock(model.config.hidden_size, n_heads=model.config.num_attention_heads,
                                        window=args.hybrid_window, conv_kernel=args.hybrid_kernel,
                                        intermediate=inter, dtype=mdtype, global_path=args.hybrid_global).to(device)
                        b.warm_init(src, arch.family)         # WARM-START attn+FFN from the parent span (not random)
                    elif args.cheap_replace:                  # CHEAP micro-model condenser (conv[+attn]+FFN)
                        from src.arch import CheapBlock
                        b = CheapBlock(model.config.hidden_size, attn=(args.cheap_replace == "attn"),
                                       dtype=mdtype).to(device)
                    elif args.replaceme_fuse:                 # ReplaceMe analytic-linear backbone + trainable block
                        from src.arch import ReplaceMeFusion, replaceme_fit_full
                        _fcal = build_calib(tok, args.calib, device, max_tokens=8192, teacher=teacher)
                        Tf = replaceme_fit_full(teacher, arch, i, j - i, _fcal, device, dtype=mdtype)  # FULL residual
                        src = layers[min(j - 1, i + int((g + 0.5) * (j - i) / max(1, nb)))]
                        base = arch.make_block(model, i + g).to(device=device, dtype=mdtype)
                        base.load_state_dict(src.state_dict())
                        b = ReplaceMeFusion(base, Tf).to(device)
                        print(f"[surgery] ReplaceMe-fuse: analytic linear + trainable block at span [{i},{j})")
                    else:
                        src = layers[min(j - 1, i + int((g + 0.5) * (j - i) / max(1, nb)))]
                        b = arch.make_block(model, i + g).to(device=device, dtype=mdtype)
                        b.load_state_dict(src.state_dict())   # warm-start from the gap block at this depth
                    new_layers.append(b); repl_blocks.append(b)
                    t_anchor = i + int(round((g + 1) * (j - i) / max(1, nb)))  # teacher hidden in [i, j]
                    spliced_anchors.append((len(new_layers), t_anchor))
                    feat_pairs.append((len(new_layers), t_anchor))
            i = j  # if nb==0 this is pure collapse for this cluster (neighbors absorb it)
    if args.add_depth > 0:
        # ADD DEPTH (toward the deep-narrow optimal aspect ratio): insert duplicated, warm-started,
        # trainable layers at evenly-spaced positions. Combined with aggressive width pruning, this
        # reshapes a wide-shallow teacher toward a narrow-deep small-model geometry. The student becomes
        # DEEPER than the teacher, so feature-distill (1:1 depth map) is off — heal via CE+KD.
        import copy as _copy
        nn0 = len(new_layers)
        pos = sorted({int(round((k + 1) * nn0 / (args.add_depth + 1))) for k in range(args.add_depth)})
        ins = 0
        from src.arch import zero_init_residual
        for p in pos:
            srcL = new_layers[min(p, len(new_layers) - 1)]
            try:
                dup = _copy.deepcopy(srcL).to(device=device, dtype=mdtype)
            except Exception:
                dup = arch.make_block(model, p).to(device=device, dtype=mdtype)
            if args.add_depth_identity:                  # IDENTITY init ("warm-init with nothing", LLaMA-Pro):
                zero_init_residual(dup, arch.family)     # zero the residual-writing projs -> starts as a no-op;
            new_layers.insert(p + ins, dup); repl_blocks.append(dup); ins += 1   # the heal REALLOCATES the
        mode = "identity (no-op, reallocate-into)" if args.add_depth_identity else "duplicated"  # removed width
        print(f"[surgery] add_depth: inserted {ins} {mode} trainable layers (now {len(new_layers)} deep)")
    repl = repl_blocks[0] if repl_blocks else None  # may be empty if max_insert=0 (pure collapse)
    start = min(removed_set) if removed_set else 0  # remove=0 -> native eval (no surgery)
    teacher_index = start + count  # back-compat for distill_only/metrics
    kept = [l for l in new_layers if l not in repl_blocks]
    new = torch.nn.ModuleList(new_layers)
    arch.set_layers(model, new)
    setattr(model.config, arch.n_layer_attr, len(new))
    if getattr(model.config, "layer_types", None) is not None:
        # per-layer config list (Qwen2.5+): resize to the new layer count or the forward IndexErrors
        lt = list(model.config.layer_types)
        base = lt[0] if lt else "full_attention"
        model.config.layer_types = [(lt[k] if k < len(lt) else base) for k in range(len(new))]
    reassign_layer_idx(new)  # KV-cache correctness after the stack shrinks
    pruned_kept = []
    if args.prune_ffn is not None:
        for b in kept:
            if arch.prune_ffn(b, args.prune_ffn):
                pruned_kept.append(b)
        print(f"[surgery] pruned FFN of {len(pruned_kept)} kept {arch.family} blocks to {args.prune_ffn}")
    if args.svd_ratio < 1.0:
        from src.arch import svd_factorize_block
        nf = sum(svd_factorize_block(b, args.svd_ratio) for b in kept)
        pruned_kept = list(kept)  # SVD'd blocks become trainable in partial-heal
        print(f"[surgery] SVD-factorized {nf} linears across kept blocks (rank ratio {args.svd_ratio})")
    if args.cba_ffn_energy is not None or args.cba_hidden_energy is not None:
        # measurement-driven PER-LAYER allocation: keep what each layer needs, discover the total
        from src.arch import cba_allocate
        _cids = build_calib(tok, args.calib, device, max_tokens=1024, teacher=teacher)
        rep = cba_allocate(model, arch, kept, _cids, device,
                           ffn_energy=args.cba_ffn_energy, hidden_energy=args.cba_hidden_energy)
        ws = list(rep["ffn_widths"].values())
        if ws:
            print(f"[CBA] per-layer FFN widths: min={min(ws)} max={max(ws)} mean={sum(ws)//len(ws)} "
                  f"(vs uniform) across {len(ws)} layers")
        if rep["hidden"]:
            print(f"[CBA] residual width -> {rep['hidden']} (energy {args.cba_hidden_energy})")
        pruned_kept = list(kept); model.to(device)
    if args.prune_heads is not None:
        if arch.family != "gpt_neox":
            raise ValueError("--prune_heads currently implemented for gpt_neox only")
        from src.arch import prune_gptneox_heads
        nh = sum(prune_gptneox_heads(b, args.prune_heads, model.config) for b in kept)
        if nh and not pruned_kept:
            pruned_kept = list(kept)
        print(f"[surgery] pruned attention to {args.prune_heads} heads in {nh} kept blocks")
    if args.prune_kv_groups is not None:  # GQA head pruning (Llama-family), uniform + global config
        from src.arch import prune_gqa_heads
        grp = model.config.num_attention_heads // model.config.num_key_value_heads
        hd = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
        ng = sum(prune_gqa_heads(b, args.prune_kv_groups, model.config) for b in kept)
        model.config.num_key_value_heads = args.prune_kv_groups
        model.config.num_attention_heads = args.prune_kv_groups * grp
        model.config.head_dim = hd
        if ng and not pruned_kept:
            pruned_kept = list(kept)
        print(f"[surgery] GQA-pruned to {args.prune_kv_groups} KV groups in {ng} kept blocks")
    if replaceme_spans:                          # ReplaceMe: fit T (subset objective) + merge into down_proj
        from src.arch import replaceme_apply
        _rcal = build_calib(tok, args.calib, device, max_tokens=8192, teacher=teacher)
        for (st, ct) in replaceme_spans:
            replaceme_apply(model, teacher, arch, st, ct, _rcal, device, cosine=args.replaceme_cosine)
            print(f"[surgery] ReplaceMe: merged T into block {st-1} down_proj for span [{st},{st+ct}) (zero params)")
    if args.prune_hidden is not None and args.prune_hidden_after:   # MULTI-AXIS: narrow residual AFTER deletion+ReplaceMe merge
        from src.arch import prune_hidden_width                     # (ReplaceMe T is built+merged at full width first -> composes)
        cids = build_calib(tok, args.calib, device, teacher=teacher)
        oldH = model.config.hidden_size
        if prune_hidden_width(model, arch, args.prune_hidden, cids, device):
            model.to(device)
            print(f"[surgery] pruned residual hidden width {oldH} -> {args.prune_hidden} (after deletion+ReplaceMe)")
    if args.slice_dim is not None and args.slice_after:  # NATIVE-REPLICATION: width reshape AFTER deletion+ReplaceMe
        _do_slice()                                      # (per-layer wrapping would break the deletion gap-fill if done first)
    if args.obs_ffn is not None:                 # local-OBS FFN-width pruning (concentrate damage, reconstruct)
        from src.arch import obs_ffn_prune
        _ocal = build_calib(tok, args.calib, device, teacher=teacher)
        obs_layers = [l for l in arch.get_layers(model) if l not in repl_blocks]
        no = obs_ffn_prune(model, arch, obs_layers, _ocal, device, args.obs_ffn)
        after_obs = count_total(model)
        print(f"[surgery] OBS-FFN pruned {no} FFNs to {args.obs_ffn} width w/ local reconstruction -> "
              f"{after_obs/1e6:.1f}M ({100*(1-after_obs/before):.1f}% total)")
    if args.canon_mix > 0:                       # NET-NEW compute: canon conv mixers (no-op init, heal-grown)
        from src.arch import add_canon_mix
        add_canon_mix(model, arch, k=args.canon_mix)
    if args.layer_tie > 0:                       # LAYER-TIE depth operator (share, not delete/merge)
        from src.arch import add_layer_tie
        _tcal = build_calib(tok, args.calib, device, teacher=teacher)
        add_layer_tie(model, arch, args.layer_tie, _tcal, device); model.to(device)
        print(f"[surgery] layer-tie -> {count_total(model)/1e6:.1f}M ({100*(1-count_total(model)/before):.1f}% total)")
    if args.bus_width is not None:               # FULL-WIDTH-BUS: per-layer read/write subsets, bus keeps all H
        from src.arch import prune_hidden_bus
        _bcal = build_calib(tok, args.calib, device, teacher=teacher)
        prune_hidden_bus(model, arch, args.bus_width, _bcal, device); model.to(device)
        print(f"[surgery] bus-width -> {count_total(model)/1e6:.1f}M ({100*(1-count_total(model)/before):.1f}% total)")
    if args.layer_merge > 0:                     # LAYER-MERGE depth operator (LaCo-style fold, not delete)
        from src.arch import add_layer_merge
        _mcal = build_calib(tok, args.calib, device, teacher=teacher)
        add_layer_merge(model, arch, args.layer_merge, _mcal, device, alpha=args.merge_alpha); model.to(device)
        print(f"[surgery] layer-merge -> {count_total(model)/1e6:.1f}M ({100*(1-count_total(model)/before):.1f}% total)")
    if args.depth_capture_rank > 0:              # DEPTH capture-not-drop: ghost least-important blocks
        from src.arch import add_depth_ghosts
        _dcal = build_calib(tok, args.calib, device, teacher=teacher)
        add_depth_ghosts(model, arch, args.depth_capture_n, args.depth_capture_rank, _dcal, device); model.to(device)
        print(f"[surgery] depth-capture -> {count_total(model)/1e6:.1f}M ({100*(1-count_total(model)/before):.1f}% total)")
    if args.capture_sequential:                  # SEQUENTIAL/GPTQ capture (cross-layer-aware) - does FFN+attn together
        from src.arch import capture_sequential
        _scal = build_calib(tok, args.calib, device, teacher=teacher)
        capture_sequential(model, arch, args.hybrid_ffn_keep, args.hybrid_ffn_rank, args.hybrid_attn_keep, args.hybrid_attn_rank, _scal, device); model.to(device)
        print(f"[surgery] sequential-capture -> {count_total(model)/1e6:.1f}M ({100*(1-count_total(model)/before):.1f}% total)")
    elif args.hybrid_ffn_keep is not None:        # CAPTURE-not-DROP FFN: top-K exact + low-rank-captured remaining
        from src.arch import add_hybrid_ffn
        _hcal = build_calib(tok, args.calib, device, teacher=teacher)
        add_hybrid_ffn(model, arch, args.hybrid_ffn_keep, args.hybrid_ffn_rank, _hcal, device); model.to(device)
        print(f"[surgery] HybridFFN -> {count_total(model)/1e6:.1f}M ({100*(1-count_total(model)/before):.1f}% total)")
    if args.hybrid_attn_keep is not None and not args.capture_sequential:   # CAPTURE-not-DROP attention (skip if sequential did it)
        from src.arch import add_hybrid_attn
        _acal = build_calib(tok, args.calib, device, teacher=teacher)
        add_hybrid_attn(model, arch, args.hybrid_attn_keep, args.hybrid_attn_rank, _acal, device); model.to(device)
        print(f"[surgery] HybridAttn -> {count_total(model)/1e6:.1f}M ({100*(1-count_total(model)/before):.1f}% total)")
    if args.heads_per_group is not None:         # GQA-aware query-head pruning (Qwen/Llama): match attn to the narrowed residual
        from src.arch import obs_gqa_prune
        _gcal = build_calib(tok, args.calib, device, teacher=teacher)
        gq_layers = [l for l in arch.get_layers(model) if l not in repl_blocks]
        ng = obs_gqa_prune(model, arch, gq_layers, _gcal, device, args.heads_per_group, metric=args.head_metric)
        after_gq = count_total(model)
        print(f"[surgery] obs_gqa pruned {ng} attn ({args.head_metric}) -> {after_gq/1e6:.1f}M ({100*(1-after_gq/before):.1f}% total)")
    if args.obs_attn is not None:                # local-OBS ATTENTION-head pruning (reconstruct o_proj)
        from src.arch import obs_attn_prune
        _acal = build_calib(tok, args.calib, device, teacher=teacher)
        oa_layers = [l for l in arch.get_layers(model) if l not in repl_blocks]
        na = obs_attn_prune(model, arch, oa_layers, _acal, device, args.obs_attn)
        after_oa = count_total(model)
        print(f"[surgery] OBS-attn pruned {na} attn to {args.obs_attn} heads w/ o_proj reconstruction -> "
              f"{after_oa/1e6:.1f}M ({100*(1-after_oa/before):.1f}% total)")
    if args.realloc_adjacent is not None:        # NEAR-ZERO-HEAL width->depth: keep/realloc/drop + adjacent
        from src.arch import realloc_ffn_adjacent  # (function-preserving, Net2DeeperNet); reallocated block
        _acal = build_calib(tok, args.calib, device, teacher=teacher)  # inserted RIGHT AFTER its source layer
        cur = [l for l in arch.get_layers(model) if l not in repl_blocks]
        nA, newA, Lnew = realloc_ffn_adjacent(model, arch, cur, _acal, device, args.realloc_adjacent, args.realloc_realloc,
                                              ffn_only=args.realloc_ffn_only)
        arch.set_layers(model, torch.nn.ModuleList(Lnew)); setattr(model.config, arch.n_layer_attr, len(Lnew))
        if getattr(model.config, "layer_types", None) is not None:
            lt = list(model.config.layer_types); base = lt[0] if lt else "full_attention"
            model.config.layer_types = [(lt[k] if k < len(lt) else base) for k in range(len(Lnew))]
        reassign_layer_idx(torch.nn.ModuleList(Lnew)); repl_blocks.extend(newA)
        if args.realloc_correct and newA:        # lstsq correction (empirically 12x lower zero-heal MSE)
            from src.arch import realloc_lstsq_correct
            nc = realloc_lstsq_correct(model, arch, newA, _acal, device)
            print(f"[surgery] realloc-correct: lstsq-refit {nc} adjacent down_proj (shift correction)")
        print(f"[surgery] realloc-adjacent: keep {args.realloc_adjacent} + realloc {args.realloc_realloc} -> {nA} "
              f"adjacent blocks (now {len(Lnew)} deep) -> {count_total(model)/1e6:.1f}M "
              f"({100*(1-count_total(model)/before):.1f}% total)")
    if args.realloc_depth is not None:           # WIDTH->DEPTH reallocation: narrow FFN + transfer removed
        from src.arch import realloc_ffn_to_depth  # width into NEW depth blocks (warm-init, not dropped)
        _rcal = build_calib(tok, args.calib, device, teacher=teacher)
        cur = [l for l in arch.get_layers(model) if l not in repl_blocks]
        nnar, new_blks = realloc_ffn_to_depth(model, arch, cur, _rcal, device, args.realloc_depth, args.realloc_n_depth)
        L = list(arch.get_layers(model))
        npos = sorted({int(round((k + 1) * len(L) / (len(new_blks) + 1))) for k in range(len(new_blks))})
        for k, blk in enumerate(new_blks):
            L.insert(npos[k] + k, blk); repl_blocks.append(blk)
        nm = torch.nn.ModuleList(L); arch.set_layers(model, nm)
        setattr(model.config, arch.n_layer_attr, len(L))
        if getattr(model.config, "layer_types", None) is not None:
            lt = list(model.config.layer_types); base = lt[0] if lt else "full_attention"
            model.config.layer_types = [(lt[k] if k < len(lt) else base) for k in range(len(L))]
        reassign_layer_idx(nm)
        print(f"[surgery] realloc width->depth: narrowed {nnar} FFNs to {args.realloc_depth} + reallocated removed "
              f"width into {len(new_blks)} new depth blocks (now {len(L)} deep) -> {count_total(model)/1e6:.1f}M "
              f"({100*(1-count_total(model)/before):.1f}% total)")
    if args.svd_frac is not None:                # RANK axis: factorize kept weights via activation-aware SVD
        from src.arch import svd_compress, factorize_embeddings
        _scids = build_calib(tok, args.calib, device, max_tokens=2048, teacher=teacher)
        svd_layers = [l for l in arch.get_layers(model) if l not in repl_blocks]
        nf = svd_compress(model, arch, svd_layers, _scids, device, args.svd_frac, do_attn=args.svd_attn,
                          eora_frac=args.eora_frac)
        ne = factorize_embeddings(model, args.svd_embed, device) if args.svd_embed is not None else 0
        after_svd = count_total(model)
        print(f"[surgery] SVD-factorized {nf} matrices + {ne} embedding/head (rank_frac {args.svd_frac}) -> "
              f"{after_svd/1e6:.1f}M ({100*(1-after_svd/before):.1f}% total reduction); PRESERVES parent directions")
    if args.svd_embed is not None and args.svd_frac is None:   # STANDALONE embedding factorization (was silently
        from src.arch import factorize_embeddings              # dead without --svd_frac -> emb5_d5/d4 ran as no-ops)
        ne = factorize_embeddings(model, args.svd_embed, device)
        print(f"[surgery] factorized {ne} embedding/head (rank_frac {args.svd_embed}) -> "
              f"{count_total(model)/1e6:.1f}M ({100*(1-count_total(model)/before):.1f}% total)")
    if args.eora_rank > 0:                       # EoRA-on-top: trainable low-rank compensation paths on down_proj (eigenspace-init)
        from src.arch import add_eora
        _ecal = build_calib(tok, args.calib, device, teacher=teacher)
        add_eora(model, arch, _ecal, device, args.eora_rank); model.to(device)
    lora_params = []
    if args.lora_seam:                           # frozen-regime seam adaptation (LoRA on layers near the gap)
        from src.arch import inject_lora_seam
        lora_params, n_seam = inject_lora_seam(model, arch, repl_blocks, r=args.lora_rank, span=args.lora_span)
        model.to(device)
        print(f"[surgery] LoRA-seam: wrapped {n_seam} kept blocks near gaps (rank {args.lora_rank}, "
              f"+{sum(p.numel() for p in lora_params)/1e6:.2f}M trainable)")
    if args.attn_res:                            # gated attention-residual retrofit (init 0 -> identity)
        from src.arch import attn_res_retrofit
        _arp, _wrapped = attn_res_retrofit(arch, model, mdtype, device)
        # the spliced/pruned blocks are now wrapped -> map names through for healing-set membership
        repl_blocks = [w for w in _wrapped if w.layer in repl_blocks]
        pruned_kept = [w for w in _wrapped if w.layer in pruned_kept] if pruned_kept else []
        print(f"[surgery] attn-res retrofit: wrapped {len(_wrapped)} layers (+{sum(p.numel() for p in _arp)/1e6:.2f}M gated params, init 0)")
    model.to(device)
    after = count_total(model)
    print(f"[surgery] {arch.family}: removed {count} layers -> {len(repl_blocks)} spliced block(s); "
          f"anchors(comp->teacher) {spliced_anchors}; params {before/1e6:.1f}M -> {after/1e6:.1f}M "
          f"({100*(1-after/before):.1f}% reduction)")
    # aspect-ratio diagnostic: hidden/layers vs the literature deep-narrow optimum for small models
    _nl = len(arch.get_layers(model)); _h = model.config.hidden_size
    print(f"[shape] depth={_nl} hidden={_h} aspect_ratio={_h/_nl:.0f} (small-model literature optimum ~55-85; "
          f"higher = too shallow-wide for the param budget)")
    # cross-width / cross-depth feature distillation: when the student's hidden dim or depth differs
    # from the teacher (add_depth, prune_hidden), a 1:1 same-dim match is invalid. Build a LEARNABLE
    # projection (student_dim -> teacher_dim, TinyBERT-style) + RELATIVE-DEPTH pairs, biased to MIDDLE
    # layers (CKA shows parent/small reps align in the middle, diverge late). Replaces the naive disable.
    feat_proj = None; cross_feat = False; rel_pairs = []
    s_hidden = model.config.hidden_size
    t_hidden = teacher.config.hidden_size
    if args.rel_distill:                                  # middle relative-depth pairs for relation transfer
        _ns = len(arch.get_layers(model)); _nt = teacher.config.num_hidden_layers
        rel_pairs = [(i, max(1, round(i / _ns * _nt))) for i in range(2, int(_ns * 0.85) + 1)]
    if args.mid_feat and args.feat_distill and s_hidden == t_hidden and args.add_depth == 0:
        # CKA-informed (same-dim): feature-distill the ALIGNED MIDDLE layers, NOT the divergent final
        # hidden. Logit-KD still carries the output behavior. Skip last 15% (divergent) + the spliced gap.
        ns = len(arch.get_layers(model)); nt = teacher.config.num_hidden_layers
        feat_pairs = [(i, max(1, round(i / ns * nt))) for i in range(2, int(ns * 0.85) + 1)]
        cross_feat = True; spliced_anchors = []
        print(f"[feat-distill] MID-only: {len(feat_pairs)} aligned middle layers (no final-hidden match)")
    if args.add_depth > 0 or s_hidden != t_hidden:
        args.deep_distill = False; args.warm_distill_steps = 0; spliced_anchors = []
        if args.feat_distill:
            cross_feat = True
            import torch.nn as _nn
            ns = len(arch.get_layers(model)); nt = teacher.config.num_hidden_layers
            # MIDDLE-biased relative-depth pairs (skip the divergent last 15%): student i -> teacher round(i/ns*nt)
            feat_pairs = [(i, max(1, round(i / ns * nt))) for i in range(1, int(ns * 0.85) + 1)]
            feat_proj = _nn.Linear(s_hidden, t_hidden, bias=False).to(device, mdtype) if s_hidden != t_hidden else None
            print(f"[feat-distill] cross-dim: {len(feat_pairs)} middle relative-depth pairs"
                  + (f" + learned {s_hidden}->{t_hidden} projection" if feat_proj is not None else ""))
    _ghost_gi = None; _ghost_proj = None; _ghost_hook_data = {}
    if args.ghost_distill and getattr(model.config, "_ghost_idx", None):
        _ghost_gi = model.config._ghost_idx                 # DEPTH ghost-distill via HOOKS (hidden_states recorder skips custom layers)
        if s_hidden != t_hidden:
            import torch.nn as _nn
            _ghost_proj = _nn.Linear(s_hidden, t_hidden, bias=False).to(device, mdtype)
        print(f"[ghost-distill] hook-based: {len(_ghost_gi)} ghosts -> teacher block outputs (idx {_ghost_gi})"
              + (" +proj" if _ghost_proj is not None else ""))
    elif args.drop_distill:                                  # FFN/attn CAPTURE: hidden-distill captured middle layers (real DecoderLayers -> recorded)
        _ns = len(arch.get_layers(model)); _nt = teacher.config.num_hidden_layers
        feat_pairs = [(i, max(1, round(i / _ns * _nt))) for i in range(2, int(_ns * 0.85) + 1)]
        args.feat_distill = True
        if s_hidden != t_hidden:
            cross_feat = True
            if feat_proj is None:
                import torch.nn as _nn
                feat_proj = _nn.Linear(s_hidden, t_hidden, bias=False).to(device, mdtype)
        else:
            args.deep_distill = True; cross_feat = False
        print(f"[drop-distill] {len(feat_pairs)} captured middle layers -> teacher" + (" (+proj)" if cross_feat else ""))
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(); model.config.use_cache = False

    # --- data: WikiText-2 test for ppl; healing corpus = WikiText-2 or streamed C4 ---
    test_txt = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    test_ids = tok(test_txt, return_tensors="pt").input_ids
    if args.heal_data == "teacher":
        # Self-Data Distillation (2410.09982): heal on text the TEACHER generates from real C4 seed prefixes,
        # so supervision sits exactly on the teacher's distribution. CACHED to disk (keyed by model+budget+
        # seqlen) and BATCH-generated — generate once (~minutes batched), reuse across every experiment.
        import os
        os.makedirs("data_cache", exist_ok=True)
        slug = args.model.replace("/", "_")
        cache = f"data_cache/sdd_{slug}_{args.heal_tokens}_{args.seqlen}.pt"
        if os.path.exists(cache):
            train_ids = torch.load(cache)
            print(f"[data] loaded cached self-distill tokens {tuple(train_ids.shape)} from {cache}")
        else:
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            seeds = []                                            # fixed 16-token prefixes -> clean batching
            for ex in ds:
                ids = tok(ex["text"], return_tensors="pt").input_ids[0]
                if ids.size(0) >= 16:
                    seeds.append(ids[:16])
                if len(seeds) >= args.sdd_seeds:
                    break
            teacher.config.use_cache = True
            gen_chunks, ntok, BS = [], 0, 32                      # BATCHED generation (~10-30x faster)
            for b in range(0, len(seeds), BS):
                batch = torch.stack(seeds[b:b + BS]).to(device)  # (BS, 16) all same length
                with torch.no_grad():
                    out = teacher.generate(batch, max_new_tokens=args.seqlen, do_sample=True,
                                           temperature=1.0, top_p=0.95, pad_token_id=tok.eos_token_id or 0)
                gen_chunks.append(out.cpu()); ntok += out.numel()
                if ntok >= args.heal_tokens:
                    break
            teacher.config.use_cache = False
            train_ids = torch.cat([g.reshape(1, -1) for g in gen_chunks], dim=1)
            torch.save(train_ids, cache)
            print(f"[data] generated + cached self-distill {tuple(train_ids.shape)} ({ntok} tok) -> {cache}")
    elif args.heal_data == "c4":
        # Stream C4 until ~heal_tokens tokens — a real healing corpus for billion-param models
        # (WikiText-2's 2.4M tokens are far too few to recover a multi-billion-param LM).
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        buf, ntok = [], 0
        if args.heal_hard > 0:                       # HARD-EXAMPLE mining: keep top-F chunks by TEACHER loss
            cand, scores, CH = [], [], 512
            pend = torch.zeros(1, 0, dtype=torch.long)
            with torch.no_grad():
                for ex in ds:
                    ids = tok(ex["text"], return_tensors="pt").input_ids
                    pend = torch.cat([pend, ids], 1)
                    while pend.size(1) >= CH:
                        ch = pend[:, :CH]; pend = pend[:, CH:]
                        out = teacher(ch.to(device), labels=ch.to(device))
                        cand.append(ch); scores.append(out.loss.item())
                    if len(cand) * CH >= args.heal_tokens / max(args.heal_hard, 1e-6):
                        break
            k = max(1, int(args.heal_tokens // CH))
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            buf = [cand[i] for i in sorted(top)]
            import statistics as _st
            print(f"[data] HARD-mined {len(buf)} / {len(cand)} chunks (teacher-loss median kept "
                  f"{_st.median([scores[i] for i in top]):.2f} vs all {_st.median(scores):.2f})")
        else:
            for ex in ds:
                ids = tok(ex["text"], return_tensors="pt").input_ids
                buf.append(ids); ntok += ids.size(1)
                if ntok >= args.heal_tokens:
                    break
        train_ids = torch.cat(buf, dim=1)
    elif args.heal_data == "mathmix":
        # CAPABILITY-TARGETED heal: 50% C4 web + 50% OpenWebMath, same total budget. The audit showed math
        # retention ~1% (gsm8k 61.7 -> 0.76) under pure-C4 healing - the heal data carried no math signal.
        ds_c4 = iter(load_dataset("allenai/c4", "en", split="train", streaming=True))
        ds_m = iter(load_dataset("open-web-math/open-web-math", split="train", streaming=True))
        buf, ntok, flip = [], 0, 0
        while ntok < args.heal_tokens:
            src = ds_m if flip % 2 == 0 else ds_c4
            try:
                ex = next(src)
            except StopIteration:
                break
            ids = tok(ex["text"], return_tensors="pt").input_ids
            buf.append(ids); ntok += ids.size(1); flip += 1
        train_ids = torch.cat(buf, dim=1)
        print(f"[data] MATHMIX heal: 50/50 OpenWebMath+C4, {ntok/1e6:.1f}M tokens")
    elif args.heal_data == "instruct":
        # INSTRUCTION heal (SlimOrca): denser reasoning/instruction signal per token than web C4. Tests whether the
        # +9pt closed-form-calib data effect transfers to the GRADIENT heal, at the SAME cheap token budget.
        ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
        buf, ntok = [], 0
        for ex in ds:
            convo = ex.get("conversations") or []
            txt = "\n".join(c.get("value", "") for c in convo)
            if not txt.strip():
                continue
            ids = tok(txt, return_tensors="pt").input_ids
            buf.append(ids); ntok += ids.size(1)
            if ntok >= args.heal_tokens:
                break
        train_ids = torch.cat(buf, dim=1)
    else:
        train_txt = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
        train_ids = tok(train_txt, return_tensors="pt").input_ids
    print(f"[data] heal={args.heal_data} train tokens {train_ids.size(1)}, test tokens {test_ids.size(1)}")

    teacher_ppl = perplexity(teacher, test_ids, device)
    pre_ppl = perplexity(model, test_ids, device)
    print(f"[eval] teacher ppl {teacher_ppl:.2f} | spliced pre-heal ppl {pre_ppl:.2f}")

    if args.offload_parent and big_teacher is not None:
        # parent's training jobs (warm-start, ppl baseline) are done; free it for a big (70B-4bit) teacher
        teacher = teacher.to("cpu"); torch.cuda.empty_cache()
        args.feat_distill = False; args.deep_distill = False; args.attn_distill = False
        print("[offload] parent teacher -> CPU; healing = CE + big-teacher logit-KD only")

    seqlen = args.seqlen
    train_start = time.perf_counter()
    # batched sequence layout: X[i], Y[i] are aligned (input, next-token) of length seqlen
    nseq = (train_ids.size(1) - 1) // seqlen
    X = train_ids[0, :nseq * seqlen].reshape(nseq, seqlen)
    Y = train_ids[0, 1:nseq * seqlen + 1].reshape(nseq, seqlen)

    # auto batch size from model size + free GPU memory (H100 fits many more than 1)
    if args.batch_size > 0:
        bs = args.batch_size
    elif device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free_gb = props.total_memory / 1e9 - torch.cuda.memory_allocated(device) / 1e9 - 6.0
        # hidden-state distillation keeps all layers' activations -> heavier per-seq
        factor = 0.65 if (args.feat_distill or args.warm_distill_steps > 0) else 0.45
        per_seq = max(0.25, (before / 1e9) * (seqlen / 512) * factor)
        bs = int(max(1, min(16, free_gb / per_seq)))
    else:
        bs = 1
    bs = min(bs, nseq)
    print(f"[batch] sequences/step = {bs} (seqlen {seqlen}); {nseq} sequences in corpus")

    # nanochat-style auto-derived LR: scale the base LR for the COMPRESSED model's geometry instead of
    # a one-size-fits-all value. (1) width: lr ∝ √(ref_hidden / compressed_hidden) — a narrowed model
    # wants a HIGHER lr (AdamW/Muon width rule). (2) batch: lr ∝ √(bs / ref_bs) (AdamW batch rule).
    if args.auto_lr:
        ref_hidden, ref_bs = 2048.0, 4.0
        lr_mult = (ref_hidden / max(1, model.config.hidden_size)) ** 0.5 * (bs / ref_bs) ** 0.5
        old_lr = args.lr; args.lr = args.lr * lr_mult
        print(f"[auto-lr] base {old_lr:.1e} x {lr_mult:.2f} (width {model.config.hidden_size}, bs {bs}) -> {args.lr:.1e}")

    def get_batch(step):
        rows = [(step * bs + j) % nseq for j in range(bs)]
        return X[rows].to(device), Y[rows].to(device)

    def make_opt(params):
        """8-bit Adam if available (memory), else AdamW."""
        params = [p for p in params if p.requires_grad]
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(params, lr=args.lr, weight_decay=0.01)
        except Exception:
            return torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    nsteps = args.heal_steps  # with batching + wraparound we can exceed one pass

    import contextlib
    if args.teacher_correct > 0 and teacher is not None:
        # TEACHER CORRECTION (Minitron-in-practice): briefly fine-tune the PARENT on the heal corpus
        # before distilling, so its distribution matches our heal data (fixes the shift when the
        # original pretraining data is unavailable). Nearly free, improves the KD target.
        teacher.train()
        if hasattr(teacher, "gradient_checkpointing_enable"):
            teacher.gradient_checkpointing_enable()       # memory: checkpoint teacher activations
        for p in teacher.parameters():
            p.requires_grad = True
        try:                                              # 8-bit Adam halves optimizer-state memory
            import bitsandbytes as bnb
            topt = bnb.optim.Adam8bit(teacher.parameters(), lr=args.teacher_correct_lr, weight_decay=0.0)
        except Exception:
            topt = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_correct_lr, weight_decay=0.0)
        for i in range(args.teacher_correct):
            inp, lab = get_batch(i)
            with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext():
                tl = teacher(inp, use_cache=False).logits
                tcl = F.cross_entropy(tl.reshape(-1, tl.size(-1)).float(), lab.reshape(-1))
            topt.zero_grad(); tcl.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0); topt.step()
            if i % 100 == 0:
                print(f"  [teacher-correct] step {i} ce {tcl.item():.3f}")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if hasattr(teacher, "gradient_checkpointing_disable"):
            teacher.gradient_checkpointing_disable()      # restore for frozen KD use
        topt.zero_grad(set_to_none=True); del topt        # free optimizer states (was leaking -> OOM)
        teacher.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        print(f"[teacher-correct] fine-tuned parent {args.teacher_correct} steps on heal corpus")

    def warm_distill():
        """Stage-1 (two-stage recipe): feature-distill the spliced block to reproduce the
        teacher's span-exit hidden state, as a good init BEFORE healing. Uses the FULL model
        forward (so rotary/masks are correct across archs) but trains ONLY the spliced block."""
        if args.warm_distill_steps <= 0 or not repl_blocks:
            return
        saved = [(p, p.requires_grad) for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad = False
        repl_params = [p for b in repl_blocks for p in b.parameters()]
        for p in repl_params:
            p.requires_grad = True
        if lora_params:         # frozen + LoRA-on-seam: train the seam adapters too
            for p in lora_params:
                p.requires_grad = True
            repl_params = repl_params + lora_params
        if args.lillama:        # Lillama recipe: also unfreeze RMSNorm (near-free, high-leverage)
            for nm, m in model.named_modules():
                if nm.endswith("norm") and hasattr(m, "weight") and m.weight is not None:
                    m.weight.requires_grad = True; repl_params.append(m.weight)
        o = make_opt(repl_params); model.train()
        best, bad, ema = float("inf"), 0, None
        for i in range(args.warm_distill_steps):
            inp, _ = get_batch(i)
            mh = model(inp, use_cache=False, output_hidden_states=True).hidden_states
            with torch.no_grad():
                th = teacher(inp, use_cache=False, output_hidden_states=True).hidden_states
            # anchor each spliced block's exit (mh[c]) to its teacher hidden th[t]
            if args.lillama:    # local L1 + log-sigmoid-cosine (2412.16719) instead of MSE
                loss = sum(_lillama(mh[c], th[t]) for c, t in spliced_anchors) / len(spliced_anchors)
            else:
                loss = sum(_feat(mh[c], th[t]) + F.mse_loss(mh[c].float(), th[t].float())
                           for c, t in spliced_anchors) / len(spliced_anchors)
            o.zero_grad(); loss.backward(); o.step()
            l = loss.item()
            ema = l if ema is None else 0.95 * ema + 0.05 * l
            if i % 100 == 0:
                print(f"  [warm-distill] step {i} feat {l:.4f} (ema {ema:.4f})")
            if args.warm_plateau:                 # stop when the condenser CONVERGES (plateau), not fixed steps
                if ema < best - args.warm_plateau_eps:
                    best, bad = ema, 0
                else:
                    bad += 1
                if i > 50 and bad >= args.warm_plateau_patience:
                    print(f"  [warm-distill] PLATEAU at step {i}: feat ema {ema:.4f} (best {best:.4f})")
                    break
        print(f"  [warm-distill] final feat {ema:.4f} after {i+1} steps (converged={args.warm_plateau and bad>=args.warm_plateau_patience})")
        for p, rg in saved:
            p.requires_grad = rg

    def make_sched(opt):
        """Linear warmup (first 5%) then cosine decay to 0 — stabilizes healing."""
        warm = max(10, nsteps // 20)
        def lr_lambda(step):
            if step < warm:
                return step / warm
            prog = (step - warm) / max(1, nsteps - warm)
            return 0.5 * (1 + math.cos(math.pi * prog))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if args.partial_heal:
        # Heal spliced block + pruned kept blocks + final-norm/head; freeze embeddings
        # and unpruned blocks. The memory-efficient regime that recovers quality at scale.
        for p in model.parameters():
            p.requires_grad = False
        heal_mods = list(repl_blocks) + pruned_kept + list(arch.trainable_extra(model))
        for mod in heal_mods:
            for p in mod.parameters():
                p.requires_grad = True
        ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[partial-heal] training {len(heal_mods)} modules, {ntr/1e6:.1f}M params "
              f"({100*ntr/after:.1f}% of model); embeddings + unpruned blocks frozen")
        warm_distill()
        opt = make_opt(model.parameters()); sched = make_sched(opt)
        model.train()
        for i in range(nsteps):
            inp, lab = get_batch(i)
            logits = model(inp, use_cache=False).logits
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), lab.reshape(-1))
            with torch.no_grad():
                t_logits = teacher(inp, use_cache=False).logits
            kd = _kd(logits, t_logits)
            loss = 0.5 * ce + 0.5 * kd
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step()
            if i % 100 == 0:
                print(f"  [partial-heal] step {i} loss {loss.item():.3f} (ce {ce.item():.3f} kd {kd.item():.3f})")
    elif args.distill_only:
        # Memory-trivial scaling path: train ONLY the spliced block to match the
        # teacher's span-exit hidden state (frozen backbone). h_in = teacher hs[start]
        # is exactly what the spliced model feeds the module (front blocks identical).
        for p in model.parameters():
            p.requires_grad = False
        tparams = list(repl.parameters())
        for p in repl.parameters():
            p.requires_grad = True
        if args.lillama:        # Lillama recipe: also unfreeze RMSNorm (near-free, high-leverage)
            for nm, m in model.named_modules():
                if nm.endswith("norm") and hasattr(m, "weight") and m.weight is not None:
                    m.weight.requires_grad = True; tparams.append(m.weight)
        opt = torch.optim.AdamW(tparams, lr=args.lr, weight_decay=0.01)
        repl.train()
        for i in range(nsteps):
            inp, _ = get_batch(i)
            with torch.no_grad():
                hs = teacher(inp, output_hidden_states=True, use_cache=False).hidden_states
            h_in, h_out = hs[start], hs[teacher_index]
            pred = repl(h_in, use_cache=False)[0]
            loss = _lillama(pred, h_out) if args.lillama else _feat(pred, h_out) + F.mse_loss(pred.float(), h_out.float())
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 200 == 0:
                print(f"  [distill{'/lillama' if args.lillama else ''}] step {i} loss {loss.item():.4f}")
    else:  # full-heal (default): train the whole compressed student end-to-end
        if heal_teacher is not None:                  # STAGED: distill toward the ORIGINAL full model (logit-KD, width-independent),
            assert args.ghost_distill or args.drop_distill or not (args.feat_distill or args.deep_distill or args.attn_distill), \
                "heal_teacher is logit-KD only; incompatible with feat/hidden/attn distill (width mismatch)"
            # --ghost_distill exception: ghosted student has ALIGNED layers w/ the original teacher (feat_proj handles any width gap)
            teacher = heal_teacher                    # not the loaded phase-1 (degraded) model -> each stage chases the original
            print(f"[staged] heal distills toward heal_teacher (hidden={teacher.config.hidden_size}) not the phase-1 model")
        import contextlib, itertools
        warm_distill()
        proj_params = list(feat_proj.parameters()) if feat_proj is not None else []
        if _ghost_gi is not None:                     # register ghost-distill hooks: student ghost out + FINALIZED teacher block out
            _sl = arch.get_layers(model); _tl = arch.get_layers(teacher)
            def _mk_gh(gi, side):
                def _h(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    _ghost_hook_data[(side, gi)] = o.detach() if side == "t" else o
                return _h
            for gi in _ghost_gi:
                _sl[gi].register_forward_hook(_mk_gh(gi, "s")); _tl[gi].register_forward_hook(_mk_gh(gi, "t"))
            if _ghost_proj is not None: proj_params += list(_ghost_proj.parameters())
        if args.eora_only:
            # FROZEN-backbone + EoRA: freeze everything, train ONLY the EoRA error-compensation paths
            # (+ final norm). Tests whether dedicated low-rank repair cheaply fixes distributed (factorized)
            # damage where full-heal couldn't. Cheap-at-scale: backbone never backprops.
            from src.arch import LowRankLinear
            for p in model.parameters():
                p.requires_grad = False
            ntr = 0
            for m in model.modules():
                if isinstance(m, LowRankLinear) and getattr(m, "eora", False):
                    for p in itertools.chain(m.down_e.parameters(), m.up_e.parameters()):
                        p.requires_grad = True; ntr += p.numel()
            for nm, m in model.named_modules():
                if nm.endswith("norm") and hasattr(m, "weight") and m.weight is not None:
                    m.weight.requires_grad = True; ntr += m.weight.numel()
            print(f"[eora-only] frozen backbone; training {ntr/1e6:.2f}M EoRA+norm params only")
        # differential LR: spliced blocks at full lr, kept (pretrained) layers at lr*kept_lr_mult
        if args.kept_lr_mult != 1.0:
            repl_ids = {id(p) for b in repl_blocks for p in b.parameters()}
            g_new = [p for p in model.parameters() if p.requires_grad and id(p) in repl_ids]
            g_kept = [p for p in model.parameters() if p.requires_grad and id(p) not in repl_ids]
            groups = [{"params": g_new, "lr": args.lr},
                      {"params": g_kept, "lr": args.lr * args.kept_lr_mult}]
            try:
                import bitsandbytes as bnb
                opt = bnb.optim.Adam8bit(groups, lr=args.lr, weight_decay=0.01)
            except Exception:
                opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
            print(f"[diff-lr] spliced lr={args.lr}, kept lr={args.lr*args.kept_lr_mult}")
        else:
            opt = make_opt(itertools.chain(model.parameters(), proj_params))
        sched = make_sched(opt)
        use_scaler = (dt == torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        model.train()
        for i in range(nsteps):
            inp, lab = get_batch(i)
            on_policy = False
            if args.gkd and torch.rand(1).item() < args.gkd_lambda:   # ON-POLICY: student generates its OWN sequence (teacher scores it)
                on_policy = True
                _eos = tok.eos_token_id if tok.eos_token_id is not None else 0
                _pad = tok.pad_token_id if tok.pad_token_id is not None else _eos
                model.eval()
                with torch.no_grad():
                    gen = model.generate(inp[:, :args.gkd_prefix], max_new_tokens=inp.size(1) - args.gkd_prefix,
                                         do_sample=True, temperature=1.0, top_p=0.95, pad_token_id=_pad)
                model.train()
                if gen.size(1) < inp.size(1):                          # pad to the corpus seqlen
                    pad = torch.full((gen.size(0), inp.size(1) - gen.size(1)), _eos, device=gen.device, dtype=gen.dtype)
                    gen = torch.cat([gen, pad], 1)
                inp = gen[:, :inp.size(1)].contiguous(); lab = inp.clone()   # lab is a placeholder (CE is zeroed on-policy)
            # bf16/fp16-loaded model already runs low-precision -> nullcontext; fp32 -> autocast bf16
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dt is None and device.type == "cuda" else contextlib.nullcontext()
            want_hs = args.feat_distill or args.deep_distill or args.rel_distill
            with ctx:
                out = model(inp, use_cache=False, output_hidden_states=want_hs,
                            output_attentions=args.attn_distill)
                logits = out.logits
                tk = args.topk_kd                              # top-K KD (0 = full distribution)
                if args.offload_parent and big_teacher is not None:
                    with torch.no_grad():                       # parent on CPU: KD from big teacher only
                        b_logits = big_teacher(inp, use_cache=False).logits[..., :model.config.vocab_size]
                    t_out = out; t_logits = b_logits
                else:
                    with torch.no_grad():
                        t_out = teacher(inp, use_cache=False, output_hidden_states=want_hs,
                                        output_attentions=args.attn_distill)
                        t_logits = t_out.logits
                # hard-example weights: focus gradient on tokens where pruning BROKE the function
                w = _disagreement(logits, t_logits) if args.disagree_weight else None
                ce_t = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       lab.reshape(-1), reduction="none")
                ce = (ce_t * w).sum() / (w.sum() + 1e-8) if w is not None else ce_t.mean()
                kd = (_jsd(logits, t_logits, beta=args.jsd_beta, weight=w) if (args.jsd or on_policy)
                      else _dkd(logits, t_logits, lab, beta=args.dkd_beta) if args.dkd
                      else _kd(logits, t_logits, topk=tk, weight=w, reverse=args.reverse_kl,
                               standardize=args.logit_std, focus=args.kd_focus))
                if not (args.offload_parent and big_teacher is not None) and big_teacher is not None:
                    with torch.no_grad():
                        b_logits = big_teacher(inp, use_cache=False).logits[..., :model.config.vocab_size]
                    kd = (1.0 - args.big_kd_weight) * kd + args.big_kd_weight * _kd(logits, b_logits, topk=tk, weight=w)
                ad = 0.0
                if args.attn_distill and spliced_anchors:
                    # match each spliced block's attention map to its teacher anchor-layer attention (KL)
                    sa, ta = out.attentions, t_out.attentions
                    ad = sum(_attn_kl(sa[c - 1], ta[t - 1]) for c, t in spliced_anchors) / len(spliced_anchors)
                cw = 0.0 if on_policy else args.ce_weight  # on-policy: no CE (student-generated tokens have no ground truth)
                if want_hs:
                    mh, th = out.hidden_states, t_out.hidden_states
                    _pf = (lambda x: feat_proj(x)) if feat_proj is not None else (lambda x: x)
                    fd = 0.0
                    if args.feat_distill or args.deep_distill:   # hidden-MSE only when explicitly requested
                        _vp = [(c, t) for c, t in feat_pairs if c < len(mh) and t < len(th)]   # guard: skip out-of-range
                        if len(_vp) < len(feat_pairs) and i == 0:
                            print(f"  [drop-distill] len(mh)={len(mh)} len(th)={len(th)} -> {len(_vp)}/{len(feat_pairs)} valid pairs")
                        if cross_feat:         # cross-dim/depth: projected middle relative-depth pairs
                            fd = sum(_feat(_pf(mh[c]), th[t]) for c, t in _vp) / max(1, len(_vp))
                        elif args.deep_distill:  # match ALL aligned hidden states (rich per-token signal)
                            fd = sum(_feat(mh[c], th[t]) for c, t in _vp) / max(1, len(_vp))
                        else:                  # spliced-block exits + final (same-dim only)
                            fd = _feat(mh[-1], th[-1])
                            if spliced_anchors:
                                fd = fd + sum(_feat(mh[c], th[t]) for c, t in spliced_anchors) / len(spliced_anchors)
                    rd = 0.0
                    if args.rel_distill and rel_pairs:    # dimension-agnostic token-token relation transfer
                        rd = sum(_rel(mh[c], th[t]) for c, t in rel_pairs) / len(rel_pairs)
                    loss = cw * ce + (1.0 - cw) * kd + args.feat_weight * fd + args.attn_weight * ad \
                        + args.rel_weight * rd
                else:
                    loss = cw * ce + (1.0 - cw) * kd + args.attn_weight * ad
                if args.mtp_weight > 0:                # MTP aux: same head predicts t+2..t+H (zero eval params)
                    for _k in range(2, max(2, args.mtp_horizon) + 1):
                        _w = args.mtp_weight * (0.5 ** (_k - 2))
                        if args.mtp_kd:                # soft teacher targets: teacher's dist for token i+k lives at pos i+k-1
                            loss = loss + _w * _kd(logits[:, :logits.size(1) - _k], t_logits[:, _k - 1:t_logits.size(1) - 1])
                        else:
                            _lk = logits[:, :-_k].reshape(-1, logits.size(-1)).float()
                            loss = loss + _w * F.cross_entropy(_lk, inp[:, _k:].reshape(-1))
                if _ghost_gi is not None:              # DROP-SPECIFIC: each ghost output -> teacher's ORIGINAL block output (hooks)
                    _gp = [(_ghost_hook_data.get(("s", gi)), _ghost_hook_data.get(("t", gi))) for gi in _ghost_gi]
                    _gp = [(so, to) for so, to in _gp if so is not None and to is not None]
                    if _gp:
                        gfd = sum(F.mse_loss((_ghost_proj(so) if _ghost_proj is not None else so).float(), to.float())
                                  for so, to in _gp) / len(_gp)
                        loss = loss + args.feat_weight * gfd
            opt.zero_grad()
            if use_scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
            sched.step()
            if i % 100 == 0:
                print(f"  [full-heal] step {i} loss {loss.item():.3f} (ce {ce.item():.3f})")
    train_seconds = time.perf_counter() - train_start
    heal_mode = ("partial" if args.partial_heal else "distill" if args.distill_only
                 else "full")

    post_ppl = perplexity(model, test_ids, device)
    print(f"[eval] compressed post-heal ppl {post_ppl:.2f}")

    if args.save_model_path:                     # STAGED phase-1: persist healed (uniformly-narrowed) model for phase-2 resume
        model.save_pretrained(args.save_model_path)            # save_pretrained round-trips (uniform hidden config + explicit head_dim)
        tok.save_pretrained(args.save_model_path)
        print(f"[staged] saved phase-1 model -> {args.save_model_path} (hidden={model.config.hidden_size})")

    # --- LM-eval-harness (downstream usability) — run IN-PROCESS on the healed model
    # object (width-pruned per-layer FFN can't round-trip through a standard config). ---
    lmeval = {}
    if args.lm_eval_tasks:
        tasks = args.lm_eval_tasks.split(",")
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        model.config.use_cache = True
        def _pick(res):  # robust across task metric names (acc_norm/acc/exact_match/mc1/f1...)
            for k in ("acc_norm,none", "acc,none", "exact_match,strict-match",
                      "exact_match,flexible-extract", "exact_match,none", "mc1,none",
                      "f1,none", "pass@1,none"):
                if res.get(k) is not None:
                    return res[k]
            for k, v in res.items():           # fallback: first numeric metric (ignore *_stderr)
                if isinstance(v, (int, float)) and not k.endswith("stderr") and "alias" not in k:
                    return v
            return None
        import gc, json, os
        gc.collect(); torch.cuda.empty_cache()                 # free optimizer/grad memory before eval
        os.makedirs("data_cache", exist_ok=True)               # TEACHER-BASKET CACHE: the teacher never changes across
        lim = getattr(args, "eval_limit", None)                # experiments -> cache its per-task scores once, reuse forever.
        tcache = f"data_cache/teacher_eval_{args.model.replace('/', '_')}.json"   # keyed by base model; spans all tasks/metrics
        if args.load_model:                                    # STAGED: the in-run "teacher" is the LOADED ckpt, not the base
            tcache = f"data_cache/teacher_eval_LOADED_{args.load_model.replace('/', '_')}.json"   # never pollute the base cache
        tscores = {}
        if os.path.exists(tcache):
            try: tscores = json.load(open(tcache))
            except Exception: tscores = {}
        _ck = lambda t: f"{t}@{lim}"                            # cache key per task (limit-aware: full vs --eval_limit differ)
        for tag_name, m in (("teacher", teacher), ("compressed", model)):
            if m is None:
                continue
            run_tasks = tasks if tag_name == "compressed" else [t for t in tasks if _ck(t) not in tscores]
            r = {}
            if run_tasks:                                       # only spin up the GPU eval for tasks not already cached
                other = model if m is teacher else teacher      # keep only ONE model on GPU during its eval
                if other is not None:
                    try: other.to("cpu")
                    except Exception: pass
                m.to(device); m.eval(); gc.collect(); torch.cuda.empty_cache()
                lm = HFLM(pretrained=m, tokenizer=tok, batch_size=args.eval_bs)
                r = simple_evaluate(model=lm, tasks=run_tasks, **({"limit": lim} if lim else {}))["results"]
            elif tag_name == "teacher":
                print(f"[lm-eval] teacher: all {len(tasks)} task(s) from cache -> skip eval ({tcache})")
            valid = []
            for t in tasks:
                if tag_name == "teacher" and t not in run_tasks:   # pull from cache
                    lmeval[f"teacher/{t}"] = tscores[_ck(t)]; valid.append(t); continue
                acc = _pick(r.get(t, {}))
                if acc is None:
                    print(f"[lm-eval] WARN no metric for {t}; skipping"); continue
                val = round(100 * acc, 2)
                lmeval[f"{tag_name}/{t}"] = val; valid.append(t)
                if tag_name == "teacher": tscores[_ck(t)] = val   # populate cache with freshly-evaluated teacher scores
            if tag_name == "teacher":
                try: json.dump(tscores, open(tcache, "w"))
                except Exception: pass
            avg = sum(lmeval[f"{tag_name}/{t}"] for t in valid) / max(1, len(valid))
            lmeval[f"{tag_name}/avg"] = round(avg, 2)
            print(f"[lm-eval] {tag_name}: avg {avg:.2f} | " +
                  " ".join(f"{t}={lmeval[f'{tag_name}/{t}']}" for t in valid))
        if lmeval.get("teacher/avg"):
            keep = lmeval["compressed/avg"] / lmeval["teacher/avg"] * 100
            print(f"[lm-eval] DOWNSTREAM RETENTION: {keep:.1f}% "
                  f"(compressed {lmeval['compressed/avg']} vs teacher {lmeval['teacher/avg']})")

    metrics = {
        "task": "wikitext2", "model": args.model, "replacement": f"{arch.family}block",
        "selection": f"remove{args.remove}_repl{K}_ffn{args.prune_ffn}", "metric_name": "ppl",
        "heal": heal_mode, "n_replace": K, "net_layers_removed": count - K,
        "method": ("AMBR" if K > 1 else "single") + ("+width" if args.prune_ffn else ""),
        "teacher_metric": float(teacher_ppl), "spliced_pre": float(pre_ppl),
        "spliced_post": float(post_ppl),
        "retained_pct": 100.0 * teacher_ppl / post_ppl,  # ppl: lower better -> ratio teacher/student
        "total_params_M": after / 1e6, "teacher_params_M": before / 1e6,
        "param_reduction_pct": 100.0 * (1 - after / before),
        "removed_blocks": count, "peak_vram_mb": report.peak_vram_mb(device),
        "train_seconds": train_seconds, **lmeval,
    }
    report.emit(metrics, output_dir="experiments", run_tag=args.tag)


if __name__ == "__main__":
    main()
