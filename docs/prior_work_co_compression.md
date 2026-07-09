# Prior Work — Measurement-Driven Co-Compression (foundational study)

> **Read this for the method lineage; read [`capability_stratified_compression.md`](capability_stratified_compression.md) for what its retention numbers do and don't capture.**
>
> This is the earlier, foundational study (encoder/GLUE + GPT-2 generation, RTX 3070) that established the compression recipe — **co-compress depth and width**, allocate the budget from cheap redundancy measurements (the CBA) — and a **small-scale scaling law**. It was later scaled to Qwen2.5-1.5B, where the capability-stratified study (the main write-up) showed that its style of aggregate retention number is a **surface-task metric that hides reasoning collapse**. Two honest caveats when reading below: (1) all retention figures here are GLUE / perplexity / surface metrics — none test multi-step reasoning; (2) the scaling law here ("compressibility rises with size", GPT-2 124M→774M) **saturates** at LLM scale — pythia 1.4B→6.9B is flat at ~86% surface retention — so scale buys *surface* headroom with diminishing returns, and whether it buys *reasoning* headroom is the open question the main write-up leaves for a 3B→1.5B run.

---

# Measurement-Driven Co-Compression of Transformers
### Allocating a compression budget across depth and width — with a mechanism and a scaling law

*Clean, ordered synthesis of the jun17 study. All numbers are from logged runs
(`results.tsv`, `experiments/`); see `research/04_paper.md` for the full
incremental log and `research/02_results_jun17.md` for the raw ledger.*

---

## Abstract
We study post-hoc compression of pretrained transformers by **pruning a redundant
span of blocks and splicing in one trained replacement** (LLM-Streamline, ICLR'25),
and show that the right move is **not to remove layers alone** but to **co-compress
depth and the surviving blocks' width**. We (i) give a training-free
**Compression-Budget Allocator (CBA)** that *computes* the optimal depth/width split
from cheap redundancy measurements and whose per-layer form **Pareto-beats uniform
pruning**; (ii) explain the effect **mechanistically** (depth is scarce, FFN width is
abundant and low-rank); (iii) validate broadly — **6 GLUE tasks × 2 encoder
architectures**, and **GPT-2 generation**; and (iv) establish a **scaling law**:
compressibility *increases* with model size. The method reaches **~3× parameter
reduction at ~95% retention** on SST-2 and **fits GPT-2-large (774M) on one 8 GB
GPU**, all with minutes of healing and no distillation pretraining.

## 1. Setup
Fine-tuned backbones (a fine-tuned teacher is **required** — a non-fine-tuned teacher
yields a random head and a meaningless `retained_pct`). Surgery: keep the first/last
blocks, remove the contiguous middle span, splice one warm-started block. Metric:
`retained_pct` = compressed metric / teacher metric. Two healing regimes: **full-heal**
(kept blocks trainable) and **frozen** (only the module + head; the PEFT-free setting).
Hardware: one RTX 3070 (8 GB).

## 2. The method: width+depth co-compression + the CBA
**Co-compression.** Remove the redundant middle span (depth) **and** prune the
surviving blocks' FFN width + attention heads (width), then heal. At a fixed budget
this beats depth-only everywhere (§4).

**Compression-Budget Allocator (CBA), training-free.** From one calibration pass:
1. measure per-layer **Block Influence** (BI = 1−cos(in,out)) and per-layer **FFN
   effective rank**; 2. remove the lowest-BI contiguous span; 3. prune each survivor's
FFN to its own 90%-energy width (+ heads); 4. splice + heal.
- *Zero-search validation:* at a 57% target the CBA outputs "remove 8 + FFN→1024" —
  exactly the config that won every empirical sweep.
- *Per-layer Pareto win:* pruning each survivor to its measured width
  (≈[512,1024,1024,2048]) gives **95.5% retained at 61.9% reduction (2 seeds)** —
  the same retention as the best uniform config at **+5 pts more compression**.

## 3. Why it works (mechanism)
On frozen fine-tuned BERT-base: the **deep-middle span has 2.45× lower Block
Influence** than the outer layers (mean BI 0.090 vs 0.221) — depth is *scarce and
concentrated*. Every layer's 3072-d FFN is **massively low-rank** — effective rank
1.3–174, 90% energy in ≤41% of dims — FFN width is *abundant*. Depth-only compression
must eventually cut the scarce resource (outer layers); co-compression cuts the two
redundant ones (middle span + low-rank FFN). This single picture predicts the Pareto
dominance, the "remove deep / keep wide" rule, and why an attention block (which
re-encodes the scarce token-mixing) is the best replacement.

## 4. Main result: width+depth ≫ depth-only, broadly
At matched compression (~57% BERT / ~50% RoBERTa), width+depth beats depth-only on
**every task/architecture** tested:

| task | backbone | width+depth | depth-only | Δ |
|---|---|---|---|---|
| QNLI | BERT-base | 92.6 | 85.5 | **+7.1** |
| QQP | BERT-base | 90.9 | 84.9 | **+5.9** |
| RTE | BERT-base | 84.1 | 81.1 | **+3.0** |
| MRPC | BERT-base | 95.5 | 92.2 | +3.3 |
| SST-2 | BERT-base | 95.5 | 94.0 | +1.5 |
| MRPC | RoBERTa-base | 90.9 | 85.8 | **+5.1** |
| SST-2 | RoBERTa-base | 92.2 | 91.1 | +1.1 |
| WikiText-2 (ppl) | GPT-2 | **dominates** (74 ppl @37% vs 96 @40%) | | |

**Design rule** (matched ~68% budget): keep2+2 (94.3) > keep3+3 (93.4) > keep4+4
(92.2) — *remove the redundant middle deeply, keep the survivors wide.*

## 5. Scaling law: larger transformers are MORE compressible
At matched ~28% parameter reduction (GPT-2 family, WikiText-2, distill-only):

| model | params | retained |
|---|---|---|
| GPT-2 | 124M | 48.0% |
| GPT-2-medium | 355M | 50.5% |
| GPT-2-large | 774M | **58.4%** |

Monotonic over a 6× range — the method *improves* with scale, predicting favorable
extrapolation to billion-parameter LLMs (consistent with larger models carrying more
redundant, lower-rank capacity). The framework fits **GPT-2-large (774M) on 8 GB**
(7.06 GB peak) via frozen-backbone distillation.

## 6. Pareto frontier (SST-2, encoder, full-heal)
| param reduction | retained |
|---|---|
| 19% | 99.0 |
| 32% | 97.6 |
| 45% | 96.9 |
| 57% (width+depth) | 95.5 |
| 67% (depth+width+heads) | 94.7 |
| 75% | ~90 |

DistilBERT-level retention at comparable compression, in minutes, **without
distillation pretraining**.

## 7. What else we tried (ablations)
- **Replacement module** (middle-8, full-heal): attention block best (96.9); glu/hybrid
  94.2; **FNet** (Fourier, 4.7M, faster) 93.9; CNN (0.6M, fastest) 93.4. Compression is
  better spent on the kept blocks than on a fancier module.
- **Accuracy↔latency axis:** replacing surviving attention with parameter-free
  **Fourier mixing** → **2.4× faster** (3.97 ms vs 9.7 ms) for ~6 pts; a latency-optimal
  variant.

## 7a. Weight-tied recurrence: a cautionary non-result
We tested replacing the span with **one block applied K times** (weight-tied,
Universal-Transformer-style, anchored). On **SST-2 frozen, steps=3** it looked like
a strong win — 97.85 vs 95.95 for the single block at *identical* params (3 seeds,
t≈3.4). **But it did not generalize**, so we do NOT bank it:

| recur(×3) setting | retained | vs encoder |
|---|---|---|
| SST-2 frozen (3 seeds) | 97.85 | +1.9 (looked significant) |
| MRPC frozen (2 seeds) | 96.46 | −0.2 (no gain) |
| SST-2 full-heal | 96.03 | ~0 (no gain) |
| steps 2 / 4 / 6 (SST-2 frozen) | ~95.4 each | no gain |

The gain held only for one task, one regime, one step count — MRPC, full-heal, and
the steps-sweep all show no benefit. This is the same pattern as the other honest
negatives: an apparent win on one saturated benchmark that evaporates under
generalization and multi-seed/setting testing. **Methodological lesson reinforced:
a 3-seed gain on a single saturated task is not a result until it generalizes.**

## 7b. What does NOT help on top: concat-fusion, skips
A nuance worth stating precisely: among **single-block** and **parallel
concat-fusion** designs the standard transformer block wins — but a **sequential
anchored 2-block stack** beats it (frozen, 45% budget): **97.2 vs 96.4** (3-seed
means; clearer on MRPC). What does NOT add anything on top:
- **Concat/parallel fusion** (encoder + conv/MLP branches → fuse): `hybrid` 94.2,
  `fusion2` 93.1 — *worse* than a bare block (the fuse dilutes the attention path);
  even a gated, can't-hurt "encoder + correction" (`encfuse`) gives no gain (95.9
  vs 96.1 at more params).
- **Skip connections** in the stack: gated input-skip = 97.1 vs no-skip 97.2 — the
  gates learn ≈0; the sub-blocks' residual stream already provides the skip.
- **Novel attention** (linear/Fourier) as the mixer: 92–94, below the block.

So the *one* architectural lever that helps is depth (a short anchored stack); the
rest is captured by the residual stream + the standard block + co-compression.

## 8. Honest negatives (what did NOT help)
Validated with ≥5 seeds at convergence — each *appeared* to help at low training budget
but the gain was an **undertraining artifact** that vanished: intermediate "anchored"
distillation, thin-deep multi-block stacks, importance-based span selection (picks the
same middle), low-rank "secretly-linear" modules, fusion modules, **iterative
prune-and-heal** (hits a genuine capacity wall at 75%), and **full feature
distillation** (≤+0.4). Methodological caution: compression "improvements" must be
tested at convergence and over multiple seeds.

## 9. Takeaways
1. Co-compress depth **and** width; don't remove layers alone.
2. The optimal split is **computable** from cheap measurements (CBA), per layer,
   beating uniform pruning with zero search.
3. **Mechanism:** depth scarce, FFN width abundant & low-rank.
4. Generalizes across 6 GLUE tasks, 2 encoder architectures, and generation.
5. **Scaling law:** compressibility rises with model size.
6. Reaches ~3× @ ~95%, fits 774M on 8 GB, minutes of healing, no pretraining.

## 10. Limitations
Validated to 774M on one 8 GB GPU; billion-parameter validation (the regime the scaling
law points to) and full LM-eval-harness benchmarks require more compute and are the
clear next step. Extreme (>60%) compression is task-dependent (e.g. CoLA, GPT-2
perplexity degrade faster than SST-2).
