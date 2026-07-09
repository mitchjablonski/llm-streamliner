# Capability-Stratified Compression: What Cheap LLM Compression Silently Destroys

**Qwen2.5-1.5B, block-replacement compression + ~30M-token distillation heal (A10, June–July 2026)**

---

## Abstract

Cheap structured compression of a 1.5B language model — layer/width surgery followed by a small (~30M-token) distillation heal — recovers **~88% of average benchmark performance at 17% compression** and appears, by the usual aggregate "retention %" metric, to be a strong, near-SOTA result. **This number is an artifact of the benchmark basket.** When we stratify the evaluation into surface/completion tasks (piqa, hellaswag, lambada, …) versus reasoning and knowledge tasks (gsm8k, MMLU, arc-challenge), the aggregate retention hides a **capability crater**: surface tasks survive (~90%), while multi-step reasoning is *annihilated* (gsm8k 61.7 → ~1) and factual knowledge collapses to near chance (MMLU 59.6 → ~23).

We then localize and causally isolate the destruction. **Reasoning requires the near-full network — both residual width and depth.** It degrades continuously with width (gsm8k halves roughly every 64 residual channels) and is destroyed *even faster* by depth removal. The heal is causally *innocent* (an uncut model put through the identical heal keeps gsm8k at 59.3); the *surgery* destroys the capability. And it is unrecoverable cheaply: **no** reshape of the residual (channel-selection, rotation, low-rank capture) and **no** heal-objective (pure-KD, math-data, MTP, focus) restores it. At ~30% compression, **no allocation of the cut** — wide-shallow, narrow-deep, or balanced — preserves reasoning.

The practical consequence: every "retention %" in the structured-compression literature is a *surface-task* number that says nothing about whether the compressed model can still reason. Reasoning-preserving cheap compression at 1.5B is limited to single-digit percentages by any shape; recovering it at meaningful compression ratios appears to require **source scale**, not a better operator or a better heal.

---

## 1. Setup

- **Base model:** Qwen2.5-1.5B (28 layers, hidden 1536, 1543.7M params). Teacher for the heal and the retention denominator.
- **Compression:** block-replacement surgery — clean layer deletion, residual-width pruning by activation energy, attention/FFN internal-dim reshaping, optional ReplaceMe-style closed-form splice. The unbeaten default is **keep-full**: keep attention/FFN internal dims full, narrow the residual width, delete depth cleanly.
- **Heal:** ~30M tokens, C4, loss `0.5·CE + 0.5·logit-KD(parent)`, ~4000 steps, lr 3e-5. Deliberately **cheap** — the research goal is few/zero-shot-preserving compression, so gains must come from the *operator and distillation quality*, not token volume.
- **Evaluation — the dual basket:**
  - **Surface/completion (easy):** lambada, hellaswag, arc-easy, piqa, winogrande, boolq, sciq, openbookqa
  - **Reasoning/knowledge (hard):** gsm8k (5-shot generative), MMLU, arc-challenge
- **Teacher baseline (11-task, full eval):** avg 64.86 — gsm8k **61.71**, MMLU **59.58**, arc-c **45.05**, lambada 61.98, piqa 76.12, sciq 93.3, hellaswag 67.88. (A parallel `eval_limit=200` cache reads teacher 64.38 / gsm8k 61.5; used only where noted.)

> Retention % = compressed avg / teacher avg on the given basket. Unless stated, numbers are the full 11-task eval.

---

## 2. The compression recipe and its (surface) frontier

On the **surface basket**, the compression frontier is convex and reproducible (±0.1), and **the simplest operator wins**. A 16+-lever search (§7) found nothing that beats plain keep-full at matched parameters.

| point | params | compression | surface retention |
|---|---|---|---|
| width-1344 (keep-full) | 1350.8M | 12.5% | **92.5%** |
| one-shot os26 | 1208.4M | 21.7% | 83.7% |
| one-shot os24 | 1130.4M | 26.8% | 80.2% |
| lm6 (depth-6) | 1052.4M | 31.8% | 73.0% |

Per-axis sensitivity (surface basket) already hints at the story to come — the residual width is the most destructive axis, attention the least:

| axis cut | params | surface retention |
|---|---|---|
| attention reshape (attn_hyb) | 1502.4M | 98.3% |
| FFN reshape (ffn_hyb) | 1306.8M | 89.0% |
| width → 1344 | 1350.8M | 92.5% |
| depth −4 (ReplaceMe splice) | 1403.3M | 91.8% |

This all looks healthy. **It is entirely misleading**, as the next section shows.

---

## 3. Fairness vs. a training-free baseline (ReplaceMe)

We measured ReplaceMe's method *in our own harness* on Llama-3.2-1B (25.4% compression, depth+width to 870.9M) to get an honest, budget-controlled comparison:

| method | heal | retention |
|---|---|---|
| ReplaceMe surgery, training-free | 0 tokens | **52.3%** |
| + instruction-calibrated splice | 0 tokens | 57.4% |
| ReplaceMe surgery **+ our 30M heal** | 30M | **73.5%** |
| our surgery, matched shape (w1920_d6) | 30M | 73.3% |

**The heal is the dominant factor (+16–21 points); the surgery choice is worth ~2.** The honest claim is therefore an *exchange rate* — "≈3 GPU-hours of cheap heal buys +16 over training-free" — not a same-budget knockout. (Note: a widely-cited "ReplaceMe Llama-3.2-1B @25% = 87.7%" figure is **not in the paper** — their smallest evaluated model is Llama-2-7B. Small-model bars must be measured, not cited.)

---

## 4. The central finding: the capability crater

Take a strong compressed model — 32% compression, keep-full, healed — with a respectable **surface** retention. Now read the hard basket:

| task | teacher | compressed (32%) | retained |
|---|---|---|---|
| piqa (surface) | 76.1 | ~68 | ~89% |
| lambada (surface) | 62.0 | ~38 | ~61% |
| **gsm8k (math)** | **61.7** | **~1** | **~2%** |
| **MMLU (knowledge)** | **59.6** | **~23** | **below chance** |
| **arc-challenge** | 45.1 | ~28 | ~62% |

**A single aggregate "retention %" averages a capability that is *intact* with one that is *destroyed*.** The model still completes sentences and answers commonsense questions; it can no longer do multi-step arithmetic and has lost most of its factual knowledge. No amount of surface-benchmark reporting reveals this — you have to evaluate reasoning and knowledge directly.

This is the paper's core claim: **cheap structured compression is capability-stratified. Reported retention numbers are surface-task numbers that are blind to reasoning collapse.**

---

## 5. Localization: which axis kills which capability

### 5.1 The reasoning–width curve (width-only prune, full depth, full attn/FFN)

Sweeping *only* residual width (keeping all 28 layers) maps a steep, **continuous** decline — not a hard cliff:

| residual width | compression | gsm8k | MMLU | arc-c |
|---|---|---|---|---|
| 1536 (native) | 0% | 61.7 | 59.6 | 45.1 |
| **1472** | 4% | **42.5** | 55.4 | 43.5 |
| 1408 | 8% | 23.2 | 51.1 | 40.5 |
| 1344 | 12.5% | 9.3 | 44.2 | 38.3 |
| 1280 | 17% | ~3.6 | 36.0 | 34.1 |

gsm8k roughly **halves for every ~64 residual channels removed**. There is no "safe" plateau — the cost begins immediately (even a 4% width cut loses 31% of gsm8k). This is the signature of a **high-dimensional capability degrading as its dimensions are removed**, not an all-or-nothing bandwidth gate.

### 5.2 Depth kills reasoning *too* — and faster

The earlier "width→math, depth→knowledge" split was **too clean**. Cutting depth at *full* width destroys math outright:

| shape (~1.2B, ~30% compression) | width | depth | gsm8k | MMLU |
|---|---|---|---|---|
| depth-only −8L | 1536 | 21 | **0.0** | 27.4 |
| — vs width-cut to ~1216M | ~1216 | 28 | ~6 (interp.) | ~34 |

At matched parameters, **depth removal is *more* damaging to reasoning than width removal.** gsm8k is sequential arithmetic — remove layers and the reasoning chain cannot complete, regardless of width. *(captrade runs use eval_limit=200; the effect size (0 vs 60) dwarfs the sampling noise.)*

### 5.3 The knowledge cliff

MMLU collapses specifically at depth deletion: width-only at 1280 holds MMLU 36.0 (cp17), but removing 3 more layers (25L) drops it to 23.5 (cp22) — chance. Knowledge is distributed in depth.

**Summary of the capability laws:**
- **Surface tasks** tolerate heavy compression on every axis.
- **Knowledge (MMLU)** needs depth; collapses at the depth-deletion cliff.
- **Reasoning (gsm8k)** needs the *near-full network* — both width and depth — and is *most* fragile to depth.

---

## 6. Causal isolation: it's the cut, and it's unrecoverable cheaply

### 6.1 The heal is innocent (`forget` control)

Put an **uncut** teacher through the *identical* 30M-token heal:

| model | gsm8k | MMLU | arc-c |
|---|---|---|---|
| uncut + 30M heal | **59.3** | 59.5 | 44.3 |
| teacher (reference) | 61.7 | 59.6 | 45.1 |

The heal costs ~2 points of math on an intact model. **Catastrophic forgetting from the C4 heal is *not* the cause** — the surgery is. This matters: it rules out "fix the heal objective" as the lever.

### 6.2 No reshape recovers reasoning

If reasoning lived in a *recoverable subspace*, a smarter residual reshape at the same compression would preserve it. Four mechanisms, all at 17%, all on the hard basket:

| reshape (17%) | mechanism | gsm8k |
|---|---|---|
| raw energy-drop (baseline) | keep top-energy channels | 3.6 |
| protect-math ×96 / ×192 | reserve math-differential channels | 3.3 / 3.5 |
| SliceGPT rotation | rotate basis, slice empty tail | 2.3 |
| add-channels capture | keep raw + low-rank capture of dropped subspace | 3.0 |

**Unanimous failure.** It is not *which basis* you keep — it is *dimensionality*. You cannot linearly project a near-full-rank capability into fewer dimensions without destroying it, and the distill cannot rebuild what the projection deleted (it operates on C4, which never exercises math). Math-based weight transfer (ReplaceMe, activation-aware SVD, closed-form fold) is subject to the same wall.

### 6.3 No heal-objective recovers reasoning (`recover` ladder)

| heal variant (cut model) | gsm8k |
|---|---|
| pure-KD, `ce_weight=0` @17% | 4.2 |
| pure-KD + math-data mix @17% | 8.5 |
| pure-KD @32% | 1.1 |

Pure distillation toward the intact parent — the textbook forgetting fix — nudges MMLU slightly (36→38) but does **nothing** for gsm8k. Combined with §6.1, the capability is *structurally gone*, not forgotten. (Also tested and null on the hard basket: MTP / MTP-KD, longer-context heal, teacher-hard-example mining, focused-KD.)

### 6.4 No *shape* preserves reasoning at real compression (`captrade`)

At ~30% compression, does *allocating* the cut differently save reasoning?

| shape (~1.1B) | width | depth | gsm8k |
|---|---|---|---|
| wide-shallow | 1472 | 19 | 0.0 |
| narrow-deep | 1152 | 28 | 1.0 |
| depth-only | 1536 | 21 | 0.0 |

**No.** Every allocation lands at 0–1. narrow-deep is *marginally* best (1.0 vs 0.0), a directional confirmation that depth is slightly more critical — which also explains why **native small models are narrower-and-deeper**: they protect the critical axis (depth), cut width, and re-learn the narrow-width reasoning *from scratch*, which post-hoc compression cannot do.

---

## 7. The negative map (what does *not* help)

A large operator and distillation search — ~90 experiment queues — converged on a small set of "laws" and a long list of dead ends. **Every lever below was ≤ the plain keep-full frontier**, or null on the capability crater:

- **Reshape / capture (all fail):** whitened/output-aware SVD, hybrid-capture, ghost-capture, orthogonal add-channels (Schur), CUR, SliceGPT rotation, closed-form fold/absorption, EoRA.
- **Structural (all fail):** layer-merge (LaCo), layer-tie (MobileLLM weight sharing), full-width bus (per-layer read/write index sets — routing incoherence), FFN→depth reallocation (Net2DeeperNet + lstsq), embedding factorization, canon conv-mixers (net-new token mixing).
- **Distillation (all ≈ vanilla at 30M tokens):** reverse-KL / JSD / DKD / GKD, feature/relation/attention distillation, focus weighting, MTP and MTP-KD, hard-example mining, longer context; **data** (C4 > teacher-generated > instruction/SlimOrca > math-mix); **teacher** (parent > big-teacher, monotonically worse).

The consolidated design principles ("three-clause law"): (1) don't modify pretrained weights (fold/merge fail); (2) don't constrain them to new roles (tie fails); (3) don't break residual-stream global coherence (bus fails). FFN units behave as key-value memories (raw cuts destroy knowledge). The heal is an information-bound, ~4%-norm **dense** polish (per-matrix SVD of healed−surgical: median r90/dim ≈ 0.76) — closing the "learned/amortized healing operator" bet at the weight level.

---

## 8. Implications

1. **Report capability-stratified evals.** A structured-compression result without gsm8k/MMLU is uninterpretable — the aggregate hides whether reasoning survived. This reframes existing "retention %" numbers as surface-task numbers.
2. **Reasoning-preserving cheap compression at 1.5B is ~4–8% max**, by any shape. Below that, math is gone and no cheap operator or heal brings it back.
3. **The path to reasoning-preserving compression is source scale, not a better operator.** Native-1.5B (width 1536, 28L) reasons fine; a *bigger* source (3B: width 2048, 36L) compressed toward 1.5B lands at an absolute width/depth the 1.5B budget proves is reasoning-capable — *if* the threshold is absolute rather than relative. This is the **H100 3B→1.5B hypothesis** (`scripts/h100_bigteach_3b15b.sh`, keep depth + width above threshold, hard-basket evals from the start). It is the one experiment that could move the top line.

---

## 9. Limitations

- Single family (Qwen2.5) at a single scale (1.5B). Cross-family confirmation of the crater (SmolLM2/TinyLlama, hard basket) is the recommended next check to promote the finding from "Qwen fact" to "transformer fact."
- Heal fixed at ~30M tokens by design (few/zero-shot regime). We do **not** claim reasoning is unrecoverable with large-scale retraining — only that it is unrecoverable *cheaply*, which is the regime of interest.
- gsm8k measured 5-shot generative; the reasoning-collapse conclusion is robust to the sampling limit because the effect (60→0) is an order of magnitude larger than the noise.

---

## Appendix: data provenance

All numbers trace to pulled A10 run summaries (`cloud_results/a10_final/`):
`a10_wthresh_summary.txt` (§5.1), `a10_captrade_summary.txt` (§5.2, §6.4),
`a10_forget_summary.txt` (§6.1), `a10_reshape_summary.txt` (§6.2),
`a10_recover_summary.txt` (§6.3), `a10_hardloc_summary.txt` (§5.3),
`a10_kverify_summary.txt` (§3), `a10_peraxis_summary.txt` / `a10_frontier2_summary.txt` (§2).
82 summaries + 454 run logs archived locally; per-run params+shape recorded in each summary line.
