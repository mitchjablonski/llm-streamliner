# Capability-Stratified Compression: What Cheap LLM Compression Silently Destroys

**Qwen2.5-1.5B, block-replacement compression + ~30M-token distillation heal (A10, June–July 2026)**

---

## Abstract

Cheap structured compression of a 1.5B language model — layer/width surgery followed by a small (~30M-token) distillation heal — recovers **88% of the 8-task surface/completion benchmark average at 17% compression** and appears, by the usual aggregate "retention %" metric, to be a strong, near-SOTA result. "Benchmark average" here is the mean over `{lambada, hellaswag, arc_easy, piqa, winogrande, boolq, sciq, openbookqa}` (teacher 68.4 → 60.4 = 88.3%) — the basket structured-compression papers typically report. **This number is an artifact of the benchmark basket.** At the *same* 17% cut, multi-step math is already *annihilated* (gsm8k 61.7 → 3.6, 6% retained) while knowledge is degrading (MMLU 59.6 → 36.0, 60%); deepen to 32% (adding depth deletion) and MMLU too collapses to near chance (→ 23) and gsm8k to ~1. Stratifying the eval into surface/completion vs. reasoning/knowledge tasks exposes a **capability crater** that the aggregate hides.

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

On the **surface basket**, the compression frontier is convex and reproducible (±0.1), and **the simplest operator wins**. A ~70-lever search (§7, Appendix B) found nothing that beats plain keep-full at matched parameters.

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

Take a compressed model at **17% compression** (residual width 1280, keep-full, healed) — the point where the surface/completion mean is a healthy **88%**. Now read every task:

| task | kind | teacher | compressed | retained |
|---|---|---|---|---|
| sciq | surface | 93.3 | 90.0 | 96% |
| piqa | surface | 76.1 | 70.6 | 93% |
| lambada | surface | 62.0 | 48.7 | 79% |
| **surface/completion mean (8 tasks)** | — | **68.4** | **60.4** | **88%** |
| arc-challenge | reasoning | 45.1 | 34.1 | 76% |
| MMLU | knowledge | 59.6 | 36.0 | 60% |
| **gsm8k** | math | 61.7 | 3.6 | **6%** |

**A single aggregate "retention %" averages a capability that is *intact* (surface, 88%) with one that is *destroyed* (math, 6%).** The model still completes sentences and answers commonsense questions; it can no longer do multi-step arithmetic. Knowledge (MMLU) is degrading here (60%) and collapses to chance (~23) only under deeper, depth-including compression (§5). No amount of surface-benchmark reporting reveals this — you have to evaluate reasoning and knowledge directly.

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

A large operator and distillation search — **~70 distinct levers across 82 experiment queues** (the full grounded catalog is in **Appendix B**, backed by the summaries in `results/`) — converged on a small set of "laws" and a long list of dead ends. **Every lever below was ≤ the plain keep-full frontier**, or null on the capability crater:

- **Reshape / capture (all fail):** whitened/output-aware SVD, hybrid-capture, ghost-capture, orthogonal add-channels (Schur), CUR, SliceGPT rotation, closed-form fold/absorption, EoRA.
- **Structural (all fail):** layer-merge (LaCo), layer-tie (MobileLLM weight sharing), full-width bus (per-layer read/write index sets — routing incoherence), FFN→depth reallocation (Net2DeeperNet + lstsq), embedding factorization, canon conv-mixers (net-new token mixing).
- **Distillation (all ≈ vanilla at 30M tokens):** reverse-KL / JSD / DKD / GKD, feature/relation/attention distillation, focus weighting, MTP and MTP-KD, hard-example mining, longer context; **data** (C4 > teacher-generated > instruction/SlimOrca > math-mix); **teacher** (parent > big-teacher, monotonically worse).

The consolidated design principles ("three-clause law"): (1) don't modify pretrained weights (fold/merge fail); (2) don't constrain them to new roles (tie fails); (3) don't break residual-stream global coherence (bus fails). FFN units behave as key-value memories (raw cuts destroy knowledge). The heal is an information-bound, ~4%-norm **dense** polish (per-matrix SVD of healed−surgical: median r90/dim ≈ 0.76) — closing the "learned/amortized healing operator" bet at the weight level.

---

## 8. Scale and compressibility — and what it does *not* tell us about reasoning

Two earlier scaling sweeps (GPT-2 and Pythia families; details and mechanism in the companion [`prior_work_co_compression.md`](prior_work_co_compression.md)) bracket how compressibility moves with model size:

| family / metric | scale range | compression | retained | trend |
|---|---|---|---|---|
| GPT-2 · WikiText ppl | 124M → 355M → 774M | ~28% | 48.0 → 50.5 → **58.4** | rises with scale |
| Pythia · surface downstream | 1.4B → 2.8B → 6.9B | ~20% | 86.1 → 86.3 → **86.6** | flat (saturated) |

Compressibility **rises with size at small scale and then saturates** — the redundancy plateau bigger models are supposed to provide is largely already reached by the low billions (pythia is flat from 1.4B to 6.9B). **Every figure here is a surface / perplexity metric; gsm8k was never evaluated at scale in these sweeps** (the native Qwen references are surface-only: 1.5B 68.1, 3B 72.0, 7B 76.2 avg).

So the scaling evidence is a **caution, not a promise**, for the source-scale hypothesis below: a bigger source does not obviously buy *surface* retention headroom past ~1.5B — surface has already saturated. The live bet is that **reasoning behaves differently** — that the larger source's redundancy, which surface no longer needs, is exactly what a reasoning-preserving compression could draw on. Nothing in the surface scaling law confirms or refutes this; it is precisely the gap a capability-stratified 3B→1.5B run would fill.

---

## 9. Implications

1. **Report capability-stratified evals.** A structured-compression result without gsm8k/MMLU is uninterpretable — the aggregate hides whether reasoning survived. This reframes existing "retention %" numbers as surface-task numbers.
2. **Reasoning-preserving cheap compression at 1.5B is ~4–8% max**, by any shape. Below that, math is gone and no cheap operator or heal brings it back.
3. **The most promising remaining lever is source scale — but it is a hypothesis, not a result.** Native-1.5B (width 1536, 28L) reasons fine, so that absolute geometry is reasoning-capable; compressing a *bigger* source (3B: width 2048, 36L) toward it *might* land above the reasoning thresholds — **if** they are absolute rather than relative, and **if** reasoning benefits from scale where surface (§8) has saturated. Neither is established. This is the **H100 3B→1.5B experiment** (`scripts/h100_bigteach_3b15b.sh`, rewritten as a compression-ratio sweep with hard-basket evals from the start): the one run that could turn the negative story into a positive one — or confirm the wall holds at scale.

---

## 10. Limitations

- Single family (Qwen2.5) at a single scale (1.5B). Cross-family confirmation of the crater (SmolLM2/TinyLlama, hard basket) is the recommended next check to promote the finding from "Qwen fact" to "transformer fact."
- Heal fixed at ~30M tokens by design (few/zero-shot regime). We do **not** claim reasoning is unrecoverable with large-scale retraining — only that it is unrecoverable *cheaply*, which is the regime of interest.
- gsm8k measured 5-shot generative; the reasoning-collapse conclusion is robust to the sampling limit because the effect (60→0) is an order of magnitude larger than the noise.

---

## Appendix A: Data provenance

All numbers trace to run summaries shipped in `results/`:
`a10_wthresh_summary.txt` (§5.1), `a10_captrade_summary.txt` (§5.2, §6.4),
`a10_forget_summary.txt` (§6.1), `a10_reshape_summary.txt` (§6.2),
`a10_recover_summary.txt` (§6.3), `a10_hardloc_summary.txt` (§5.3),
`a10_kverify_summary.txt` (§3), `a10_peraxis_summary.txt` / `a10_frontier2_summary.txt` (§2),
the scaling sweeps in `scaling_law_gpt2_pythia.tsv` (§8), and the full set of **82 lever summaries**
backing the negative map (§7, Appendix B). 454 raw run logs are archived with the authors; per-run
params+shape are recorded in each summary line. The earlier co-compression study and mechanism are in
the companion [`prior_work_co_compression.md`](prior_work_co_compression.md).

---

## Appendix B: The full negative map (~70 levers)

Every lever we tried on Qwen2.5-1.5B, grouped, with its best result and verdict against the plain
keep-full frontier (surface-8 teacher ≈ 68.4; frontier ≈ 79% @ 32% compression → 88.5% @ 17%).
Capability-basket runs (different teacher/tasks) are marked **NEUTRAL** — they answer the crater
question, not the frontier question. Each row traces to a `results/a10_<lever>_summary.txt` file.

**What won.** Plain keep-full width-prune (raw activation-energy) + an adequate C4 heal is the champion — 88.5% @ ~17%, ~79–80% @ ~32%; nothing displaced it. The *only* lever to edge it was iterative-staged surgery (`iterbest`, +~1.5 at the aggressive band). Everything that merely matched it added no value (all KD-loss variants, extra heal tokens, capture bases). Everything else lost. And no lever recovered the crater — math capability craters regardless of operator, loss, data, or shape.

### B.1 Reshape / capture (rotation, SVD, fold, low-rank capture, channel selection)

| lever | what it tests | best | verdict |
|---|---|---|---|
| outraw | raw activation-energy width prune (+sink-pin) — **the baseline** | 88.5% | BASELINE (sink-pin adds nothing) |
| capture | capture-basis variants (rrr/ortho/wanda/combine/pls) | combine 88.4% | TIED |
| refit | extra heal tokens on capture (10M/100M) | 88.8% | TIED (just more data) |
| combosolver | joint OSAL/SAL closed-form solver | 87.2% | LOST |
| eorapush / svdcap | push EoRA / SVD-basis capture rank | 85.4 / 81.1% | LOST |
| head / head2 | GQA query-head prune (wanda>energy>norm criterion) | 87.0% | LOST |
| hetero | heterogeneous per-layer FFN widths | 76.4% | LOST |
| perlayer | per-layer SliceGPT rotate + slice residual | 59.1% | LOST (badly) |
| decomp | SliceGPT slice vs layer-deletion | del 77.1 vs slice 64.4 | LOST (decomposition loses to deletion) |
| absorb | absorb dropped width into kept (fold) | ≤67.7% | LOST |
| multiaxis / cov / ridge / hybffn / deepcap / stack / dirtransfer / matched | assorted capture/rotation bases | 62–79% | LOST |

### B.2 Structural (merge/tie, bus, reallocation, embedding factorization, added compute)

| lever | what it tests | best | verdict |
|---|---|---|---|
| iterbest | iterative staged surgery (s1→s2→final) ± EoRA | **80.5%** | **BEAT** (marginal, ~+1.5 at aggressive band — the one process lever that edged the frontier) |
| depthcap | add low-rank depth-capture block | 89.8% (light) | TIED on light frontier (collapses at deep) |
| prog | progressive shrink vs one-shot | 87.6% | LOST (≤ one-shot) |
| micro | added-compute micro-ops (recur/stack/conv/attn) | 87.1% (only at tiny compression) | LOST |
| busffn | layer-tie / hidden-bus / FFN-widen / emb-factor | ≤76.7% | LOST |
| realloc | reallocate pruned width into new depth blocks | 54.7% | LOST (badly) |
| addon | add canon / MTP prediction heads | 79.0% ≈ base | TIED-at-baseline (heads add nothing) |
| staged / seq / joint / wider / linattn / kill25 | staging, sequential, off-shape, linear-attn | 63–77% | LOST |

### B.3–B.5 Distillation — loss, data/teacher, harder task

| lever | what it tests | best | verdict |
|---|---|---|---|
| kdpush | vanilla vs feat / rev-KL / DKD / relation / all | dkd 88.6 ≈ vanilla 88.5 | TIED (no loss beats vanilla KD) |
| gkd / terminal | JSD-GKD; focus (topk/disagree/fusion) | ≤88.3% | TIED / LOST |
| healdata | heal corpus C4 vs instruct vs teacher-gen | **C4 79.0** > teacher 78.9 > instruct 77.6 | LOST (plain C4 is best) |
| retpush / gapcloser | bidirectional + big-teacher KD | ≤80.6% (many OOM) | LOST |
| harder | MTP / MTP-KD / seq1024 / hard-mining | 66.9–68.0% (flat) | LOST/NEUTRAL (harder task gives no lift) |
| biggerbase | 3B as teacher | mostly OOM on 23 GB | INCONCLUSIVE (memory-blocked → the H100) |

### B.6 Capability recovery (the crater — all NEUTRAL, none restore reasoning)

| lever | what it tests | result (gsm8k) | verdict |
|---|---|---|---|
| wthresh | width-threshold sweep, hard basket | 42.5 → 9.3 as width 1472→1344 | shows the reasoning cliff |
| recover | pure-KD ± math mix @17/29% | 4.2 / 8.5 / 1.1 | NEUTRAL (math barely moves) |
| reshape | protect-math / slice / add-capture, hard basket | 2.3–3.6 | NEUTRAL (basis doesn't restore math) |
| captrade | shape trade (deep/wide/narrow) | ≈0 in all | NEUTRAL (no shape preserves math) |
| forget | uncut-heal reference | 59.3 | NEUTRAL (heal is innocent) |
| aspect / nshape / native | off keep-full shape | ≤79% surface, gsm8k dead | LOST |
| meta / richmeta | amortized "meta-ReplaceMe" operator | 13–700× worse than per-span oracle | NEUTRAL (learned-operator bet closed) |
