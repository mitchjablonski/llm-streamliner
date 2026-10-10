# Capability-Stratified Compression: What Cheap LLM Compression Silently Destroys

**Qwen2.5-1.5B, block-replacement compression + ~30M-token distillation heal (A10, June–July 2026)**

---

## Abstract

Cheap structured compression of a 1.5B language model — layer/width surgery followed by a small (~30M-token) distillation heal — recovers **88% of the 8-task surface/completion benchmark average at 17% compression** and looks strong by the usual aggregate "retention %" metric. "Benchmark average" here is the mean over `{lambada, hellaswag, arc_easy, piqa, winogrande, boolq, sciq, openbookqa}` (teacher 68.4 → 60.4 = 88.3%) — a basket of the kind structured-compression papers commonly report. **This eight-task average does not establish GSM8K or MMLU retention.** At the *same* 17% cut, multi-step math has already *collapsed* (gsm8k 61.7 → 3.6, 6% retained) while knowledge is degrading (MMLU 59.6 → 36.0, 60%); deepen to 32% (adding depth deletion) and MMLU too collapses to near chance (→ 23) and gsm8k to ~1. Stratifying the eval into surface/completion vs. reasoning/knowledge tasks exposes a **capability crater** that the aggregate hides.

We then localize the loss and test candidate explanations. **In the tested shapes, GSM8K degrades under both residual-width and depth cuts:** continuously with width (gsm8k roughly halves every 64 residual channels removed) and faster with depth removal. Putting an *uncut* model through the identical heal keeps gsm8k at 59.3, which argues against heal-induced forgetting alone as the cause (it does not isolate every surgery-by-heal interaction). At this budget, none of the tested residual reshapes (channel selection, rotation, low-rank capture) or heal objectives (pure-KD, math-data mix, MTP, focus) restored baseline gsm8k; the best, pure-KD with a math-data mix, reached 8.5 (from 4.2) against a 61.7 baseline. At ~30% compression, **no allocation of the cut** — wide-shallow, narrow-deep, or balanced — preserves reasoning. The loss is not explained by the answer-extraction and letter-bias effects we tested (§6.5): gsm8k is identical under strict and flexible extraction on 200 problems, the below-chance MMLU score is letter collapse over chance-level accuracy, and the tested linear probes do not recover teacher-like accuracy. Compression also degrades the model's *calibration* (§6.6): at 32% its MMLU confidence no longer separates right from wrong answers (AUROC 0.51), and a cascade held within 1 point of the teacher on MMLU must escalate 92–98% of items.

The practical consequence: an average over surface/completion tasks is not evidence that a compressed model retains reasoning or knowledge, so compression results should report those capabilities directly. In our tests at 1.5B, no tested shape kept gsm8k near baseline beyond small cuts (a ~4% width cut already drops it from 61.7 to 42.5); whether compressing a larger source model avoids this is an open hypothesis. Task-dependent collapse after layer removal has been reported before — ShortGPT ([Men et al., 2024](https://arxiv.org/abs/2403.03853), §4.6) shows near-zero GSM8K and HumanEval after removing 25% of layers — so the contribution here is not the discovery of the collapse but the cheap-heal controls, the residual-width versus depth comparison, and the confidence/cascade characterization in this setting.

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

This is the case study's core claim: **in this setting, cheap structured compression is capability-stratified, and an eight-task surface average does not reveal the reasoning loss.**

**Relation to prior work.** Task-dependent collapse under layer removal is not new. ShortGPT ([Men et al., 2024](https://arxiv.org/abs/2403.03853)) reports near-zero GSM8K and HumanEval after removing 25% of layers (§4.6) and discusses sharp MMLU drops (§4.3). What this case study adds is (i) controls for the cheap ~30M-token heal (healing an uncut model; reshape, objective and data variants), (ii) a residual-width versus depth comparison at matched compression, and (iii) measurement checks and a confidence/cascade characterization of the compressed models. A fuller related-work survey is still owed.

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

At matched parameters, **depth removal was *more* damaging to gsm8k than width removal** in these runs. One possible reading is that multi-step arithmetic depends on sequential computation across layers, but these runs do not test that mechanism. *(captrade runs use eval_limit=200; the effect size (0 vs 60) dwarfs the sampling noise.)*

### 5.3 The knowledge cliff

MMLU collapses specifically at depth deletion: width-only at 1280 holds MMLU 36.0 (cp17), but removing 3 more layers (25L) drops it to 23.5 (cp22) — chance. Knowledge is distributed in depth.

**Summary of the observed patterns** (Qwen2.5-1.5B, these shapes and this heal; not general laws):
- **Surface tasks** held up under heavy compression on every tested axis.
- **Knowledge (MMLU)** degraded with width and fell to chance at the depth-deletion step tested (28 → 25 layers).
- **Reasoning (gsm8k)** degraded under every tested width and depth cut, fastest under depth.

---

## 6. Testing explanations: heal, reshape, objective, shape

### 6.1 Healing an uncut model (`forget` control)

Put an **uncut** teacher through the *identical* 30M-token heal:

| model | gsm8k | MMLU | arc-c |
|---|---|---|---|
| uncut + 30M heal | **59.3** | 59.5 | 44.3 |
| teacher (reference) | 61.7 | 59.6 | 45.1 |

The heal costs ~2 points of math on an intact model. This argues against heal-induced forgetting **alone** as the cause of the collapse. It does not isolate every surgery-by-heal interaction: in §6.2–6.4 the heal acts on a cut model, which an uncut control cannot fully stand in for.

### 6.2 No reshape recovers reasoning

If reasoning lived in a *recoverable subspace*, a smarter residual reshape at the same compression would preserve it. Four mechanisms, all at 17%, all on the hard basket:

| reshape (17%) | mechanism | gsm8k |
|---|---|---|
| raw energy-drop (baseline) | keep top-energy channels | 3.6 |
| protect-math ×96 / ×192 | reserve math-differential channels | 3.3 / 3.5 |
| SliceGPT rotation | rotate basis, slice empty tail | 2.3 |
| add-channels capture | keep raw + low-rank capture of dropped subspace | 3.0 |

**None of the four tested reshapes restores gsm8k** (2.3–3.6 at 17%). Choosing a different basis did not help in these tests. One hypothesis consistent with this is that the capability depends on more residual dimensions than are kept, and that a C4-only heal does not rebuild it; no theorem or direct measurement here establishes that.

### 6.3 No heal-objective recovers reasoning (`recover` ladder)

| heal variant (cut model) | gsm8k |
|---|---|
| pure-KD, `ce_weight=0` @17% | 4.2 |
| pure-KD + math-data mix @17% | 8.5 |
| pure-KD @32% | 1.1 |

Pure distillation toward the intact parent — the textbook forgetting fix — nudges MMLU slightly (36→38) and barely moves gsm8k: 4.2, or 8.5 with a math-data mix. That is an improvement, but far from the 61.7 baseline at this budget. (Also tested and null on the hard basket: MTP / MTP-KD, longer-context heal, teacher-hard-example mining, focused-KD.)

### 6.4 No *shape* preserves reasoning at real compression (`captrade`)

At ~30% compression, does *allocating* the cut differently save reasoning?

| shape (~1.1B) | width | depth | gsm8k |
|---|---|---|---|
| wide-shallow | 1472 | 19 | 0.0 |
| narrow-deep | 1152 | 28 | 1.0 |
| depth-only | 1536 | 21 | 0.0 |

**No.** Every allocation lands at 0–1. narrow-deep is *marginally* best (1.0 vs 0.0), a directional confirmation that depth is slightly more critical — which also explains why **native small models are narrower-and-deeper**: they protect the critical axis (depth), cut width, and re-learn the narrow-width reasoning *from scratch*, which post-hoc compression cannot do.

### 6.5 Measurement checks: answer extraction, letter bias, linear probes

A collapse this large invites three measurement objections. We rebuilt the 17% (cp17: width 1280, 28L) and 32% (cp29: width 1280, 22L) checkpoints with the identical recipe and re-scored them with per-sample logging (September 2026). The rebuild reproduces the originals (cp29 MMLU 22.99 vs 23.0, arc-c 28.75 vs 28.5; teacher MMLU 59.67), and our option-scoring agrees with lm-eval's own argmax on all 22,698 multiple-choice items.

**Objection 1: gsm8k fails on answer *format*, not arithmetic.** Everything above was scored with lm-eval's `strict-match` filter, which requires a literal `#### N`. Re-scoring the same 200 problems with `flexible-extract` (last number in the response) changes nothing:

| model | strict-match | flexible-extract |
|---|---|---|
| teacher | 61.5 | 61.5 |
| cp17 (17%) | 4.0 | 4.0 |
| cp29 (32%) | 1.5 | 1.5 |

The failure modes differ in kind: cp17 keeps the format and the solution structure but gets the arithmetic wrong (e.g. "she sells the remaining 16 eggs" without subtracting); cp29 frequently degenerates (restating the question). Neither is a scoring artifact.

**Objection 2: MMLU *below* chance (23 < 25) is a scoring pathology.** Partly yes — and the pathology is diagnostic. cp29 predicts option **A on 98% of items** (gold share of A: 23%), so its 23% is exactly the A-share. cp17 shows the inverse bias, almost never predicting B (2%). Dividing out each model's letter prior (fit on a hash-split half, scored on the other half) gives:

| model | holdout acc | letter-debiased acc | AUROC after debias |
|---|---|---|---|
| teacher | 59.9 | 60.1 | 0.78 |
| cp17 | 35.3 | 37.1 | 0.63 |
| cp29 | 23.2 | **25.1** | **0.51** |

Below-chance was letter collapse; underneath it, cp29's option likelihoods carry no usable MMLU signal (chance accuracy, uninformative ranking).

**Objection 3: the knowledge is present but *inaccessible* to option scoring.** A frozen-feature linear probe (last-token hidden state at 50/75/100% depth, 4-way logistic regression, 5-fold CV grouped by subject, with a randomly initialized Qwen2.5-0.5B as a rough control for surface cues — note it is **not** shape-matched to cp29):

| model | best probe acc (layer) | zero-shot MMLU |
|---|---|---|
| teacher | 55.0 (21/28) | 59.7 |
| cp17 | 36.9 (21/28) | 36.1 |
| cp29 | **27.8** (11/22) | 23.0 |
| random-init Qwen2.5-0.5B (control, not shape-matched) | 25.7 | — |

For every model the probe tracks zero-shot accuracy, and the tested probes **do not recover teacher-like accuracy** from cp29: 27.8 ± 1.1 vs 25.7 ± 0.6 for the control, where ± is the spread across subject-grouped folds, not a significance test of the difference. We therefore claim neither erasure (a nonlinear readout could find more) nor a statistically resolved residual signal.

**Verdict.** Within this scope — Qwen2.5-1.5B, this surgery and ~30M-token heal budget, these tasks — the loss is not explained by the tested answer-extraction and letter-bias effects; the strict/flexible comparison covers 200 problems.

### 6.6 A second metric warning: calibration degrades with compression

If a compressed model at least *knew* when it was out of its depth, its confidence could route hard inputs back to the parent (a cascade), recovering quality cheaply. On MMLU, the knowledge-heavy task, it increasingly cannot:

| model | MMLU acc | AUROC (confidence → correct) | ECE | mean conf, right / wrong |
|---|---|---|---|---|
| teacher | 59.7 | 0.78 | 8.6 | 0.77 / 0.55 |
| cp17 (17%) | 36.1 | 0.62 | 4.0 | 0.41 / 0.37 |
| cp29 (32%) | 23.0 | **0.51** | 32.6 | 0.56 / 0.56 |

Confidence degrades with compression and is **near-uninformative at the deeper cut**: cp29 is equally confident on its right and wrong answers, and badly overconfident (ECE 32.6). cp17 is degraded but still informative. In a holdout cascade (escalation threshold chosen on one half, applied to the other) that must stay within 1 point of the teacher **on MMLU**, cp17 escalates 92% of items and cp29 98% — little saving. This is an MMLU result at that target, not a general statement about cascades; on surface tasks the picture is milder (BoolQ AUROC teacher 0.70, cp17 0.63, cp29 0.55).

The practical point extends §4: in this setting, compression removed a capability **and** degraded the model's ability to signal that it had lost it, so confidence-based safeguards cannot be assumed to survive compression and must be re-measured. (For scale: a *native* Qwen2.5-0.5B — 494M parameters, under half of cp29's 1052M — scores MMLU 47.5 with AUROC 0.71 and gsm8k 26.5 on the same harness.)

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

1. **Report capability-stratified evals.** A structured-compression result without gsm8k/MMLU is uninterpretable — the aggregate hides whether reasoning survived. An eight-task surface average alone should not be read as reasoning or knowledge retention. Report generative tasks under both strict and flexible extraction, and check multiple-choice tasks for answer-letter collapse before reading a below-chance score.
1b. **Re-measure calibration after compression.** Confidence can degrade alongside the capability (§6.6); a compressed model should not inherit the parent's confidence thresholds, abstention rules, or escalation policies without re-validation.
2. **In our tests, no shape preserved gsm8k beyond small cuts.** A ~4% residual-width cut already drops it from 61.7 to 42.5 and a 12.5% cut to 9.3; at ~30% every tested allocation lands at 0–1, and no tested operator or heal at this budget restored it. We do not claim a general maximum compression ratio: the tolerable loss depends on the application.
3. **The most promising remaining lever is source scale — but it is a hypothesis, not a result.** Native-1.5B (width 1536, 28L) reasons fine, so that absolute geometry is reasoning-capable; compressing a *bigger* source (3B: width 2048, 36L) toward it *might* land above the reasoning thresholds — **if** they are absolute rather than relative, and **if** reasoning benefits from scale where surface (§8) has saturated. Neither is established. This is the **H100 3B→1.5B experiment** (`scripts/h100_bigteach_3b15b.sh`, rewritten as a compression-ratio sweep with hard-basket evals from the start): the one run that could turn the negative story into a positive one — or confirm the wall holds at scale.

---

## 10. Limitations

- Single family (Qwen2.5) at a single scale (1.5B). Cross-family confirmation of the crater (SmolLM2/TinyLlama, hard basket) is the recommended next check to promote the finding from "Qwen fact" to "transformer fact."
- Heal fixed at ~30M tokens by design (few/zero-shot regime). We do **not** claim reasoning is unrecoverable with large-scale retraining — only that it is unrecoverable *cheaply*, which is the regime of interest.
- gsm8k measured 5-shot generative; the reasoning-collapse conclusion is robust to the sampling limit because the effect (60→0) is an order of magnitude larger than the noise. The §6.5 strict/flexible re-score uses 200 problems.
- The linear probe (§6.5) bounds *linearly* accessible knowledge only; it does not establish erasure, and its random-init control is not shape-matched to the probed checkpoint.
- Related work: task-dependent collapse under layer removal was reported by ShortGPT (§4.6); a fuller survey of capability-stratified compression evaluations is still owed. The calibration results (§6.6) are measured on MMLU, BoolQ, ARC and PIQA for the two rebuilt checkpoints; the cascade figures are for MMLU at a within-1-point target.

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

§6.5–6.6 (September 2026 rebuild of cp17/cp29 with the identical recipe): `results/a10_calib_summary.txt` and
`results/calib_summary_v2.md` (per-task accuracy, ECE, AUROC, strict/flexible gsm8k, letter bias, holdout cascade),
produced by `scripts/calib_eval.py` and `scripts/calib_compare.py`; the letter-debias table by `scripts/letter_debias.py`
(`results/a10_letter_debias_summary.txt`); the linear probe by `scripts/mmlu_probe.py`
(`results/a10_probe_summary.txt`; layer accuracies ± fold spread).
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
| forget | uncut-heal reference | 59.3 | NEUTRAL (heal alone does not explain the loss) |
| aspect / nshape / native | off keep-full shape | ≤79% surface, gsm8k dead | LOST |
| meta / richmeta | amortized "meta-ReplaceMe" operator | 13–700× worse than per-span oracle | NEUTRAL (learned-operator bet closed) |
