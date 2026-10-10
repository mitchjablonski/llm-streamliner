# LLM-Streamliner — Capability-Stratified Compression

**Cheap structured compression of a 1.5B language model keeps 88% of the surface/completion benchmark average at 17% compression — while its GSM8K math score collapses (61.7 → 3.6, 6% retained).**

Here "benchmark average" means the mean over the **8 surface/completion tasks** — `lambada, hellaswag, arc_easy, piqa, winogrande, boolq, sciq, openbookqa` — a basket of the kind structured-compression papers commonly report. At the same 17% cut, GSM8K (math) has already collapsed and MMLU (knowledge) is degrading; push to 32% (adding depth deletion) and MMLU too falls to chance. This repository contains the compression method, the capability-stratified evaluation, and the reproduction scripts behind that finding. The headline is not a new retention number: it is that **this eight-task surface average does not reveal the reasoning loss**. Task-dependent collapse after layer removal was reported earlier — [ShortGPT](https://arxiv.org/abs/2403.03853) (§4.6) shows near-zero GSM8K after removing 25% of layers — so this case study's contribution is the cheap-heal controls, the residual-width versus depth comparison, and measurement and calibration checks, scoped to Qwen2.5-1.5B and a ~30M-token heal.

📄 **Full write-up:** [`docs/capability_stratified_compression.md`](docs/capability_stratified_compression.md) · visual version: [`docs/index.html`](docs/index.html) · foundational method + scaling law: [`docs/prior_work_co_compression.md`](docs/prior_work_co_compression.md)

---

## The finding in one table

A compressed model at **17% compression** (Qwen2.5-1.5B, residual width 1280, keep-full block-replacement + 30M-token distillation heal), evaluated task-by-task against its teacher (5-shot GSM8K; `acc`/`acc_norm` per lm-eval defaults):

| task | kind | teacher | compressed | retained |
|---|---|--:|--:|--:|
| sciq | surface | 93.3 | 90.0 | 96% |
| piqa | surface | 76.1 | 70.6 | 93% |
| lambada | surface | 62.0 | 48.7 | 79% |
| **surface/completion mean (8 tasks)** | — | **68.4** | **60.4** | **88%** |
| arc-challenge | reasoning | 45.1 | 34.1 | 76% |
| MMLU | knowledge | 59.6 | 36.0 | 60% |
| **gsm8k** | math | 61.7 | 3.6 | **6%** |

The 88% headline is the surface-mean row. One aggregate "retention %" averages a capability that is intact (surface, ~88%) with one that is destroyed (math, 6%). Deepen to 32% (width 1280 + 6 layers removed) and MMLU also collapses to chance (~23) and GSM8K to ~1 — see the write-up.

## What we found (Qwen2.5-1.5B, this surgery and heal budget)

- **Localization.** GSM8K degrades *continuously* with residual width (roughly halving per ~64 channels removed) and faster with depth removal; MMLU degrades with width and falls to chance at the depth-deletion step tested; surface tasks held up under heavy compression on every tested axis. These are observed patterns for this model, not general laws.
- **Testing explanations.** An *uncut* model put through the identical heal keeps gsm8k 59.3, which argues against heal-induced forgetting alone (it does not isolate every surgery-by-heal interaction). At this budget none of the tested residual reshapes (channel-protection, SliceGPT rotation, low-rank capture), heal objectives (pure-KD, math-data mix, MTP, focus) or cut allocations (wide-shallow / narrow-deep / depth-only) restored baseline gsm8k; the best, pure-KD with a math-data mix, reached 8.5.
- **Measurement checks.** The loss is not explained by the tested answer-extraction and letter-bias effects: gsm8k is identical under strict and flexible extraction (200 problems), and the below-chance MMLU score at 32% is letter collapse ("A" on 98% of items) over chance-level accuracy. Linear probes on frozen features do not recover teacher-like accuracy (no erasure claim).
- **Calibration.** MMLU confidence degrades with compression (AUROC 0.78 → 0.62 → 0.51 for teacher / 17% / 32%); a cascade held within 1 point of the teacher on MMLU must escalate 92–98% of items.
- **Consequence.** In our tests no shape preserved gsm8k beyond small cuts (a ~4% width cut already drops it 61.7 → 42.5). We do not claim a general maximum compression ratio. Whether compressing a larger source model avoids this is an open hypothesis this repo does not resolve.

## Method

`keep-full` block-replacement compression: keep attention/FFN internal dims full, prune residual-stream **width** by activation energy, delete depth cleanly, optional ReplaceMe-style closed-form splice. Then a **cheap** heal — ~30M tokens, `0.5·CE + 0.5·logit-KD(parent)`. Everything lives in [`src/`](src/) (`arch.py` is the operator zoo); [`scripts/run_generative.py`](scripts/run_generative.py) is the entry point with every lever as a flag.

## Research arc

The project began as an encoder-era co-compression study ([`docs/prior_work_co_compression.md`](docs/prior_work_co_compression.md) — 6 GLUE tasks × 2 architectures + GPT-2, developed entirely on an 8 GB RTX 3070). Three pieces of it remain load-bearing here:

- **The mechanism.** *Depth is scarce and concentrated; FFN width is abundant and low-rank* (middle-span Block Influence 2.45× lower than outer layers; FFN 90%-energy in ≤41% of dims). The capability findings deepen this: depth turned out to be scarcest **for reasoning** in these runs — removing 8 layers at full width took gsm8k to zero.
- **The CBA** — a training-free Compression-Budget Allocator that *computes* the depth/width split from one calibration pass and Pareto-beats uniform pruning (95.5% retained at 61.9% reduction, zero search).
- **The methodology.** Every "win" tested at convergence, multi-seed, and under generalization — which is how apparent wins (weight-tied recurrence, fusion modules, staged healing) were caught as artifacts then, and how the ~70-lever negative map (Appendix B) stayed honest now.

Its aggregate retention numbers, however, are exactly the **surface-style metrics this write-up critiques** — read them with that lens.

## Install

```bash
uv sync          # or: pip install -e .
```

Requires a CUDA GPU. The 1.5B experiments were run on a single 24 GB A10; the method was developed on an 8 GB RTX 3070 at smaller scale.

## Reproduce the headline claims

```bash
bash reproduce/01_reasoning_width_curve.sh   # the continuous reasoning collapse vs residual width
bash reproduce/02_capability_crater.sh       # deep compression + dual-basket eval (surface intact, reasoning gone)
bash reproduce/03_causal_controls.sh         # uncut-heal control + a reshape control
```

Each writes an lm-eval line with per-task `gsm8k=… mmlu=… arc_challenge=…`. Expected numbers are recorded in [`results/`](results/) (the exact run summaries these findings are drawn from).

## Evaluation: the dual basket

The core methodological recommendation — **report capability-stratified evals** — is baked into the runner via `--lm_eval_tasks`:

- **surface / completion:** `lambada_openai, hellaswag, arc_easy, piqa, winogrande, boolq, sciq, openbookqa`
- **reasoning / knowledge:** `gsm8k` (5-shot generative), `mmlu`, `arc_challenge`

A structured-compression result reported only on the first basket says nothing about whether the model can still reason.

## Repository layout

```
src/                     the method — arch.py (operator zoo), surgery, distill, eval, data
scripts/run_generative.py entry point (all levers as flags)
scripts/heal_delta.py    per-matrix SVD of the heal delta (shows the heal is a dense polish)
reproduce/               clean scripts for the headline claims
results/                 82 lever summaries (the full negative map), the scaling-law table, and the
                         Sep 2026 measurement checks (calibration, strict/flexible, letter bias, probe)
scripts/calib_eval.py    calibration / per-sample eval; calib_compare.py (cascade); mmlu_probe.py (linear probe)
docs/                    write-up (md) · standalone HTML · prior-work co-compression study
```

The write-up also includes a **source-scale section** (§8: compressibility rises then saturates with model size — on surface metrics only) and **Appendix B: the full ~70-lever negative map** — the catalog of operators, losses, data, and shapes that did *not* beat the simple recipe or recover reasoning.

## Scope & limitations

Single model family (Qwen2.5) at a single scale (1.5B). The heal is fixed at ~30M tokens *by design* (the regime of interest is cheap, few/zero-shot-preserving compression) — we do **not** claim reasoning is unrecoverable with large-scale retraining, only that the tested cheap controls did not recover it. gsm8k is 5-shot generative; the reasoning-collapse conclusion is robust to the sampling limit because the effect (60→0) dwarfs the noise. Cross-family confirmation and a larger-source (3B→1.5B) run are the open follow-ups, and a fuller related-work survey is owed.

## License

MIT — see [LICENSE](LICENSE).
