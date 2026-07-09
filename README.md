# LLM-Streamliner — Capability-Stratified Compression

**Cheap structured compression of a 1.5B language model keeps ~88% of average benchmark performance at 17% compression — and silently destroys the model's ability to reason.**

This repository contains the compression method, the capability-stratified evaluation, and the reproduction scripts behind that finding. The headline result is not a new state-of-the-art retention number; it is that **aggregate retention numbers are surface-task numbers that hide a reasoning crater**, and a mechanism-level account of where and why reasoning breaks under compression.

📄 **Full write-up:** [`docs/capability_stratified_compression.md`](docs/capability_stratified_compression.md) · visual version: [`docs/index.html`](docs/index.html)

---

## The finding in one table

A strong compressed model (Qwen2.5-1.5B, 32% compression, block-replacement + 30M-token distillation heal), evaluated task-by-task against its teacher:

| task | kind | teacher | compressed | retained |
|---|---|--:|--:|--:|
| piqa | surface | 76.1 | ~68 | ~89% |
| lambada | surface | 62.0 | ~38 | ~61% |
| arc-challenge | reasoning-lite | 45.1 | ~28 | ~62% |
| **MMLU** | knowledge | 59.6 | ~23 | **chance** |
| **gsm8k** | math | 61.7 | ~1 | **~2%** |

One aggregate "retention %" averages a capability that is intact with one that is destroyed.

## What we establish

- **Localization.** Reasoning degrades *continuously* with residual width (gsm8k retention roughly halves per ~64 channels removed) and is destroyed *even faster* by depth removal. Reasoning needs the near-full network — width **and** depth; knowledge needs depth; surface tasks tolerate heavy compression on any axis.
- **Causal isolation.** Four controls show the **surgery** destroys reasoning, not the heal: an *uncut* model put through the identical heal keeps gsm8k 59.3; and **no** residual reshape (channel-protection, SliceGPT rotation, low-rank capture), **no** heal objective (pure-KD, math-data, MTP, focus), and **no** allocation of the cut (wide-shallow / narrow-deep / depth-only) recovers it.
- **Consequence.** Reasoning-preserving *cheap* compression tops out at ~4–8% at 1.5B, by any shape. A 16-lever operator search and a saturated distillation sweep confirm there is no cheap exception. The remaining lever is **source scale** (compressing a larger model), which this repo does not resolve.

## Method

`keep-full` block-replacement compression: keep attention/FFN internal dims full, prune residual-stream **width** by activation energy, delete depth cleanly, optional ReplaceMe-style closed-form splice. Then a **cheap** heal — ~30M tokens, `0.5·CE + 0.5·logit-KD(parent)`. Everything lives in [`src/`](src/) (`arch.py` is the operator zoo); [`scripts/run_generative.py`](scripts/run_generative.py) is the entry point with every lever as a flag.

## Install

```bash
uv sync          # or: pip install -e .
```

Requires a CUDA GPU. The 1.5B experiments were run on a single 24 GB A10; the method was developed on an 8 GB RTX 3070 at smaller scale.

## Reproduce the headline claims

```bash
bash reproduce/01_reasoning_width_curve.sh   # the continuous reasoning collapse vs residual width
bash reproduce/02_capability_crater.sh       # deep compression + dual-basket eval (surface intact, reasoning gone)
bash reproduce/03_causal_controls.sh         # heal-is-innocent + a reshape control (cut destroys, not heal)
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
results/                 run summaries the findings are drawn from
docs/                    full write-up (md + standalone HTML)
```

## Scope & limitations

Single model family (Qwen2.5) at a single scale (1.5B). The heal is fixed at ~30M tokens *by design* (the regime of interest is cheap, few/zero-shot-preserving compression) — we do **not** claim reasoning is unrecoverable with large-scale retraining, only that it is unrecoverable *cheaply*. gsm8k is 5-shot generative; the reasoning-collapse conclusion is robust to the sampling limit because the effect (60→0) dwarfs the noise. Cross-family confirmation and a larger-source (3B→1.5B) run are the open follow-ups.

## License

MIT — see [LICENSE](LICENSE).
