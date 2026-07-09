#!/usr/bin/env bash
# CAUSAL CONTROLS — is reasoning killed by the CUT or by the HEAL?
#
# (A) forget control: an UNCUT model put through the identical 30M-token heal.
#     If reasoning survives here, the heal is innocent and the surgery is the cause.
#     Expected: gsm8k ~59 (teacher 61.7). See results/a10_forget_summary.txt.
#
# (B) reshape control: at the same 17% width compression, protect the residual channels
#     most important to math (capability-aware selection) instead of plain energy ranking.
#     If reasoning is recoverable by a better basis, this lifts gsm8k. It does not
#     (~3, same as raw drop) — reasoning loss is dimensionality, not basis.
#     See results/a10_reshape_summary.txt.
set -euo pipefail
M="Qwen/Qwen2.5-1.5B"
TASKS="lambada_openai,hellaswag,arc_easy,piqa,winogrande,boolq,sciq,openbookqa,gsm8k,mmlu,arc_challenge"
HEAL="--full_heal --bf16 --grad_ckpt --heal_steps 4000 --lr 3e-5 --heal_data c4 \
  --heal_teacher $M --heal_tokens 30000000 --batch_size 2 --auto_lr --seqlen 384 \
  --eval_bs 4 --eval_limit 200 --prune_hidden_after --lm_eval_tasks $TASKS"

echo "### (A) forget control — uncut model, identical heal (expect gsm8k ~59) ###"
uv run python scripts/run_generative.py --tag forget_uncut --model $M --remove 0 $HEAL

echo "### (B) reshape control — capability-aware channel protection @17% (expect gsm8k ~3) ###"
uv run python scripts/run_generative.py --tag reshape_protect --model $M \
  --prune_hidden 1280 --remove 0 --protect_math 96 $HEAL
