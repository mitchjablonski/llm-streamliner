#!/usr/bin/env bash
# THE CAPABILITY CRATER — one deep-compression run, evaluated on BOTH baskets, shows the split:
# surface tasks survive while reasoning/knowledge are destroyed. This is the whole point —
# an aggregate "retention %" averages an intact capability with a destroyed one.
# ~32% compression (residual width 1280 + 6 layers removed). Expected: piqa ~89% retained,
# gsm8k ~1 (teacher 61.7), MMLU ~23 (chance). See results/a10_hardloc_summary.txt / a10_recover_summary.txt.
set -euo pipefail
M="Qwen/Qwen2.5-1.5B"
TASKS="lambada_openai,hellaswag,arc_easy,piqa,winogrande,boolq,sciq,openbookqa,gsm8k,mmlu,arc_challenge"

uv run python scripts/run_generative.py --tag crater \
  --model $M --prune_hidden 1280 --remove 6 \
  --full_heal --bf16 --grad_ckpt --heal_steps 4000 --lr 3e-5 --heal_data c4 \
  --heal_teacher $M --heal_tokens 30000000 --batch_size 2 --auto_lr --seqlen 384 \
  --eval_bs 4 --eval_limit 200 --prune_hidden_after --lm_eval_tasks $TASKS
