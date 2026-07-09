#!/usr/bin/env bash
# THE REASONING-WIDTH CURVE — reasoning collapses continuously as residual width is pruned.
# Prune ONLY the residual-stream width (keep all 28 layers, full attention/FFN), sweep it,
# and read the hard basket. Expected (gsm8k, teacher = 61.7):
#   width 1472 -> ~42   width 1408 -> ~23   width 1344 -> ~9   width 1280 -> ~3
# i.e. gsm8k roughly halves for every ~64 channels removed. See results/a10_wthresh_summary.txt.
set -euo pipefail
M="Qwen/Qwen2.5-1.5B"
TASKS="lambada_openai,hellaswag,arc_easy,piqa,winogrande,boolq,sciq,openbookqa,gsm8k,mmlu,arc_challenge"
# ~30M-token cheap heal; --eval_limit 200 keeps the slow generative gsm8k tractable (~20 min/run).
# Remove --eval_limit for the paper's exact numbers (full gsm8k ~2.5 h/run).
HEAL="--remove 0 --full_heal --bf16 --grad_ckpt --heal_steps 4000 --lr 3e-5 --heal_data c4 \
  --heal_teacher $M --heal_tokens 30000000 --batch_size 2 --auto_lr --seqlen 384 \
  --eval_bs 4 --eval_limit 200 --prune_hidden_after --lm_eval_tasks $TASKS"

for W in 1472 1408 1344 1280; do
  echo "### residual width $W ###"
  uv run python scripts/run_generative.py --tag width_$W --model $M --prune_hidden $W $HEAL
done
