"""Run reporting + logging harness (the experiment's ground-truth output).

This module is the analog of autoresearch's read-only ``prepare.py``: it owns
how an experiment's results are measured-into-numbers, printed, and logged.
**Do not edit it to make a run look better** — it is the metric of record.

It provides:

* :func:`reset_peak_vram` / :func:`peak_vram_mb` — GPU memory accounting.
* :func:`git_commit` — the short commit the run was executed at.
* :func:`retained` — the headline objective: spliced metric / teacher metric.
* :func:`emit` — print a stable, greppable ``=== METRICS ===`` block, persist a
  per-run ``metrics.json``, and append a row to the machine ledger
  (``experiments/runs.jsonl``). The agent reads the printed block; the JSON
  files guarantee nothing is lost or mis-transcribed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Keys printed in the `=== METRICS ===` block, in order. Anything else in the
# results dict is still persisted to JSON but not printed.
_BLOCK_KEYS = [
    "task",
    "model",
    "replacement",
    "selection",
    "teacher_metric",
    "spliced_pre",
    "spliced_post",
    "retained_pct",
    "total_params_M",
    "param_reduction_pct",
    "removed_blocks",
    "stage1_init_loss",
    "stage1_final_loss",
    "latency_ms",
    "peak_vram_mb",
    "train_seconds",
    "total_seconds",
]


def reset_peak_vram(device) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_mb(device) -> float:
    if getattr(device, "type", None) == "cuda":
        import torch

        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 -- no repo / no commits yet
        return "nogit"


def retained(spliced_metric: float, teacher_metric: float) -> float:
    """Headline objective: fraction of the full model's quality retained.

    1.0 == the compressed model matches the full teacher. Higher is better.
    For metrics that can be <= 0 (e.g. CoLA's Matthews corr) this ratio is only
    meaningful when the teacher score is comfortably positive; report the raw
    ``spliced_post`` alongside it in those cases.
    """
    if teacher_metric == 0:
        return 0.0
    return spliced_metric / teacher_metric


def emit(metrics: dict, *, output_dir: str = "experiments", run_tag: str = "run") -> dict:
    """Print the greppable block and persist the run. Returns ``metrics`` enriched."""
    metrics = dict(metrics)
    metrics.setdefault("commit", git_commit())
    metrics.setdefault("run_tag", run_tag)

    print("\n=== METRICS ===")
    width = max(len(k) for k in _BLOCK_KEYS)
    for k in _BLOCK_KEYS:
        if k not in metrics:
            continue
        v = metrics[k]
        v = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{k + ':':<{width + 1}} {v}")

    run_dir = Path(output_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    ledger = Path(output_dir) / "runs.jsonl"
    with ledger.open("a") as f:
        f.write(json.dumps(metrics, sort_keys=True) + "\n")

    return metrics
