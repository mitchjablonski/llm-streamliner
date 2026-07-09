"""Evaluation: task metric, parameter reduction, latency, and reconstruction error.

The story we want to tell per replacement: how much task quality is retained
per parameter / per unit latency saved, and how faithfully the module
reconstructs the removed span's hidden states.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class EvalResult:
    replacement: str
    task: str
    metric_name: str
    metric_value: float = 0.0
    total_params: int = 0
    teacher_params: int = 0
    latency_ms: float = 0.0
    hidden_recon_error: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def param_reduction(self) -> float:
        if not self.teacher_params:
            return 0.0
        return 1.0 - self.total_params / self.teacher_params

    def row(self) -> str:
        return (
            f"| {self.replacement} | {self.metric_name}={self.metric_value:.4f} | "
            f"{self.total_params/1e6:.2f}M | {self.param_reduction*100:.1f}% | "
            f"{self.latency_ms:.2f}ms | "
            f"{'-' if self.hidden_recon_error is None else f'{self.hidden_recon_error:.4f}'} |"
        )


def table_header() -> str:
    return (
        "| replacement | metric | params | reduction | latency | recon err |\n"
        "|---|---|---|---|---|---|"
    )


@torch.no_grad()
def evaluate_task(model, dataloader, device, metric_name="accuracy", max_batches=None) -> float:
    """Compute a GLUE-style metric over a dataloader.

    Supports ``accuracy``, ``f1``, and ``matthews_correlation`` without an extra
    dependency (falls back to sklearn for f1/mcc).
    """
    model.eval()
    preds, labels = [], []
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        inputs = {k: batch[k].to(device)
                  for k in ("input_ids", "attention_mask", "token_type_ids") if k in batch}
        logits = model(**inputs).logits
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(batch["label"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    if metric_name == "accuracy":
        return float((preds == labels).mean())
    if metric_name == "f1":
        from sklearn.metrics import f1_score

        return float(f1_score(labels, preds))
    if metric_name == "matthews_correlation":
        from sklearn.metrics import matthews_corrcoef

        return float(matthews_corrcoef(labels, preds))
    raise ValueError(f"Unknown metric '{metric_name}'.")


@torch.no_grad()
def measure_latency(model, sample_batch, device, n_warmup=5, n_iter=20) -> float:
    """Mean forward latency in ms over a fixed batch."""
    model.eval()
    batch = {k: v.to(device) for k, v in sample_batch.items() if torch.is_tensor(v)}
    for _ in range(n_warmup):
        model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000.0
