"""Dataset loading for quick iteration on an 8GB GPU.

Defaults to GLUE single-/pair-sentence classification subsets (SST-2, MRPC,
CoLA) which train in minutes on small encoders.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoTokenizer

# task -> (text_field_a, text_field_b_or_None, num_labels, metric)
GLUE_TASKS = {
    "sst2": ("sentence", None, 2, "accuracy"),
    "mrpc": ("sentence1", "sentence2", 2, "f1"),
    "cola": ("sentence", None, 2, "matthews_correlation"),
    # full-GLUE classification tasks (regression STS-B omitted: different head/loss)
    "qnli": ("question", "sentence", 2, "accuracy"),
    "rte": ("sentence1", "sentence2", 2, "accuracy"),
    "qqp": ("question1", "question2", 2, "f1"),
    # MNLI omitted: the available fine-tuned checkpoint uses a permuted label order
    # vs the GLUE dataset (teacher acc ~0.05), which would invalidate retained_pct.
}


@dataclass
class DataConfig:
    task: str = "sst2"
    model_name: str = "distilbert-base-uncased"
    max_seq_len: int = 128
    batch_size: int = 32
    pad_to_max: bool = False  # set True for the `mixer` replacement (fixed seq len)


def load_glue(cfg: DataConfig):
    """Return ``(tokenized_datasets, tokenizer, num_labels, metric_name)``."""
    if cfg.task not in GLUE_TASKS:
        raise KeyError(f"Unknown task '{cfg.task}'. Options: {sorted(GLUE_TASKS)}")
    field_a, field_b, num_labels, metric = GLUE_TASKS[cfg.task]

    ds = load_dataset("nyu-mll/glue", cfg.task)  # canonical repo id ("glue" alias is deprecated)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    padding = "max_length" if cfg.pad_to_max else False

    def encode(batch):
        args = (batch[field_a],) if field_b is None else (batch[field_a], batch[field_b])
        return tok(*args, truncation=True, max_length=cfg.max_seq_len, padding=padding)

    ds = ds.map(encode, batched=True)
    keep = ["input_ids", "attention_mask", "token_type_ids", "label"]
    keep = [c for c in keep if c in ds["train"].column_names]
    ds.set_format("torch", columns=keep)
    return ds, tok, num_labels, metric


def collate_pad(tokenizer):
    """Collate fn that pads a batch to its longest sequence (dynamic padding).

    Needed because ``pad_to_max=False`` leaves examples at their true lengths;
    the default collate cannot stack variable-length tensors.
    """
    import torch

    pad_id = tokenizer.pad_token_id or 0

    def _collate(features):
        maxlen = max(f["input_ids"].size(0) for f in features)
        out = {}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key not in features[0]:
                continue
            fill = pad_id if key == "input_ids" else 0
            rows = []
            for f in features:
                t = f[key]
                pad = maxlen - t.size(0)
                rows.append(torch.cat([t, torch.full((pad,), fill, dtype=t.dtype)]) if pad else t)
            out[key] = torch.stack(rows)
        out["label"] = torch.stack([f["label"] for f in features])
        return out

    return _collate
