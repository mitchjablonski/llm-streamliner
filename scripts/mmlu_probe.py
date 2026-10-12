"""Frozen-feature linear probe on MMLU: is the answer still linearly decodable from a model's hidden
states, even when its option likelihoods are at chance (cp29: 98% "A", debiased 25.1%)?

For each model: format every MMLU test question as lm-eval does ("...\nA. ..\nB. ..\nAnswer:"), take
the last-token hidden state at ~50/75/100% depth, and fit a 4-way logistic regression with 5-fold CV
(folds grouped by subject so a probe can't memorize subject-specific answer-letter skew).
A random-init model of the same shape is the control for surface cues in the text.

A NEGATIVE probe does not prove the knowledge is erased (a nonlinear readout might still find it);
a POSITIVE probe well above the control shows the knowledge is present but inaccessible to scoring.

  uv run python scripts/mmlu_probe.py --models native05=Qwen/Qwen2.5-0.5B cp29=~/ckpt_cp29 \
      --control Qwen/Qwen2.5-0.5B
"""
import argparse
import json
import os
import time

import numpy as np
import torch


def fmt(x):
    ch = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", x["choices"]))
    subj = x["subject"].replace("_", " ")
    return (f"The following are multiple choice questions (with answers) about {subj}.\n\n"
            f"{x['question'].strip()}\n{ch}\nAnswer:")


@torch.no_grad()
def features(model, tok, texts, layers, bs=16, max_len=768):
    feats = {l: [] for l in layers}
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(last.size(0), device=last.device)
        for l in layers:
            feats[l].append(hs[l][rows, last].float().cpu())
    return {l: torch.cat(v).numpy() for l, v in feats.items()}


def probe(X, y, groups, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, random_state=seed))
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return 100 * float(np.mean(accs)), 100 * float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--control", default=None, help="config source for a random-init control model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/probe/mmlu_probe.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset("cais/mmlu", "all", split="test")
    if args.limit:
        ds = ds.shuffle(seed=0).select(range(args.limit))
    texts = [fmt(x) for x in ds]
    y = np.array(ds["answer"])
    subjects = sorted(set(ds["subject"]))
    groups = np.array([subjects.index(s) for s in ds["subject"]])
    print(f"[probe] MMLU test n={len(y)} subjects={len(subjects)} gold dist={np.bincount(y, minlength=4) / len(y)}")

    specs = [tuple(m.split("=", 1)) for m in args.models]
    if args.control:
        specs.append(("random_init", args.control))
    out = {"n": len(y), "results": {}}
    for name, path in specs:
        t0 = time.time()
        path = os.path.expanduser(path)
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side, tok.truncation_side = "right", "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if name == "random_init":
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        model = model.cuda().eval()
        L = model.config.num_hidden_layers
        layers = sorted({L // 2, (3 * L) // 4, L})                      # hidden_states[L] = final (post-norm)
        F = features(model, tok, texts, layers)
        del model; torch.cuda.empty_cache()
        res = {}
        for l in layers:
            m, sd = probe(F[l], y, groups)
            res[f"layer{l}/{L}"] = {"acc": round(m, 2), "fold_sd": round(sd, 2)}
        out["results"][name] = res
        print(f"[probe] {name}: " + " | ".join(f"{k} {v['acc']}±{v['fold_sd']}" for k, v in res.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("PROBE DONE")


if __name__ == "__main__":
    main()
