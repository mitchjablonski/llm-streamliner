"""Calibration eval: does a (compressed) model still KNOW when it's wrong?

Runs lm-eval with per-sample logging and turns each multiple-choice answer into a probability
distribution over the options (softmax of the option log-likelihoods). Per task it reports accuracy,
ECE, Brier, AUROC(confidence -> correct) and mean confidence on right vs wrong answers, and writes
per-sample records so calib_compare.py can build parent-escalation cascade curves.

Generative tasks (gsm8k) have no option distribution; for them we record exact-match under BOTH
lm-eval filters (strict-match AND flexible-extract) - the crater was only ever scored strict.

  uv run python scripts/calib_eval.py --model Qwen/Qwen2.5-1.5B --name teacher \
      --tasks mmlu,arc_challenge,arc_easy,boolq,piqa --gen_tasks gsm8k --gen_limit 200
  uv run python scripts/calib_eval.py --model ~/ckpt_cp29 --name cp29 ...
"""
import argparse
import json
import os
import time

import numpy as np
import torch

N_BINS = 15


def _ll(resp):
    """One option's log-likelihood from a filtered_resps entry ((ll, is_greedy) or [ll, is_greedy])."""
    while isinstance(resp, (list, tuple)) and resp and isinstance(resp[0], (list, tuple)):
        resp = resp[0]
    return float(resp[0]) if isinstance(resp, (list, tuple)) else float(resp)


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def ece(conf, correct, n_bins=N_BINS):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return tot


def auroc(conf, correct):
    """P(conf of a random right answer > conf of a random wrong answer); 0.5 = confidence is useless."""
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=bool)
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]).astype(float)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):                       # AVERAGE ranks for ties (Mann-Whitney U); distinct ranks for tied
        j = i                                # scores biased AUROC (Sol review of PR #1: [.5]*4 gave 0.0, not 0.5)
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def group_of(task):
    return "mmlu" if task.startswith("mmlu") else task


def mc_records(task, samples):
    """Per-sample records for a loglikelihood multiple-choice task.

    Primary scoring matches the accuracy lm-eval reports: acc_norm (log-likelihood / choice char-length,
    lm-eval's definition) when the task has it, matching run_generative's _pick preference; else raw acc.
    A softmax over length-normalized scores is only a heuristic probability, so the RAW-likelihood
    softmax (a real distribution over the options) is always recorded too (conf_raw / correct_raw)."""
    recs, mismatch = [], 0
    for s in samples:
        lls = np.array([_ll(r) for r in s["filtered_resps"]])
        conts = [a[1] if isinstance(a, (list, tuple)) and len(a) > 1 else "" for a in s["arguments"]]
        # lm-eval normalizes by len(choice) WITHOUT the target delimiter the continuation carries
        clen = np.array([max(1, len(c[1:] if c.startswith(" ") else c)) for c in conts], dtype=float)
        use_norm = "acc_norm" in s
        p_raw = _softmax(lls)
        probs = _softmax(lls / clen) if use_norm else p_raw
        target = s["target"]
        if not isinstance(target, (int, np.integer)):          # some tasks give the answer string
            choices = s["doc"].get("choices") if isinstance(s["doc"], dict) else None
            target = choices.index(target) if choices and target in choices else int(target)
        target = int(target)
        correct = int(int(np.argmax(probs)) == target)
        ref = s.get("acc_norm" if use_norm else "acc")
        if ref is not None and int(round(ref)) != correct:
            mismatch += 1
        recs.append({"task": group_of(task), "sub": task, "doc_id": s["doc_id"], "target": target,
                     "pred": int(np.argmax(probs)), "correct": correct, "conf": float(probs.max()),
                     "probs": [round(float(p), 5) for p in probs],
                     "correct_raw": int(int(np.argmax(p_raw)) == target), "conf_raw": float(p_raw.max()),
                     "scoring": "acc_norm" if use_norm else "acc"})
    return recs, mismatch


def summarize(recs):
    conf = np.array([r["conf"] for r in recs])
    cor = np.array([r["correct"] for r in recs])
    n_opt = np.array([len(r["probs"]) for r in recs])
    tgt_p = np.array([r["probs"][r["target"]] for r in recs])
    return {"n": len(recs), "acc": round(100 * cor.mean(), 2),
            "chance": round(100 * float(np.mean(1.0 / n_opt)), 2),
            "ece": round(100 * ece(conf, cor), 2),
            "brier": round(float(np.mean([(1 - p) ** 2 + sum(q ** 2 for i, q in enumerate(r["probs"]) if i != r["target"])
                                          for p, r in zip(tgt_p, recs)])), 4),
            "auroc": round(auroc(conf, cor), 4),
            "conf_right": round(float(conf[cor == 1].mean()), 4) if cor.any() else None,
            "conf_wrong": round(float(conf[cor == 0].mean()), 4) if (cor == 0).any() else None,
            "mean_conf": round(float(conf.mean()), 4),
            "scoring": recs[0]["scoring"],
            # raw-likelihood scoring (a true distribution); identical to the above for acc-scored tasks
            "acc_raw": round(100 * float(np.mean([r["correct_raw"] for r in recs])), 2),
            "ece_raw": round(100 * ece([r["conf_raw"] for r in recs], [r["correct_raw"] for r in recs]), 2),
            "auroc_raw": round(auroc([r["conf_raw"] for r in recs], [r["correct_raw"] for r in recs]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir (e.g. ~/ckpt_cp29)")
    ap.add_argument("--name", required=True, help="label; output -> <out_dir>/<name>.json")
    ap.add_argument("--tasks", default="mmlu,arc_challenge,arc_easy,boolq,piqa",
                    help="loglikelihood multiple-choice tasks (calibration measured on these)")
    ap.add_argument("--gen_tasks", default="gsm8k", help="generative tasks: strict + flexible exact-match only")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per MC task (smoke tests)")
    ap.add_argument("--gen_limit", type=int, default=200, help="cap examples per generative task (slow)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out_dir", default="results/calib")
    args = ap.parse_args()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = os.path.expanduser(args.model)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype)).to("cuda").eval()
    model.config.use_cache = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[calib] {args.name}: {path} | {n_params / 1e6:.1f}M params | "
          f"hidden={model.config.hidden_size} layers={model.config.num_hidden_layers}")
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.bs)

    out = {"name": args.name, "model": path, "params_M": round(n_params / 1e6, 1),
           "hidden": model.config.hidden_size, "layers": model.config.num_hidden_layers,
           "tasks": {}, "gen": {}, "samples": []}

    mc = [t for t in args.tasks.split(",") if t]
    if mc:
        res = simple_evaluate(model=lm, tasks=mc, log_samples=True,
                              **({"limit": args.limit} if args.limit else {}))
        by_group, mism = {}, 0
        for task, samples in res["samples"].items():
            recs, m = mc_records(task, samples)
            mism += m
            for r in recs:
                by_group.setdefault(r["task"], []).append(r)
        if mism:
            print(f"[calib] WARN {mism} samples where our argmax disagrees with lm-eval's own acc")
        out["argmax_mismatch"] = mism
        for g, recs in by_group.items():
            out["tasks"][g] = summarize(recs)
            out["samples"].extend(recs)

    gen = [t for t in args.gen_tasks.split(",") if t]
    if gen:
        res = simple_evaluate(model=lm, tasks=gen, log_samples=True,
                              **({"limit": args.gen_limit} if args.gen_limit else {}))
        for task in gen:
            r = res["results"].get(task, {})
            out["gen"][task] = {k.replace("exact_match,", ""): round(100 * v, 2) for k, v in r.items()
                                if k.startswith("exact_match,") and "stderr" not in k and isinstance(v, (int, float))}
            ex = [s for s in res["samples"].get(task, []) if s.get("filter") == "strict-match"][:3]
            out["gen"][task]["examples"] = [{"target": s["target"], "resp": str(s["resps"][0][0])[:600]} for s in ex]

    out["seconds"] = round(time.time() - t0)
    os.makedirs(args.out_dir, exist_ok=True)
    fn = os.path.join(args.out_dir, f"{args.name}.json")
    json.dump(out, open(fn, "w"))
    print(f"\n[calib] {args.name} ({out['params_M']}M) -> {fn}  [{out['seconds']}s]")
    print(f"{'task':<14}{'n':>6}{'acc':>7}{'chance':>7}{'ECE':>6}{'AUROC':>7}{'conf_R':>8}{'conf_W':>8}"
          f"  | raw: {'acc':>6}{'ECE':>6}{'AUROC':>7}")
    for g, s in out["tasks"].items():
        print(f"{g:<14}{s['n']:>6}{s['acc']:>7}{s['chance']:>7}{s['ece']:>6}{s['auroc']:>7}"
              f"{s['conf_right'] or float('nan'):>8.3f}{s['conf_wrong'] or float('nan'):>8.3f}"
              f"  |      {s['acc_raw']:>6}{s['ece_raw']:>6}{s['auroc_raw']:>7}")
    for t, g in out["gen"].items():
        print(f"{t:<14} strict={g.get('strict-match')} flexible={g.get('flexible-extract')}")
    print("CALIB DONE")


if __name__ == "__main__":
    main()
