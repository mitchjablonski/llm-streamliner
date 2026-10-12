"""Answer-letter debias check for MMLU (write-up §6.5, "Objection 2").

From calib_eval.py per-sample outputs: split items into two halves by a hash of (task, subtask, doc_id); fit each
model's mean predicted probability per answer position (A/B/C/D) on the first half; divide it out of the second
half's option distributions, renormalize, and report holdout accuracy before/after plus AUROC of the debiased
max-probability. Also reports the share of predictions per letter (answer-letter collapse check).

  python scripts/letter_debias.py results/calib/teacher.json results/calib/cp17.json results/calib/cp29.json
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from calib_eval import auroc, ece  # noqa: E402


def half(key):
    return int(hashlib.md5(repr(key).encode()).hexdigest(), 16) % 2


def main(files):
    print("model     holdout acc -> letter-debiased acc | debiased AUROC  ECE | pred share A/B/C/D (all items)")
    for fn in files:
        d = json.load(open(fn))
        R = [s for s in d["samples"] if s["task"] == "mmlu"]
        P = np.array([s["probs"] for s in R])
        y = np.array([s["target"] for s in R])
        k = np.array([half((s["task"], s["sub"], s["doc_id"])) for s in R])
        prior = P[k == 0].mean(0)
        Q = P / prior
        Q /= Q.sum(1, keepdims=True)
        h = k == 1
        acc = 100 * (P[h].argmax(1) == y[h]).mean()
        dacc = 100 * (Q[h].argmax(1) == y[h]).mean()
        conf, cor = Q[h].max(1), (Q[h].argmax(1) == y[h]).astype(int)
        share = np.bincount(P.argmax(1), minlength=4) / len(P)
        print(f"{d['name']:<9} {acc:5.1f} -> {dacc:5.1f}               | {auroc(conf, cor):.3f}        "
              f"{100 * ece(conf, cor):4.1f} | {' / '.join(f'{x:.2f}' for x in share)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
