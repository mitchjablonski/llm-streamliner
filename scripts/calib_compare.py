"""Compare calib_eval.py outputs: teacher vs compressed students, plus the parent-escalation cascade.

Cascade: the student answers the questions it is most confident on (coverage c) and hands the rest
to the teacher. If the student's confidence is informative, combined accuracy beats RANDOM escalation
(c*acc_student + (1-c)*acc_teacher) - that gap is what calibration is worth as a router.

`oracle esc@-1pt` picks the escalation cutoff on the SAME labels it is scored on (an upper bound).
The honest number is the HOLDOUT cascade: a confidence threshold is chosen on a fixed half of the
questions (split by hash of the question key) and applied, unchanged, to the other half.

  uv run python scripts/calib_compare.py results/calib/teacher.json results/calib/cp17.json results/calib/cp29.json
"""
import hashlib
import json
import sys

import numpy as np

COVS = (1.0, 0.9, 0.8, 0.7, 0.5)


def load(fn):
    d = json.load(open(fn))
    d["by_key"] = {(s["task"], s["sub"], s["doc_id"]): s for s in d["samples"]}
    return d


def _half(k):
    """Deterministic calibration/holdout split: 0 = calibration half, 1 = holdout half."""
    return int(hashlib.md5(repr(k).encode()).hexdigest(), 16) % 2


def holdout(stu, tea, task, sfx="", tol=0.01, risk=False):
    """Pick the lowest student-confidence threshold whose cascade stays within tol of the teacher on the
    calibration half; apply it unchanged to the holdout half. Returns (esc share, combined acc,
    teacher acc, student acc) on the holdout half.

    risk=True: require the one-sided 95% UPPER bound of the accuracy drop (teacher - cascade, paired
    per question on the calibration half) to be <= tol, instead of the point estimate - so the target
    holds on new data with high probability rather than about half the time."""
    keys = [k for k in stu["by_key"] if k[0] == task and k in tea["by_key"]]
    cal = [k for k in keys if _half(k) == 0]
    test = [k for k in keys if _half(k) == 1]
    if len(cal) < 20 or len(test) < 20:
        return None
    def arr(ks):
        return (np.array([stu["by_key"][k]["conf" + sfx] for k in ks]),
                np.array([stu["by_key"][k]["correct" + sfx] for k in ks], dtype=float),
                np.array([tea["by_key"][k]["correct" + sfx] for k in ks], dtype=float))
    c_conf, c_s, c_t = arr(cal)
    tau = np.inf                                  # default: escalate everything
    for th in np.unique(c_conf):                  # ascending: first passing threshold keeps the most
        drop = c_t - np.where(c_conf >= th, c_s, c_t)                    # per-question paired drop
        bound = drop.mean() + (1.645 * drop.std(ddof=1) / np.sqrt(len(drop)) if risk else 0.0)
        if bound <= tol:
            tau = th
            break
    h_conf, h_s, h_t = arr(test)
    keep = h_conf >= tau
    return 1 - keep.mean(), 100 * np.where(keep, h_s, h_t).mean(), 100 * h_t.mean(), 100 * h_s.mean()


def position_bias(run, task="mmlu"):
    """Share of predictions per option index vs share of gold answers (answer-letter bias check)."""
    recs = [s for s in run["samples"] if s["task"] == task]
    if not recs:
        return None
    n_opt = max(len(r["probs"]) for r in recs)
    pred = np.bincount([r["pred"] for r in recs], minlength=n_opt) / len(recs)
    gold = np.bincount([r["target"] for r in recs], minlength=n_opt) / len(recs)
    return pred, gold


def gold_nll(run, task):
    """Mean negative log-likelihood of the gold option under the primary-scoring distribution."""
    recs = [s for s in run["samples"] if s["task"] == task]
    return float(np.mean([-np.log(max(r["probs"][r["target"]], 1e-6)) for r in recs])) if recs else None


def cascade(stu, tea, task, sfx=""):
    """sfx="" -> primary scoring (matches reported acc); sfx="_raw" -> raw-likelihood softmax."""
    keys = [k for k in stu["by_key"] if k[0] == task and k in tea["by_key"]]
    if not keys:
        return None
    sc = np.array([stu["by_key"][k]["conf" + sfx] for k in keys])
    s_ok = np.array([stu["by_key"][k]["correct" + sfx] for k in keys], dtype=float)
    t_ok = np.array([tea["by_key"][k]["correct" + sfx] for k in keys], dtype=float)
    order = np.argsort(-sc)
    rows = []
    for c in COVS:
        n = int(round(c * len(keys)))
        keep = np.zeros(len(keys), bool)
        keep[order[:n]] = True
        comb = np.where(keep, s_ok, t_ok).mean()
        rand = c * s_ok.mean() + (1 - c) * t_ok.mean()
        rows.append((c, 100 * comb, 100 * (comb - rand)))
    # smallest escalation share that gets within 1pt of the teacher
    s_cum = np.concatenate([[0], np.cumsum(s_ok[order])])            # student-correct among top-n
    t_rest = t_ok.sum() - np.concatenate([[0], np.cumsum(t_ok[order])])  # teacher-correct on the rest
    comb_n = (s_cum + t_rest) / len(keys)                                # combined acc when student keeps n
    ok = np.nonzero(comb_n >= t_ok.mean() - 0.01)[0]
    need = 1 - ok.max() / len(keys) if len(ok) else None
    return rows, need, 100 * t_ok.mean()


def main(files):
    runs = [load(f) for f in files]
    tea, studs = runs[0], runs[1:]
    lines = [f"# Calibration under compression\n\nteacher = `{tea['name']}` ({tea['params_M']}M)\n"]
    lines.append("## Per-task calibration\n")
    lines.append("| model | params | task | scoring | n | acc | chance | ECE | Brier | gold NLL | AUROC "
                 "| conf right | conf wrong | raw acc | raw ECE | raw AUROC |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        for t, s in r["tasks"].items():
            lines.append(f"| {r['name']} | {r['params_M']}M | {t} | {s['scoring']} | {s['n']} | {s['acc']} | "
                         f"{s['chance']} | {s['ece']} | {s['brier']} | {gold_nll(r, t):.3f} | {s['auroc']} | "
                         f"{s['conf_right']} | {s['conf_wrong']} | {s['acc_raw']} | {s['ece_raw']} | {s['auroc_raw']} |")
    lines.append("\n## MMLU answer-position bias (prediction share per option vs gold share)\n")
    lines.append("| model | pred A/B/C/D | gold A/B/C/D | max pred share |")
    lines.append("|---|---|---|---|")
    for r in runs:
        pb = position_bias(r)
        if pb:
            pred, gold = pb
            lines.append(f"| {r['name']} | {' / '.join(f'{x:.2f}' for x in pred)} | "
                         f"{' / '.join(f'{x:.2f}' for x in gold)} | {pred.max():.2f} |")
    lines.append("\n## gsm8k: strict vs flexible exact-match\n")
    lines.append("| model | strict | flexible |")
    lines.append("|---|---|---|")
    for r in runs:
        g = r.get("gen", {}).get("gsm8k")
        if g:
            lines.append(f"| {r['name']} | {g.get('strict-match')} | {g.get('flexible-extract')} |")
    for sfx, title in (("", "primary scoring (matches reported acc)"),
                       ("_raw", "raw-likelihood scoring (true option distribution)")):
        lines.append(f"\n## Cascade, {title}: student answers its most-confident share, teacher the rest\n")
        lines.append("Cells: combined acc (gain over random escalation). `oracle esc@-1pt` = share escalated to "
                     "get within 1pt of the teacher, cutoff chosen on the scored labels (upper bound).\n")
        lines.append("| student | task | teacher acc | " + " | ".join(f"cov {c:.0%}" for c in COVS) + " | oracle esc@-1pt |")
        lines.append("|---|---|---|" + "---|" * len(COVS) + "---|")
        for st in studs:
            for t in st["tasks"]:
                res = cascade(st, tea, t, sfx)
                if res is None:
                    continue
                rows, need, tacc = res
                cells = " | ".join(f"{a:.1f} ({g:+.1f})" for _, a, g in rows)
                lines.append(f"| {st['name']} | {t} | {tacc:.1f} | {cells} | "
                             f"{'n/a' if need is None else f'{need:.0%}'} |")
        lines.append(f"\n**Holdout cascade ({title})**: threshold chosen on the calibration half to stay within 1pt "
                     "of the teacher, applied unchanged to the holdout half.\n")
        lines.append("| student | task | holdout n | escalated | combined acc | teacher acc | student-alone acc |")
        lines.append("|---|---|---|---|---|---|---|")
        for st in studs:
            for t in st["tasks"]:
                h = holdout(st, tea, t, sfx)
                if h is None:
                    continue
                esc, comb, tacc, sacc = h
                n_h = sum(1 for k in st["by_key"] if k[0] == t and k in tea["by_key"] and _half(k) == 1)
                lines.append(f"| {st['name']} | {t} | {n_h} | {esc:.0%} | {comb:.1f} | {tacc:.1f} | {sacc:.1f} |")
    txt = "\n".join(lines)
    print(txt)
    return txt


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: calib_compare.py TEACHER.json STUDENT.json [STUDENT.json ...]")
    main(sys.argv[1:])
