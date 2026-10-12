"""Recompute the calibration metrics in write-up §6.5–6.6 from saved per-sample outputs (no GPU, no model).

calib_eval.py stores per-task summaries computed at run time. After the AUROC tie fix (average ranks; review of
PR #1), every summary is recomputed here from the per-sample records with the current calib_eval functions, and
written out with calib_compare.py-compatible structure.

  python scripts/recompute_calib.py results/calib_per_sample/{teacher,cp17,cp29}.json.gz --out_dir /tmp/recalc
  python scripts/calib_compare.py /tmp/recalc/teacher.json /tmp/recalc/cp17.json /tmp/recalc/cp29.json
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from calib_eval import summarize  # noqa: E402


def load(fn):
    with (gzip.open(fn, "rt") if fn.endswith(".gz") else open(fn)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"{'model':<9}{'task':<15}{'n':>6}{'acc':>7}{'ECE':>7}{'AUROC':>8}{'conf_R':>8}{'conf_W':>8}"
          f"  | raw:{'acc':>6}{'ECE':>6}{'AUROC':>7}")
    for fn in a.files:
        d = load(fn)
        by = {}
        for s in d["samples"]:
            by.setdefault(s["task"], []).append(s)
        d["tasks"] = {t: summarize(R) for t, R in by.items()}
        for t, s in d["tasks"].items():
            print(f"{d['name']:<9}{t:<15}{s['n']:>6}{s['acc']:>7}{s['ece']:>7}{s['auroc']:>8}"
                  f"{s['conf_right'] or float('nan'):>8.3f}{s['conf_wrong'] or float('nan'):>8.3f}"
                  f"  |    {s['acc_raw']:>6}{s['ece_raw']:>6}{s['auroc_raw']:>7}")
        json.dump(d, open(os.path.join(a.out_dir, f"{d['name']}.json"), "w"))
    print("RECOMPUTE DONE ->", a.out_dir)


if __name__ == "__main__":
    main()
