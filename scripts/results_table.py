"""Consolidate every logged trial (experiments/runs.jsonl) into one clean sorted table —
the single reference for how the levers (select/method/heal/reduction) interact.

    uv run python scripts/results_table.py            # markdown to stdout
    uv run python scripts/results_table.py --tsv out.tsv
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="experiments/runs.jsonl")
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()
    rows = []
    for line in Path(args.jsonl).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        down = ""
        if "compressed/avg" in d and "teacher/avg" in d and d["teacher/avg"]:
            down = round(100 * d["compressed/avg"] / d["teacher/avg"], 1)
        rows.append({
            "model": d.get("model", "").split("/")[-1],
            "tag": d.get("run_tag", ""),
            "select": d.get("selection", ""),
            "method": d.get("method", ""),
            "heal": d.get("heal", ""),
            "reduction%": round(d.get("param_reduction_pct", 0), 1),
            "ppl_ret%": round(d.get("retained_pct", 0), 1),
            "downstream%": down,
        })
    rows.sort(key=lambda r: (r["model"], r["reduction%"]))
    cols = ["model", "tag", "select", "method", "heal", "reduction%", "ppl_ret%", "downstream%"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"\n[wrote {args.tsv}: {len(rows)} trials]")


if __name__ == "__main__":
    main()
