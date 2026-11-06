"""Evaluation report: markdown summary + comparison charts from results.jsonl.

Usage:
    python eval/report.py eval/output/baseline/results.jsonl
    python eval/report.py eval/output/baseline/results.jsonl eval/output/optimized/results.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "output"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    passes = sum(r["passed"] for r in rows)
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["passed"]
        c[1] += 1
    fails = Counter(r["failure_category"] for r in rows if not r["passed"])
    return {"n": n, "pass": passes, "rate": passes / n if n else 0,
            "avg_score": sum(r["score"] for r in rows) / n if n else 0,
            "tokens": sum(r["tokens"] for r in rows),
            "avg_latency": sum(r["latency_s"] for r in rows) / n if n else 0,
            "by_cat": by_cat, "fails": fails}


def make_chart(summaries: list[tuple[str, dict]], out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC",
                                       "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                       "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [n for n, _ in summaries]
    rates = [s["rate"] for _, s in summaries]
    axes[0].bar(names, rates, color=["#9db8d2", "#d29d9d"][: len(names)], width=0.5)
    for i, r in enumerate(rates):
        axes[0].text(i, r, f"{r:.0%}", ha="center", va="bottom")
    axes[0].set_title("Task pass rate (higher is better)")
    axes[0].set_ylim(0, 1.1)

    worst = summaries[-1][1]
    labels = list(worst["fails"].keys()) or ["(no failures)"]
    vals = list(worst["fails"].values()) or [0]
    axes[1].barh(labels, vals, color="#d2a59d")
    axes[1].set_title(f"{names[-1]} failure attribution")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"chart saved: {out_png}")


def main():
    paths = [Path(p) for p in sys.argv[1:]] or [OUT / "baseline" / "results.jsonl"]
    summaries = []
    md = ["# Evaluation report\n"]
    for p in paths:
        rows = load(p)
        s = summarize(rows)
        name = p.parent.name
        summaries.append((name, s))
        md.append(f"## Config: {name}\n")
        md.append(f"- tasks {s['n']}, passed **{s['pass']} ({s['rate']:.0%})**, "
                  f"avg score {s['avg_score']:.2f}")
        md.append(f"- avg latency {s['avg_latency']:.0f}s/task, total tokens {s['tokens']}\n")
        md.append("| category | passed/total |")
        md.append("|---|---|")
        for cat, (pn, nn) in s["by_cat"].items():
            md.append(f"| {cat} | {pn}/{nn} |")
        md.append("\nFailure attribution: " + (", ".join(f"{k}×{v}" for k, v in
                                                          s["fails"].items()) or "none") + "\n")
    if len(summaries) == 2:
        a, b = summaries[0][1], summaries[1][1]
        md.append(f"## Optimization result\n\npass rate {a['rate']:.0%} → **{b['rate']:.0%}**\n")
    out_md = OUT / "report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"report saved: {out_md}")
    print("\n".join(md))
    if len(sys.argv) > 1:
        make_chart(summaries, OUT / "compare.png")


if __name__ == "__main__":
    main()
