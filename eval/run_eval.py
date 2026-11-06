"""Evaluation runner: run the task set in batch → score → attribute → write results.

Usage (from the repo root):
    python eval/run_eval.py --config baseline     # all optimization switches off
    python eval/run_eval.py --config optimized    # query rewriting + send-back + top_k=5
    python eval/run_eval.py --config baseline --limit 5 --tag smoke   # 5-task smoke run

Output: eval/output/<tag>/results.jsonl (one full record per task).
The before/after pass rates "A% -> B%" used to demonstrate optimization come
from the baseline and optimized configs respectively.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redtide import config as cfg            # noqa: E402
from redtide.kb import KnowledgeBase         # noqa: E402
from redtide.graph import run as run_graph   # noqa: E402

from judge import judge_task, load_tasks     # noqa: E402
from attribute import attribute_task         # noqa: E402

CONFIGS = {
    "baseline":  {"query_rewrite": False, "top_k": 3, "critic_allow_return": False},
    "optimized": {"query_rewrite": True, "top_k": 5, "critic_allow_return": True},
}


def apply_config(name: str):
    c = CONFIGS[name]
    cfg.RETRIEVAL.query_rewrite = c["query_rewrite"]
    cfg.RETRIEVAL.top_k = c["top_k"]
    cfg.ORCHESTRATOR.critic_allow_return = c["critic_allow_return"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="baseline")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全部")
    ap.add_argument("--tag", default="", help="输出子目录名，默认同 config")
    args = ap.parse_args()
    apply_config(args.config)
    tag = args.tag or args.config

    tasks = load_tasks(Path(__file__).parent / "tasks.jsonl")
    if args.limit:
        tasks = tasks[:args.limit]

    kb = KnowledgeBase(provider=cfg.RETRIEVAL.embedding_provider)
    kb.ensure()

    outdir = Path(__file__).parent / "output" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        started = time.time()
        try:
            res = run_graph(task["question"], kb, tag=tag)
            state, trace = res["state"], res["trace"]
            report = state.get("report", "")
        except Exception as exc:                     # 单任务失败不拖垮整批
            state, trace, report = {"error": str(exc)}, None, ""
        judged = judge_task(task, report, trace)
        judged["_report"] = report
        attr = attribute_task(task, judged, trace)
        row = {
            "task_id": task["id"], "category": task["category"],
            "question": task["question"],
            "score": judged["score"], "passed": judged["passed"],
            "cite_ok": judged["cite_ok"], "tool_ok": judged["tool_ok"],
            "misses": judged["misses"], "failure_category": attr["category"],
            "rounds": state.get("round", 0),
            "n_evidence": len(state.get("evidence", [])),
            "tokens": trace.total_tokens if trace else 0,
            "latency_s": round(time.time() - started, 1),
            "error": state.get("error", ""),
            "trace_path": res["trace_path"] if trace else "",
            "report": report,
        }
        rows.append(row)
        flag = "✓" if row["passed"] else f"✗({row['failure_category']})"
        print(f"[{i}/{len(tasks)}] {task['id']} {flag} score={row['score']} "
              f"rounds={row['rounds']} tokens={row['tokens']}")

    out = outdir / "results.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pass = sum(r["passed"] for r in rows)
    print("\n" + "=" * 60)
    print(f"配置={args.config}  通过 {n_pass}/{len(rows)} = "
          f"{n_pass / len(rows):.0%}  总耗时 {time.time() - t0:.0f}s  "
          f"总 tokens={sum(r['tokens'] for r in rows)}")
    print(f"结果文件：{out}")


if __name__ == "__main__":
    main()
