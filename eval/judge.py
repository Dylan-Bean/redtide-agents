"""Evaluation scorer: rule-based first, optional LLM re-check.

Design trade-offs:
- Rule scoring (keyword-group coverage + citation check + tool-usage check) is
  free, reproducible and explainable, but cannot recognize semantic paraphrase —
  which is why each key-point group lists synonyms separated by "|";
- LLM-as-judge is semantically fair but costs tokens and is not fully
  reproducible. Production evaluation combines both: rules first, LLM re-check
  for borderline samples.
"""
import json
import re
from pathlib import Path

PASS_SCORE = 0.6   # 覆盖 60% 关键词组即通过


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge_task(task: dict, report: str, trace=None) -> dict:
    report_norm = re.sub(r"\s+", "", report or "").lower()
    hits, misses = [], []
    for group in task.get("key_points", []):
        alts = [a.strip().lower() for a in group.split("|")]
        if any(a and a.replace(" ", "") in report_norm for a in alts):
            hits.append(group)
        else:
            misses.append(group)
    score = len(hits) / len(task["key_points"]) if task.get("key_points") else 0.0

    cite_ok = (not task.get("must_cite")) or bool(re.search(r"\[KB\d+\]", report or ""))
    tool_events = [e for e in (trace.tool_calls if trace else [])
                   if e.get("tool") == "check_warning_threshold"]
    tool_ok = (not task.get("needs_threshold_tool")) or bool(tool_events)

    passed = score >= PASS_SCORE and cite_ok and tool_ok
    return {"score": round(score, 3), "hits": hits, "misses": misses,
            "cite_ok": cite_ok, "tool_ok": tool_ok, "passed": passed}
