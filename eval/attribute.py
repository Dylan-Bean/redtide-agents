"""Failure attribution: sort every failed task into a concrete bucket —
this is where the optimization loop starts.

Attribution rules (by priority):
  tool_error        a tool execution raised an error
  tool_not_called   the threshold tool should have been called but was not
                    (prompt / tool description problem)
  no_report         no report was produced (no convergence / crash)
  citation_missing  citations absent or non-existent (output format problem)
  retrieval_miss    no documents retrieved at all (retrieval quality problem)
  evidence_underused documents retrieved but key points not covered
                    (reasoning / synthesis problem)
  pass              passed
"""
import json


def attribute_task(task: dict, judge_res: dict, trace=None) -> dict:
    events = list(trace.events) if trace else []
    if any("error" in json.dumps(e.get("result", "")).lower()
           for e in events if e.get("action") == "tool_result"):
        return {"category": "tool_error"}
    if task.get("needs_threshold_tool") and not judge_res["tool_ok"]:
        return {"category": "tool_not_called"}
    report = judge_res.get("_report", "")
    if not report:
        return {"category": "no_report"}
    if task.get("must_cite") and not judge_res["cite_ok"]:
        return {"category": "citation_missing"}
    n_docs = sum(1 for e in events if e.get("action") == "tool_result"
                 and e.get("tool") == "kb_search")
    if n_docs == 0:
        return {"category": "retrieval_miss"}
    if judge_res["hits"] == 0:
        return {"category": "evidence_underused"}
    return {"category": "evidence_underused"}
