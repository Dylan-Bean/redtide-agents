"""The four agent nodes: hypothesize → retrieve → audit → report.

Each node = a role system prompt + structured input + output parsing + state
update. Nodes never call each other directly; graph.py's StateGraph
orchestrates them, and the critic's "insufficient evidence" send-back is just
a conditional edge.

Note: the role system prompts stay in **Chinese** on purpose — the corpus and
the retrieval model are Chinese-optimized, so Chinese queries retrieve far
better. Translate them together with the corpus if you re-domain the project.

Parsing tolerance matters in practice: LLM JSON often arrives inside ```
fences or with preamble, so extract_json recovers it; the hypothesize node
also retries once with tighter constraints when truncation breaks the JSON.
"""
import json
import re
from typing import Any, TypedDict

from . import llm
from .config import ORCHESTRATOR, RETRIEVAL
from .loop import run_tool_loop
from .tools import TOOLS_SCHEMA, make_executor


class State(TypedDict, total=False):
    question: str
    hypotheses: list[dict]
    evidence: list[dict]     # {"doc": {...}, "query": str}
    critiques: dict
    round: int               # 已完成的"检索->审计"轮数
    ready: bool
    missing_evidence: list[str]
    report: str
    error: str


def extract_json(text: str) -> Any:
    """Recover the first valid JSON object/array from an LLM response
    (tolerates code fences and surrounding prose)."""
    if not text:
        raise ValueError("空输出")
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [fence.group(1), text] if fence else [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    # 兜底：找最外层花括号/方括号
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cand.find(opener), cand.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(cand[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"无法解析 JSON: {text[:120]}")


def _clip(text: str, n: int = 160) -> str:
    return text if len(text) <= n else text[:n] + "…"


def normalize_hypotheses(data: Any) -> list[dict]:
    """Coerce parsed JSON into a flat list of hypothesis dicts.

    Real models occasionally nest the output differently: a bare array, a dict
    of arrays, or list-wrapped items (the v1.0 baseline eval had two runs die
    with "'list' object has no attribute 'get'" from exactly this). Anything
    unusable yields an empty list, which the caller treats as a parse failure.
    """
    if isinstance(data, dict):
        data = data.get("hypotheses", data)
    items = data if isinstance(data, list) else [data]
    out: list[dict] = []
    for h in items:
        if isinstance(h, dict):
            out.append(h)
        elif isinstance(h, list):
            out.extend(x for x in h if isinstance(x, dict))
    return out


def make_nodes(kb, trace):
    """闭包工厂：注入知识库与轨迹记录器，返回节点函数字典。"""
    executor = make_executor(kb)
    search_tools = TOOLS_SCHEMA[:1]          # kb_search
    report_tools = TOOLS_SCHEMA              # kb_search + kb_stats + 阈值判定

    def hypothesize(state: State) -> dict:
        sys_p = ("【假设生成】你是赤潮机制推演专家。针对给定问题提出 2-3 个竞争性假设，"
                 "每个假设说明：支持它需要什么证据（support_expect）、"
                 "什么证据会推翻它（refute_expect）。"
                 "各字段务必精简（每条 ≤60 字）。只输出 JSON："
                 '{"hypotheses":[{"id":"H1","statement":"...","support_expect":"...","refute_expect":"..."}]}')
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"推演问题：{state['question']}"},
        ]
        data = None
        hyps: list[dict] = []
        for attempt in range(2):
            try:
                resp = llm.chat(messages, max_tokens=4096 if attempt else None)
                trace.log("hypothesize", "llm_response",
                          prompt_tokens=resp.prompt_tokens,
                          completion_tokens=resp.completion_tokens,
                          latency_s=resp.latency_s,
                          attempt=attempt + 1)
                data = extract_json(resp.content)
                hyps = normalize_hypotheses(data)
                if not hyps:
                    raise ValueError(f"unusable hypothesis structure: {str(data)[:120]}")
                break
            except ValueError:
                if attempt == 1:
                    raise
                # parse failure, truncation, or unexpected structure: tighten
                # the constraints and retry once
                messages[0]["content"] = (sys_p +
                                          " 注意：上次输出无法解析或超长。这次极简——"
                                          "每个字段 ≤40 字，只输出 JSON 本体，"
                                          '格式固定为 {"hypotheses":[{...},{...}]}。')
        trace.log("hypothesize", "output", n=len(hyps),
                  statements=[_clip(h.get("statement", "")) for h in hyps])
        return {"hypotheses": hyps}

    def _rewrite_queries(state: State) -> list[str]:
        """优化点1：检索前先让 LLM 把问题改写成 2-3 条领域查询。"""
        resp = llm.chat([
            {"role": "system", "content":
                "【查询改写】把用户的赤潮问题改写成 2-3 条适合知识库检索的中文短语查询，"
                "覆盖不同角度（机制/藻种/监测/治理）。只输出 JSON：{\"queries\":[\"...\"]}"},
            {"role": "user", "content":
                f"问题：{state['question']}\n待检验假设："
                + "；".join(h.get("statement", "") for h in state.get("hypotheses", []))},
        ], temperature=0.2)
        trace.log("retrieve", "query_rewrite", prompt_tokens=resp.prompt_tokens,
                  completion_tokens=resp.completion_tokens)
        try:
            return extract_json(resp.content).get("queries", []) or [state["question"]]
        except (ValueError, AttributeError):
            return [state["question"]]

    def retrieve(state: State) -> dict:
        queries = _rewrite_queries(state) if RETRIEVAL.query_rewrite else []
        sys_p = ("【证据检索】你是证据收集员。用 kb_search 工具为各假设检索领域证据，"
                 "每次检索聚焦一个假设的角度；检索完成后用一句话总结拿到什么证据。"
                 + (f"\n建议优先检索的查询：{json.dumps(queries, ensure_ascii=False)}"
                    if queries else ""))
        user_parts = [f"问题：{state['question']}",
                      "待检验假设：" + json.dumps(
                          [{"id": h.get("id"), "statement": h.get("statement"),
                            "support_expect": h.get("support_expect")}
                           for h in state.get("hypotheses", [])], ensure_ascii=False)]
        if state.get("missing_evidence"):
            user_parts.append("审计要求补充的证据：" +
                              json.dumps(state["missing_evidence"], ensure_ascii=False))
        msgs, collected = run_tool_loop(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": "\n".join(user_parts)}],
            search_tools, executor, trace=trace, agent="retrieve", max_steps=5)

        evidence = list(state.get("evidence", []))
        seen_ids = {e["doc"]["id"] for e in evidence}
        for item in collected:
            docs = item if isinstance(item, list) else item.get("docs", []) \
                if isinstance(item, dict) else []
            for d in docs:
                if isinstance(d, dict) and d.get("id") and d["id"] not in seen_ids:
                    seen_ids.add(d["id"])
                    evidence.append({"doc": d, "query": d.get("query", "")})
        trace.log("retrieve", "output", n_new=len(evidence) - len(state.get("evidence", [])),
                  total=len(evidence))
        return {"evidence": evidence}

    def critique(state: State) -> dict:
        sys_p = ("【审计批判】你是证据审计员。逐条判定每个假设"
                 "（supported 有证据支持 / refuted 被证据推翻 / uncertain 证据不足），"
                 "并判断证据整体是否足以出报告。证据不足时给出 missing_evidence。只输出 JSON："
                 '{"per_hypothesis":[{"id":"H1","verdict":"...","reason":"..."}],'
                 '"ready":true/false,"missing_evidence":["..."]}')
        evidence_view = "\n".join(
            f"[{e['doc']['id']}] {e['doc']['title']}：{_clip(e['doc']['abstract'], 120)}"
            for e in state.get("evidence", [])) or "（无证据）"
        resp = llm.chat([
            {"role": "system", "content": sys_p},
            {"role": "user", "content":
                f"问题：{state['question']}\n假设："
                + json.dumps(state.get("hypotheses", []), ensure_ascii=False)
                + f"\n已收集证据（第{state.get('round', 1)}轮）：\n{evidence_view}"},
        ])
        trace.log("critique", "llm_response", prompt_tokens=resp.prompt_tokens,
                  completion_tokens=resp.completion_tokens, latency_s=resp.latency_s)
        try:
            data = extract_json(resp.content)
            if isinstance(data, list):
                # the model sometimes returns a bare verdict array instead of
                # the documented object — wrap it instead of crashing
                verdicts = [h for h in data if isinstance(h, dict)]
                data = {"per_hypothesis": verdicts,
                        "ready": bool(verdicts) and all(
                            h.get("verdict") in ("supported", "refuted")
                            for h in verdicts),
                        "missing_evidence": []}
            if not isinstance(data, dict):
                raise ValueError("critic output is neither object nor array")
        except (ValueError, AttributeError, TypeError):
            # unparsable output: proceed to report rather than stall the run
            data = {"ready": True, "per_hypothesis": [], "missing_evidence": []}
        can_return = ORCHESTRATOR.critic_allow_return and state.get("round", 1) < 2
        ready = bool(data.get("ready")) or not can_return
        trace.log("critique", "output", ready=ready,
                  missing=data.get("missing_evidence", []))
        return {"critiques": data, "ready": ready,
                "round": state.get("round", 1) + (0 if ready else 1),
                "missing_evidence": data.get("missing_evidence", [])}

    def report(state: State) -> dict:
        sys_p = ("【报告撰写】你是报告撰写员。基于假设、证据与审计结论撰写推演报告，"
                 "结构：## 推演报告 / **结论**（按证据强度分述） / **证据与推理** / "
                 "**置信度与数据缺口**。引用证据时必须标注编号，如 [KB016]。"
                 "若问题涉及具体细胞密度数值，必须调用 check_warning_threshold 工具判定。")
        evidence_view = "\n".join(
            f"[{e['doc']['id']}] {e['doc']['title']}：{_clip(e['doc']['abstract'], 200)}"
            for e in state.get("evidence", [])) or "（无证据）"
        msgs, collected = run_tool_loop(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content":
                 f"问题：{state['question']}\n审计结论："
                 + json.dumps(state.get("critiques", {}), ensure_ascii=False)
                 + f"\n可用证据：\n{evidence_view}"}],
            report_tools, executor, trace=trace, agent="report", max_steps=4)
        content = next((m["content"] for m in reversed(msgs)
                        if m["role"] == "assistant" and m.get("content")),
                       "（报告生成失败）")
        trace.log("report", "output", length=len(content),
                  extra_tool_results=len(collected))
        return {"report": content}

    return {"hypothesize": hypothesize, "retrieve": retrieve,
            "critique": critique, "report": report}
