"""Unified LLM client: three backends behind one interface.

  zhipu             the official Zhipu SDK (chat.completions, OpenAI-style
                    function calling)
  anthropic_compat  any Anthropic-compatible gateway (Zhipu Coding Plan,
                    self-hosted proxies)
  mock              a deterministic offline fake model for tests and demos

Internally messages use the OpenAI style (system/user/assistant/tool); each
provider translates them to its own wire format, so agent code upstream never
knows which vendor is serving the request.
"""
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, replace

import httpx

from .config import LLM

# Default model (GLM-5.3-Flash); if the API reports "model not found",
# set LLM_MODEL in .env to a working id (e.g. glm-4-flash, the free stable tier)
ZHIPU_DEFAULT_MODEL = "glm-5.3-flash"
ANTHROPIC_DEFAULT_MODEL = "glm-5.3-flash"
# Note: the Anthropic-compatible gateway path never sends the thinking field
# (gateway implementations differ). If reasoning starves the answer there,
# just raise LLM_MAX_TOKENS.


@dataclass
class ToolCall:
    """模型要求调用某个工具的一次请求。"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """三种后端归一化后的响应。上层只看这个对象。"""
    content: str = ""
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = ""
    latency_s: float = 0.0


# --------------------------------------------------------------------------
# 消息格式翻译
# --------------------------------------------------------------------------

def _to_zhipu_messages(messages):
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": [{
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"],
                                 "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                } for tc in m["tool_calls"]],
            })
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def _to_anthropic_payload(messages, tools, model, temperature, max_tokens):
    system_parts, convo = [], []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        elif m["role"] == "assistant" and m.get("tool_calls"):
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": tc["name"], "input": tc["arguments"]})
            convo.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            convo.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": m["tool_call_id"],
                "content": m.get("content") or "",
            }]})
        else:
            convo.append({"role": m["role"], "content": m.get("content") or ""})
    tool_schema = [{
        "name": t["function"]["name"],
        "description": t["function"].get("description", ""),
        "input_schema": t["function"]["parameters"],
    } for t in (tools or [])]
    payload = {
        "model": model, "messages": convo, "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if tool_schema:
        payload["tools"] = tool_schema
    return payload


# --------------------------------------------------------------------------
# 三种后端
# --------------------------------------------------------------------------

def _chat_zhipu(messages, tools, cfg: LLM, temperature, max_tokens) -> LLMResponse:
    from zhipuai import ZhipuAI  # 延迟导入：mock 模式不需要装
    client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])

    def _call(**extra):
        kwargs = dict(
            model=cfg.model or ZHIPU_DEFAULT_MODEL,
            messages=_to_zhipu_messages(messages),
            temperature=temperature, max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        kwargs.update(extra)
        return client.chat.completions.create(**kwargs)

    if LLM.thinking != "default":
        try:
            resp = _call(thinking={"type": "enabled", "effort": LLM.thinking})
        except Exception as exc:
            # 部分模型不支持 thinking 参数：去掉后原样重发（错误码 1210 = 思考档位无效）
            if "思考" in str(exc) or "1210" in str(exc):
                resp = _call()
            else:
                raise
    else:
        resp = _call()
    msg = resp.choices[0].message
    tcs = [ToolCall(id=tc.id, name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"))
           for tc in (getattr(msg, "tool_calls", None) or [])]
    return LLMResponse(content=msg.content or "", tool_calls=tcs,
                       prompt_tokens=resp.usage.prompt_tokens,
                       completion_tokens=resp.usage.completion_tokens,
                       provider="zhipu")


def _chat_anthropic_compat(messages, tools, cfg: LLM, temperature, max_tokens) -> LLMResponse:
    url = os.environ["ANTHROPIC_BASE_URL"].rstrip("/") + "/v1/messages"
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "authorization": "Bearer " + os.environ["ANTHROPIC_AUTH_TOKEN"],
    }
    payload = _to_anthropic_payload(messages, tools,
                                    cfg.model or ANTHROPIC_DEFAULT_MODEL,
                                    temperature, max_tokens)
    with httpx.Client(timeout=120) as cli:
        r = cli.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    content, tcs = "", []
    for block in data.get("content", []):
        if block["type"] == "text":
            content += block["text"]
        elif block["type"] == "tool_use":
            tcs.append(ToolCall(id=block["id"], name=block["name"],
                                arguments=block.get("input") or {}))
    usage = data.get("usage", {})
    return LLMResponse(content=content, tool_calls=tcs,
                       prompt_tokens=usage.get("input_tokens", 0),
                       completion_tokens=usage.get("output_tokens", 0),
                       provider="anthropic_compat")


# --------------------------------------------------------------------------
# Mock backend: deterministic answers keyed on the role tag in the latest
# system message plus whether tool results already exist. The canned text is
# Chinese to mirror the agents' Chinese role prompts (see agents.py).
# --------------------------------------------------------------------------

def _latest_system_tag(messages) -> str:
    """Which agent role is calling? Matched against the Chinese role tags
    used in agents.py system prompts."""
    for m in reversed(messages):
        if m["role"] == "system":
            c = m["content"]
            for tag in ("假设生成", "证据检索", "审计批判", "报告撰写", "查询改写"):
                if tag in c:
                    return tag
    return ""


def _has_tool_result(messages) -> bool:
    return any(m["role"] == "tool" for m in messages)


def _chat_mock(messages, tools, cfg: LLM, temperature, max_tokens) -> LLMResponse:
    tag = _latest_system_tag(messages)
    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    # user 消息可能是"问题：xxx\n假设：…"的拼装串，报告模板只取第一行问题本体
    question = question.splitlines()[0].replace("问题：", "").strip()
    n_tool_results = sum(1 for m in messages if m["role"] == "tool")

    if tag == "假设生成":
        content = json.dumps({
            "hypotheses": [
                {"id": "H1", "statement": f"「{question[:40]}」主要受营养盐（氮磷）富集驱动",
                 "support_expect": "营养盐浓度/氮磷比升高与藻华增长的关联证据",
                 "refute_expect": "营养盐与藻情变化脱钩的证据"},
                {"id": "H2", "statement": f"「{question[:40]}」主要受水文气象条件（水温、分层、上涌）驱动",
                 "support_expect": "海温/分层/环流与藻华爆发的关联证据",
                 "refute_expect": "气象条件平稳年份仍爆发的证据"},
            ]}, ensure_ascii=False, indent=1)
        return LLMResponse(content=content, prompt_tokens=500, completion_tokens=180,
                           provider="mock")

    if tag == "查询改写":
        queries = [question, question + " 机制 营养盐 水文", question + " 监测 阈值 治理"]
        return LLMResponse(content=json.dumps({"queries": queries}, ensure_ascii=False),
                           prompt_tokens=200, completion_tokens=60, provider="mock")

    if tag == "证据检索":
        if not _has_tool_result(messages):
            # 第一轮：发起工具调用（教学点——mock 也走同一条 function calling 通路）
            tc = ToolCall(id="mock_" + uuid.uuid4().hex[:8], name="kb_search",
                          arguments={"query": question, "top_k": 3})
            return LLMResponse(content="", tool_calls=[tc],
                               prompt_tokens=400, completion_tokens=30, provider="mock")
        # 第二轮：读完工具结果，做证据小结
        docs = json.loads(messages[-1]["content"]) if n_tool_results else []
        titles = "；".join(d.get("title", "") for d in docs if isinstance(d, dict))
        return LLMResponse(content=f"已检索 {len(docs)} 条证据：{titles}",
                           prompt_tokens=600, completion_tokens=80, provider="mock")

    if tag == "审计批判":
        # 默认一轮即通过；REDTIDE_MOCK_CRITIC_RETURN=1 时强制打回一次（测打回链路）
        send_back = os.getenv("REDTIDE_MOCK_CRITIC_RETURN") == "1" and \
            not any("补充检索" in (m.get("content") or "") for m in messages if m["role"] == "tool")
        verdict = {"per_hypothesis": [
            {"id": "H1", "verdict": "uncertain" if send_back else "supported",
             "reason": "教学 mock 判定"},
            {"id": "H2", "verdict": "refuted", "reason": "教学 mock 判定"}],
            "ready": not send_back,
            "missing_evidence": ["需要更多营养盐过程证据"] if send_back else []}
        return LLMResponse(content=json.dumps(verdict, ensure_ascii=False),
                           prompt_tokens=700, completion_tokens=120, provider="mock")

    if tag == "报告撰写":
        text_all = " ".join((m.get("content") or "") for m in messages)
        doc_ids = list(dict.fromkeys(re.findall(r"\[KB\d+\]", text_all)))
        cites = "".join(doc_ids[:6])
        cites = "".join(f"[{i}]" for i in dict.fromkeys(doc_ids)) or "[KB001]"
        return LLMResponse(
            content=f"## 推演报告（mock）\n\n**问题**：{question}\n\n"
                    f"**结论**：营养盐与水文条件的共同作用为主因（mock 示例结论，仅用于流程演示）。\n\n"
                    f"**证据与推理**：检索到 {len(doc_ids)} 条知识库证据 {cites}，"
                    f"其要点与问题相关，详见轨迹文件。\n\n"
                    f"**置信度与数据缺口**：内置教学语料有限，结论不代表真实科研判断。",
            prompt_tokens=800, completion_tokens=200, provider="mock")

    return LLMResponse(content="(mock) 收到。", provider="mock")


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------

def detect_provider() -> str:
    if LLM.provider:
        return LLM.provider
    if os.environ.get("ZHIPUAI_API_KEY"):
        return "zhipu"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic_compat"
    return "mock"


def chat(messages, tools=None, temperature=None, max_tokens=None,
         model=None) -> LLMResponse:
    """统一入口。messages 为 OpenAI 风格；tools 为 OpenAI 风格 function schema。

    model 可按次覆盖全局配置（如 SVG 插图走轻量免思考模型）。
    """
    provider = detect_provider()
    fn = {"zhipu": _chat_zhipu,
          "anthropic_compat": _chat_anthropic_compat,
          "mock": _chat_mock}[provider]
    cfg = replace(LLM, model=model or LLM.model) if model else LLM
    t0 = time.time()
    resp = fn(messages, tools, cfg,
              LLM.temperature if temperature is None else temperature,
              LLM.max_tokens if max_tokens is None else max_tokens)
    resp.latency_s = round(time.time() - t0, 2)
    return resp
