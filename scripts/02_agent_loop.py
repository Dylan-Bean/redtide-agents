"""Teaching step 2: hand-write the agent tool loop — no framework.

Run: py scripts/02_agent_loop.py

"What is an agent, really?" — the while loop below is the whole secret:
    model response -> tool_calls present? -> run tools -> feed results back -> ask again
                                     └-> none -> content is the final answer, exit
LangGraph and Co-Scientist are orchestration layers on top of exactly this
loop. Read this file and "can you write an agent without a framework?" is
answered.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redtide import llm                       # noqa: E402
from redtide.kb import KnowledgeBase          # noqa: E402
from redtide.tools import TOOLS_SCHEMA, make_executor  # noqa: E402

kb = KnowledgeBase()
kb.ensure()
executor = make_executor(kb)
TOOLS = TOOLS_SCHEMA[:1]   # 只挂 kb_search

messages = [
    {"role": "system",
     "content": "你是赤潮知识助手。回答前必须先用工具检索知识库；"
                "最终回答须标注引用的证据编号，如 [KB016]。"},
    {"role": "user", "content": "夜光藻赤潮有毒吗？养殖户需要封塘吗？"},
]

MAX_STEPS = 20
for step in range(1, MAX_STEPS + 1):
    print(f"\n{'=' * 60}\n第 {step} 轮：调用模型…")
    resp = llm.chat(messages, tools=TOOLS)

    if resp.tool_calls:
        print(f"→ 模型要求调用工具：{[(tc.name, tc.arguments) for tc in resp.tool_calls]}")
        messages.append({
            "role": "assistant", "content": resp.content or "",
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                           for tc in resp.tool_calls],
        })
        for tc in resp.tool_calls:
            result = executor(tc.name, tc.arguments)
            print(f"→ 执行 {tc.name}，返回前 300 字：\n{result[:300]}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": tc.name, "content": result})
    else:
        print(f"→ 模型给出最终回答（不再调工具，循环结束）：\n\n{resp.content}")
        print(f"\n[tokens] prompt={resp.prompt_tokens} "
              f"completion={resp.completion_tokens}")
        break
else:
    print("达到最大步数仍未收敛——生产系统里这属于需要归因的失败案例。")
