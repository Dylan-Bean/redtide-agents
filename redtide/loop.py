"""Agent tool loop (production version): the LLM decides on its own when to
call tools, reads the results, and wraps up.

scripts/02_agent_loop.py is the standalone teaching version with identical
logic — between them they answer "can you implement an agent without a
framework?" with a plain while loop.
"""
import json

from . import llm


def run_tool_loop(messages, tools, executor, trace=None, agent="", max_steps=5):
    """Extend `messages` in place with one round of tool calling.

    Returns (messages, collected): collected holds every parsed tool result.
    """
    collected = []
    for _ in range(max_steps):
        resp = llm.chat(messages, tools=tools)
        if trace is not None:
            trace.log(agent, "llm_response",
                      prompt_tokens=resp.prompt_tokens,
                      completion_tokens=resp.completion_tokens,
                      latency_s=resp.latency_s, provider=resp.provider,
                      n_tool_calls=len(resp.tool_calls))
        if not resp.tool_calls:
            if resp.content:
                messages.append({"role": "assistant", "content": resp.content})
            return messages, collected

        messages.append({
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                           for tc in resp.tool_calls],
        })
        for tc in resp.tool_calls:
            if trace is not None:
                trace.log(agent, "tool_call", tool=tc.name, args=tc.arguments)
            result = executor(tc.name, tc.arguments)
            if trace is not None:
                trace.log(agent, "tool_result", tool=tc.name, result=result[:500])
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": tc.name, "content": result})
            try:
                collected.append(json.loads(result))
            except (json.JSONDecodeError, TypeError):
                collected.append(result)
    return messages, collected
