"""Teaching step 1: say hello to GLM.

Run (from the repo root):
    py scripts/01_hello_glm.py

What to look at:
1. how the answer changes with the system prompt;
2. prompt/completion tokens — the unit of billing and context length;
3. without an API key the provider falls back to mock and still runs
   (deterministic canned replies).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redtide import llm  # noqa: E402

resp = llm.chat([
    {"role": "system", "content": "你是赤潮科普助手，回答控制在三句话以内。"},
    {"role": "user", "content": "一句话解释：为什么赤潮的海水会变色？"},
])

print("=" * 60)
print(resp.content)
print("=" * 60)
print(f"[provider={resp.provider}] prompt_tokens={resp.prompt_tokens} "
      f"completion_tokens={resp.completion_tokens} latency={resp.latency_s}s")
