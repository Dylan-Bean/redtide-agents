"""One-shot reasoning entry point.

Run: py scripts/05_run_once.py "Why do red tides often follow heavy rain?"
Prints the report and the trajectory file path when done.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redtide.config import RETRIEVAL     # noqa: E402
from redtide.graph import run            # noqa: E402
from redtide.kb import KnowledgeBase     # noqa: E402

question = sys.argv[1] if len(sys.argv) > 1 else "为什么雨后容易爆发赤潮？"

kb = KnowledgeBase(provider=RETRIEVAL.embedding_provider)
kb.ensure()
result = run(question, kb, tag="once")

print(result["state"].get("report", "(无报告)"))
print("\n" + "=" * 60)
print(f"轮次: {result['state'].get('round')}  证据: "
      f"{len(result['state'].get('evidence', []))} 条  "
      f"tokens: {result['trace'].total_tokens}")
print(f"轨迹: {result['trace_path']}")
