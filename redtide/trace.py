"""Run-trajectory logging: one JSONL file per run, one event per line.

This is the data backbone behind trajectory analysis and failure attribution —
scoring, attribution, the UI timeline and post-run reviews all read from here.
"""
import json
import time
from datetime import datetime
from pathlib import Path

from .config import TRACE_DIR


class Trace:
    def __init__(self, question: str, tag: str = ""):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + \
            (tag or "run") + "_" + str(int(time.time() * 1000) % 100000)
        self.question = question
        self.events = []
        self.log("meta", "start", question=question, tag=tag)

    def log(self, agent: str, action: str, **payload):
        self.events.append({
            "t": round(time.time(), 3),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **payload,
        })

    @property
    def tool_calls(self):
        return [e for e in self.events if e["action"] == "tool_call"]

    @property
    def n_llm_calls(self):
        return sum(1 for e in self.events if e["action"] == "llm_response")

    @property
    def total_tokens(self):
        return sum(e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
                   for e in self.events if e["action"] == "llm_response")

    def save(self, outdir: Path | None = None) -> Path:
        outdir = Path(outdir or TRACE_DIR)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{self.run_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return path
