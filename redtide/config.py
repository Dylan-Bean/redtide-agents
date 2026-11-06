"""Central configuration: every knob lives in this file.

The baseline vs optimized evaluation comparison is just three switches here:
  RETRIEVAL_QUERY_REWRITE   let an LLM rewrite the query before retrieval (optimization 1)
  RETRIEVAL_TOP_K           how many knowledge-base entries each search returns
  CRITIC_ALLOW_RETURN       whether the audit agent may send work back for one more
                            retrieval round (optimization 2)

Flip them via environment variables (.env) — no code changes — which is what
lets eval/run_eval.py compare --config baseline / optimized automatically.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class LLMConfig:
    # Empty provider auto-detects:
    #   ZHIPUAI_API_KEY present      -> zhipu (official SDK)
    #   ANTHROPIC_AUTH_TOKEN present -> anthropic_compat (Coding Plan / gateway)
    #   neither                      -> mock (deterministic offline model)
    provider: str = os.getenv("LLM_PROVIDER", "")
    model: str = os.getenv("LLM_MODEL", "")  # empty = provider default
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    # 4096: real models writing structured JSON routinely need more than 2048
    # tokens; truncation breaks JSON parsing downstream
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 4096)
    # Thinking effort: GLM-5.x always thinks and a heavy default starves the
    # output budget (zero answer tokens); "low" prioritizes the final answer.
    # "default" omits the thinking field for models that reject it.
    # One of: low | high | max | default
    thinking: str = os.getenv("LLM_THINKING", "low")


@dataclass
class RetrievalConfig:
    # local = local BGE Chinese model (free, offline, reproducible); zhipu = GLM embedding-3
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    top_k: int = _env_int("RETRIEVAL_TOP_K", 3)
    query_rewrite: bool = _env_bool("RETRIEVAL_QUERY_REWRITE", False)  # optimization 1


@dataclass
class OrchestratorConfig:
    # Optimization 2: when the audit agent judges evidence insufficient it may
    # send the question back to retrieval for one more round
    critic_allow_return: bool = _env_bool("CRITIC_ALLOW_RETURN", False)


LLM = LLMConfig()
RETRIEVAL = RetrievalConfig()
ORCHESTRATOR = OrchestratorConfig()

KB_CORPUS = ROOT / "knowledge" / "corpus.jsonl"
KB_DIR = ROOT / "knowledge" / "index"
TRACE_DIR = ROOT / "traces"
