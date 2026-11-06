"""MCP server: exposes the same knowledge-base tools over the MCP protocol.

Why this matters: wrap the tools once, use them everywhere —
1. this project's agents call the executor in-process;
2. any MCP client (Claude Code / Claude Desktop / other agent frameworks)
   connects over stdio and gets the same tools.

Run: py mcp/redtide_mcp_server.py
  (stdio mode: no output and a hanging process means it is waiting for a
   client — that is normal; Ctrl+C to exit)
Connect Claude Code: merge the repo-root .mcp.json into your project
  (Windows users: change `command` from python to py)
Protocol self-test: py scripts/04_test_mcp.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP   # noqa: E402

from redtide.kb import KnowledgeBase     # noqa: E402
from redtide.tools import make_executor  # noqa: E402

kb = KnowledgeBase()
kb.ensure()
# 预热：在主线程先跑一次嵌入，提前初始化 CUDA 上下文。
# 否则首个 tools/call 在 anyio 工作线程里首次初始化 CUDA，
# Windows 上（torch 2.x）可能死锁——这是本机踩过的真实坑。
kb.embedder.embed_query("预热")

executor = make_executor(kb)

mcp = FastMCP("redtide-kb")


@mcp.tool()
def kb_search(query: str, top_k: int = 3) -> str:
    """在赤潮领域知识库中做语义检索，返回最相关条目（标题、摘要、相似度）。"""
    return executor("kb_search", {"query": query, "top_k": top_k})


@mcp.tool()
def kb_stats() -> str:
    """查看知识库统计：条目总数与分类分布。"""
    return executor("kb_stats", {})


@mcp.tool()
def check_warning_threshold(species: str, cell_density: float) -> str:
    """按预警阈值表判定实测细胞密度（cells/L）是否超阈、超阈倍数与响应级别。"""
    return executor("check_warning_threshold",
                    {"species": species, "cell_density": cell_density})


if __name__ == "__main__":
    mcp.run(transport="stdio")
