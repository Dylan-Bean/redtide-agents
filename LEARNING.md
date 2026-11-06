# LEARNING.md — understand every line in five days (~2–3 h/day)

> Commands use `py`, the Windows Python launcher; on macOS/Linux use `python3`.
>
> Role prompts, the corpus and the eval task set are in **Chinese on purpose** —
> the retrieval model (BGE-zh) and the demo domain are Chinese. Translate them
> freely when you swap in your own domain; everything else is English.

Goal: after this path you can explain why every line of this project exists.
Each day ends with self-check questions — if you can't answer one, re-read the
file it points to; that is where the key insight lives.

---

## Day 1 — The LLM API & the essence of an agent

**Run**: `py scripts/01_hello_glm.py` → `py scripts/02_agent_loop.py`
**Read**: `redtide/llm.py` (≤30 min; focus on how three providers normalize to one interface)

| Concept | Where to see it |
|---|---|
| Message roles: system / user / assistant / tool | script 01 |
| Tokens = the unit of billing and context length | prompt/completion tokens in script 01 output |
| Temperature: 0 = stable, 1 = creative; evaluations use low temperature | llm.py / config.py |
| **An agent = a while loop + function calling.** Nothing more magical | script 02 |
| Why a tool error must feed back to the model instead of crashing | tools.py try/except vs. script 02 |

**Self-check**:
1. What three things does a model-returned tool call contain? (id / name / arguments)
2. Why feed a tool result back as a `role=tool` message instead of appending it to the user message?
3. What happens if you set `MAX_STEPS` to 1 in script 02?

## Day 2 — RAG (retrieval-augmented generation)

**Run**: `py scripts/03_build_kb.py` (first run downloads the BGE model; watch what each self-test query retrieves)
**Read**: `redtide/embeddings.py` → `redtide/kb.py` → skim 5 entries of `knowledge/corpus.jsonl`

| Concept | Where to see it |
|---|---|
| Embedding = text → vector; semantically close = vector close | script 03 retrieval output |
| Queries and documents must share one embedding model; BGE wants a query-side instruction prefix | embeddings.py header |
| L2-normalized vectors: inner product = cosine similarity | embeddings.py |
| The embedding cache keeps repeated evaluation runs cheap | embeddings.py `_cache` |
| Index invalidation: rebuild whenever the model or corpus changes, or old and new vectors silently mix | kb.py `meta.json` |

**Self-check**:
1. What are the risks of raising vs. lowering `top_k`? (context noise vs. missing evidence)
2. How is the text chunked here, and why is "one entry = one chunk" acceptable?
3. Why must switching to embedding-3 force an index rebuild?

## Day 3 — Multi-agent orchestration (LangGraph)

**Run**: the Streamlit UI (`streamlit run app/streamlit_app.py`), ask an example question, expand the trajectory, and open the matching JSONL in `traces/`.
**Read**: `redtide/agents.py` (node by node) → `redtide/graph.py` (90 lines; focus on `add_conditional_edges`)

| Concept | Where to see it |
|---|---|
| State: the explicit data passed between nodes | `State` TypedDict in agents.py |
| A node = role system prompt + input assembly + output parsing + state update | any node in agents.py |
| Competing hypotheses: what would support / refute each one (guards against confirmation bias) | `hypothesize` node |
| **The send-back** is a conditional edge, not the model being "well-behaved" | graph.py `route_after_critique` |
| Output parsing tolerance: code fences, chatty preamble | `extract_json` in agents.py |
| Trajectories: one JSONL per run, the foundation of every analysis | trace.py |

**Self-check**:
1. When the critic says "evidence insufficient", what state changes, and who decides to route back to retrieve?
2. Why does each node wrap its work defensively so one failure only sets `state["error"]`?
3. How would you add a human-approval step before the report node?

## Day 4 — MCP: USB for tools

**Run**: `py scripts/04_test_mcp.py` (speak the protocol by hand) → merge the repo-root `.mcp.json` into your own project (Windows: `command` `python` → `py`), restart Claude Code, and see the three tools appear.
**Read**: `mcp/redtide_mcp_server.py` (30 lines)

| Concept | Where to see it |
|---|---|
| MCP = JSON-RPC 2.0 over stdio: initialize → tools/list → tools/call | script 04 output |
| The `@mcp.tool()` decorator: the docstring is the tool description the model reads | mcp server |
| One tool set, two consumers: in-process calls + external MCP clients | `make_executor` reuse in tools.py |

**Self-check**:
1. MCP vs. plain function calling? (protocol, cross-client reuse, process isolation)
2. Why does `tools/list` return descriptions too? (a new consumer still needs the prompt)

**A real pitfall from this repo (a good debugging story)**:
The first `tools/call` deadlocked in the MCP server. Root cause: the embedder
initialized its CUDA context on an anyio worker thread, which hangs on
Windows + torch 2.x. Fix: warm up one embedding call on the main thread at
server start (see the comment in `mcp/redtide_mcp_server.py`). Debugging path:
initialize/tools/list answered instantly while only tools/call hung → isolate
the inference step → minimal reproduction + one-variable-at-a-time experiments
→ thread/CUDA initialization ordering.

## Day 5 — Evaluation & failure attribution

**Run**:
```bash
py eval/run_eval.py --config baseline --limit 5 --tag smoke   # 5-task smoke test
py eval/report.py eval/output/smoke/results.jsonl
# with a real key:
py eval/run_eval.py --config baseline && py eval/run_eval.py --config optimized
py eval/report.py eval/output/baseline/results.jsonl eval/output/optimized/results.jsonl
```
**Read**: `eval/judge.py` → `eval/attribute.py` → `eval/run_eval.py`

| Concept | Where to see it |
|---|---|
| Task design: each item lists synonym groups of required points, rule-based and reproducible | eval/tasks.jsonl |
| Rule scoring vs. LLM-as-judge trade-offs | judge.py header |
| **Failure attribution buckets**: locate before you optimize, never blind-tune prompts | attribute.py |
| Comparison harness: same task set, only the switches move (rewrite / top_k / send-back) | run_eval.py CONFIGS |
| What mock mode proves: pipeline correctness ≠ answer quality | README note |

**Self-check**:
1. Why should evaluation use low temperature?
2. What is the difference between `evidence_underused` and `retrieval_miss`, and how do you fix each?
3. If optimized scores worse than baseline, how do you explain and debug it?
