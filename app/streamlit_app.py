"""Red Tide Multi-Agent Reasoning — demo UI.

Run (from the repo root): streamlit run app/streamlit_app.py

UI labels are English; the example questions stay Chinese because the built-in
corpus and retrieval model are Chinese-optimized.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from redtide import config as cfg
from redtide import llm
from redtide.kb import KnowledgeBase
from redtide.svg_art import SVGGenerationError, generate_report_svg

st.set_page_config(page_title="Red Tide Multi-Agent Reasoning", page_icon="🌊", layout="wide")

EXAMPLES = [
    "为什么暴雨后一周内容易发生赤潮？请给出机制链条。",
    "夜光藻赤潮有毒吗？养殖户需要采取什么措施？",
    "监测到米氏凯伦藻 5000 cells/L，是否超过预警阈值？应启动什么级别响应？",
    "为什么控源截污多年后赤潮仍然年年发生？",
]


@st.cache_resource(show_spinner="Loading knowledge base and embedding model…")
def get_kb() -> KnowledgeBase:
    kb = KnowledgeBase(provider=cfg.RETRIEVAL.embedding_provider)
    kb.ensure()
    return kb


# ---------------- Sidebar: system status and switches ----------------

with st.sidebar:
    st.header("System status")
    provider = llm.detect_provider()
    st.markdown(f"- LLM backend: `{provider}`"
                + (" (no key configured — offline mock mode)" if provider == "mock" else ""))
    st.json(get_kb().stats(), expanded=False)

    st.header("Optimization switches (for A/B evaluation)")
    cfg.RETRIEVAL.query_rewrite = st.toggle(
        "① Query rewriting before retrieval", value=cfg.RETRIEVAL.query_rewrite,
        help="Let an LLM rewrite the question into several domain queries first")
    cfg.RETRIEVAL.top_k = st.slider("Retrieval top_k", 1, 6, cfg.RETRIEVAL.top_k)
    cfg.ORCHESTRATOR.critic_allow_return = st.toggle(
        "② Allow audit agent to send work back", value=cfg.ORCHESTRATOR.critic_allow_return,
        help="When evidence is insufficient the audit agent may request one more retrieval round")

# ---------------- Main area ----------------

st.title("🌊 Red Tide Multi-Agent Reasoning")
st.caption("Hypothesis generation → evidence retrieval (RAG) → evidence audit (with send-back) → "
           "report writing · GLM API + LangGraph + FAISS + MCP")

question = st.text_area("Question", key="question",
                        value=st.session_state.get("question", EXAMPLES[0]), height=80)
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(f"Example {i + 1}", key=f"ex{i}", help=ex):
        st.session_state["question"] = ex
        st.rerun()

if st.button("Run reasoning", type="primary"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        from redtide.graph import run as run_graph
        with st.status("Reasoning in progress…", expanded=True) as status:
            st.write("① Generating competing hypotheses…")
            result = run_graph(question, get_kb(), tag="ui")
            st.write("② Retrieving evidence → ③ auditing (verdict / send-back) → ④ writing the report ✓")
            status.update(label="Reasoning complete", state="complete", expanded=False)

        state = result["state"]
        st.session_state["last_result"] = {
            "state": state,
            "trace_events": result["trace"].events,
            "trace_id": result["trace"].run_id,
            "total_tokens": result["trace"].total_tokens,
        }
        st.session_state["generated_svg"] = ""
        st.session_state["svg_generation_error"] = ""
        st.session_state["svg_question"] = state.get("question", question)
        st.session_state["svg_report"] = state.get("report", "")

if last_result := st.session_state.get("last_result"):
    state = last_result["state"]
    if state.get("error"):
        st.error(f"Run failed: {state['error']}")
    st.markdown(state.get("report") or "(no report output)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Evidence", f"{len(state.get('evidence', []))} docs")
    m2.metric("Rounds", state.get("round", 1))
    m3.metric("tokens", last_result["total_tokens"])

    st.subheader("Report figure (SVG)")
    st.caption("Generated on demand by the configured LLM as SVG markup and rendered after "
               "sanitization — this is code-drawn, not native image generation.")
    if st.button("Generate report figure", key="generate_report_svg"):
        st.session_state["svg_generation_error"] = ""
        with st.spinner("Generating SVG figure…"):
            try:
                st.session_state["generated_svg"] = generate_report_svg(
                    st.session_state["svg_question"],
                    st.session_state["svg_report"],
                )
            except SVGGenerationError as exc:
                st.session_state["svg_generation_error"] = str(exc)

    if st.session_state.get("svg_generation_error"):
        st.error(st.session_state["svg_generation_error"])
    if st.session_state.get("generated_svg"):
        svg = st.session_state["generated_svg"]
        st.html(svg)
        st.download_button(
            "Download SVG figure",
            data=svg,
            file_name="redtide-report.svg",
            mime="image/svg+xml",
        )

    with st.expander("🔍 Trajectory (what each agent did, step by step)"):
        for e in last_result["trace_events"]:
            icon = {"llm_response": "🧠", "tool_call": "🔧",
                    "tool_result": "📄", "output": "✅"}.get(e["action"], "•")
            brief = {k: v for k, v in e.items()
                     if k not in ("t", "run_id") and v != ""}
            st.markdown(f"`{icon} {e['agent']} · {e['action']}`  \n"
                        f"```\n{str(brief)[:400]}\n```")

    st.download_button("Download trajectory JSONL",
                       data="\n".join(str(e) for e in last_result["trace_events"]),
                       file_name=f"{last_result['trace_id']}.jsonl",
                       mime="application/jsonl")
