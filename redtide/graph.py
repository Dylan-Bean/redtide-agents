"""LangGraph orchestration: a supervisor-style linear backbone with a
critic-driven conditional send-back.

Flow:
  START -> hypothesize -> retrieve -> critique -(insufficient evidence and
          send-back allowed)-> retrieve
                              └(otherwise)----------------------> report -> END

Why a graph instead of one giant prompt:
1. Single-role nodes keep prompts short, controllable and independently testable;
2. Conditional edges make "send back" a first-class citizen instead of hoping
   the model complies;
3. Explicit state passed between nodes makes trajectories and failure
   attribution natural.
"""
from langgraph.graph import END, START, StateGraph

from .agents import State, make_nodes


def route_after_critique(state: State) -> str:
    return "report" if state.get("ready") else "retrieve"


def build_graph(kb, trace):
    nodes = make_nodes(kb, trace)
    g = StateGraph(State)
    g.add_node("hypothesize", nodes["hypothesize"])
    g.add_node("retrieve", nodes["retrieve"])
    g.add_node("critique", nodes["critique"])
    g.add_node("report", nodes["report"])
    g.add_edge(START, "hypothesize")
    g.add_edge("hypothesize", "retrieve")
    g.add_edge("retrieve", "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"retrieve": "retrieve", "report": "report"})
    g.add_edge("report", END)
    return g.compile()


def run(question: str, kb, trace=None, tag: str = "") -> dict:
    """One full reasoning run. Creates and saves a Trace unless one is given."""
    from .trace import Trace
    trace = trace or Trace(question, tag=tag)
    app = build_graph(kb, trace)
    final = app.invoke({"question": question, "round": 1,
                        "hypotheses": [], "evidence": [], "missing_evidence": []})
    trace.log("meta", "end", error=final.get("error", ""),
              report_len=len(final.get("report", "")),
              total_tokens=trace.total_tokens)
    path = trace.save()
    return {"state": final, "trace_path": str(path), "trace": trace}
