"""Tools (the core of Tool Use): the LLM reaches them through function calling.

Design principles:
1. Tool schemas are standard JSON Schema — what the model actually reads is the
   description, so descriptions deserve real care;
2. `execute_tool` always returns a string and never raises: tool errors are fed
   back to the model as messages so the agent can retry or change strategy —
   a crashing tool is the classic agent mistake;
3. The warning-threshold table ships as Chinese teaching data (species names and
   verdicts are Chinese, matching the corpus and eval tasks); magnitudes follow
   public industry standards.
"""
import json

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": "Semantic search in the red-tide domain knowledge base; "
                           "returns the most relevant entries (title, abstract, keywords, "
                           "score). Always search for evidence before answering domain questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query; domain keyword phrases work best"},
                    "top_k": {"type": "integer", "description": "Number of entries to return, default 3", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_stats",
            "description": "Knowledge-base statistics: total entries and category "
                           "distribution. Use it to describe evidence coverage.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_warning_threshold",
            "description": "Given a species name and a measured cell density (cells/L), "
                           "check it against the warning-threshold table. Species names are "
                           "fuzzy-matched; unlisted species fall back to a conservative default. "
                           "Returns Chinese field names/values (domain output).",
            "parameters": {
                "type": "object",
                "properties": {
                    "species": {"type": "string", "description": "Species name, e.g. 夜光藻 (Noctiluca), 亚历山大藻 (Alexandrium)"},
                    "cell_density": {"type": "number", "description": "Measured cell density in cells/L"},
                },
                "required": ["species", "cell_density"],
            },
        },
    },
]

# Teaching-example thresholds (cells/L); magnitudes follow public industry
# standards — species keys stay Chinese to match the corpus and eval tasks
WARNING_THRESHOLDS = {
    "夜光藻": 3e4,       # non-toxic, lenient threshold
    "亚历山大藻": 1e3,   # produces PSP (paralytic shellfish poison) — strict
    "米氏凯伦藻": 1e3,   # hemolytic toxin, severe fish kills
    "东海原甲藻": 1e4,
    "中肋骨条藻": 5e4,   # diatom, partly valuable as feed — lenient
    "链状裸甲藻": 1e3,   # PSP producer
}
DEFAULT_THRESHOLD = 1e4


def _match_threshold(species: str) -> tuple[str, float]:
    for name, th in WARNING_THRESHOLDS.items():
        if name in species or species in name:
            return name, th
    return species + "(未收录，用默认)", DEFAULT_THRESHOLD


def make_executor(kb):
    """Factory: inject the knowledge-base instance into the tool executor."""

    def execute_tool(name: str, args: dict) -> str:
        try:
            if name == "kb_search":
                docs = kb.search(args["query"], int(args.get("top_k", 3)))
                return json.dumps(docs, ensure_ascii=False, indent=1)

            if name == "kb_stats":
                return json.dumps(kb.stats(), ensure_ascii=False, indent=1)

            if name == "check_warning_threshold":
                sp = str(args["species"])
                dens = float(args["cell_density"])
                matched, th = _match_threshold(sp)
                ratio = dens / th
                level = "未超阈" if ratio < 1 else ("超阈-关注级" if ratio < 10 else "超阈-应急级")
                return json.dumps({
                    "species_matched": matched, "threshold_cells_per_L": th,
                    "measured_cells_per_L": dens, "exceed_ratio": round(ratio, 2),
                    "verdict": level,
                    "note": "Teaching-example thresholds; real response must follow current regulations",
                }, ensure_ascii=False, indent=1)

            return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)

        except Exception as exc:  # tools never raise — errors go back to the model
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)

    return execute_tool
