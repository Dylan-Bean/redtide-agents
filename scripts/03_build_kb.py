"""Teaching step 3: build the knowledge-base vector index and self-test search.

Run: py scripts/03_build_kb.py

What to look at:
1. the first run downloads the BGE Chinese model (~100 MB) and embeds every
   entry (cached to disk — from the second run on it finishes in seconds);
2. which entries and scores each self-test query recalls — feel the
   difference between semantic search and keyword matching;
3. to use GLM embedding-3 instead: set EMBEDDING_PROVIDER=zhipu in .env with
   ZHIPUAI_API_KEY filled, then run this script again (the index rebuilds).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redtide.config import RETRIEVAL    # noqa: E402
from redtide.kb import KnowledgeBase    # noqa: E402

kb = KnowledgeBase(provider=RETRIEVAL.embedding_provider)
t0 = time.time()
kb.ensure()
print(f"索引就绪：{kb.stats()}（本次耗时 {time.time() - t0:.1f}s）")

for q in ["为什么雨后容易爆发赤潮",
          "夜光藻赤潮有毒吗",
          "赤潮治理最主流的应急技术是什么"]:
    print(f"\n查询：{q}")
    for d in kb.search(q, top_k=2):
        print(f"  [{d['id']}] ({d['score']}) {d['title']}  —  {d['category']}")
