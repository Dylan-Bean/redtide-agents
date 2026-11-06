"""Domain knowledge base: FAISS vector search + metadata management.

The corpus ships as 40 built-in Chinese teaching entries on red tide
(knowledge/corpus.jsonl, compiled from public domain knowledge — replace the
file with your own corpus in any language and the index rebuilds itself).

Index invalidation: meta.json records the embedding provider plus a content
hash of the corpus; if either changes the index rebuilds automatically —
this avoids the classic accident of querying new vectors against an old index.
"""
import hashlib
import json

import numpy as np

try:
    import faiss
except ImportError:      # fall back to numpy search if faiss is missing
    faiss = None

from .config import KB_CORPUS, KB_DIR
from .embeddings import Embedder

INDEX_FILE = KB_DIR / "index.npz"
META_FILE = KB_DIR / "meta.json"


def _corpus_hash(path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def load_corpus(path=KB_CORPUS) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


class KnowledgeBase:
    def __init__(self, provider: str = "local"):
        self.provider = provider
        self.embedder = Embedder(provider)
        self.entries = load_corpus()
        self._index = None  # faiss.IndexFlatIP, built lazily

    # ---- build / load ----

    def ensure(self):
        """Load the index if it exists and matches; rebuild otherwise."""
        meta = self._meta_ok()
        if meta and INDEX_FILE.exists():
            data = np.load(INDEX_FILE)
            self._matrix = data["vecs"]
        else:
            self.build()

    def _meta_ok(self) -> dict | None:
        if not META_FILE.exists():
            return None
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        if (meta.get("provider") == self.provider
                and meta.get("corpus_hash") == _corpus_hash(KB_CORPUS)):
            return meta
        return None

    def build(self):
        KB_DIR.mkdir(parents=True, exist_ok=True)
        texts = [f"{e['title']}。{e['abstract']} 关键词：{'、'.join(e['keywords'])}"
                 for e in self.entries]
        vecs = self.embedder.embed_documents(texts)
        np.savez(INDEX_FILE, vecs=vecs)
        META_FILE.write_text(json.dumps({
            "provider": self.provider,
            "dim": int(vecs.shape[1]),
            "n": int(vecs.shape[0]),
            "corpus_hash": _corpus_hash(KB_CORPUS),
        }, ensure_ascii=False), encoding="utf-8")
        self._matrix = vecs

    @property
    def matrix(self):
        if getattr(self, "_matrix", None) is None:
            self.ensure()
        return self._matrix

    # ---- search ----

    def _faiss_search(self, qvec: np.ndarray, top_k: int):
        import faiss
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.matrix.shape[1])
            self._index.add(self.matrix)
        scores, ids = self._index.search(qvec[None, :], top_k)
        return scores[0].tolist(), ids[0].tolist()

    def _numpy_search(self, qvec: np.ndarray, top_k: int):
        sims = self.matrix @ qvec
        ids = np.argsort(-sims)[:top_k]
        return sims[ids].tolist(), ids.tolist()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        qvec = self.embedder.embed_query(query)
        fn = self._faiss_search if faiss is not None else self._numpy_search
        scores, ids = fn(qvec, min(top_k, len(self.entries)))
        out = []
        for s, i in zip(scores, ids):
            if i < 0:
                continue
            e = self.entries[int(i)]
            out.append({"id": e["id"], "title": e["title"], "abstract": e["abstract"],
                        "keywords": e["keywords"], "category": e["category"],
                        "score": round(float(s), 4)})
        return out

    def stats(self) -> dict:
        by_cat: dict = {}
        for e in self.entries:
            by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
        return {"entries": len(self.entries), "categories": by_cat,
                "embedding": self.provider}
