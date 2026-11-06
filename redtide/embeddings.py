"""Embedding layer: turns text into vectors for FAISS retrieval.

Two backends:
  local  BAAI/bge-small-zh-v1.5 (small local model — free, offline, reproducible; default)
  zhipu  GLM embedding-3 (API, 2048-dim)

Four points worth knowing:
1. Queries and documents must be embedded by the SAME model, or the two vector
   spaces are not comparable;
2. BGE retrieval models expect an instruction prefix on the query side
   (instruction-tuned retrieval);
3. Vectors are L2-normalized so inner product equals cosine similarity;
4. Embeddings are cached on disk, so repeated evaluation runs recompute
   (and re-bill) nothing.
"""
import hashlib
import json
import threading

import numpy as np

from .config import KB_DIR

# Official BGE-zh query-side instruction (the retrieval model is Chinese)
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    def __init__(self, provider: str = "local"):
        self.provider = provider
        self._model = None
        self._cache = self._load_cache()
        self._lock = threading.Lock()

    # ---- backends ----

    def _load_model(self):
        if self._model is not None:
            return
        if self.provider == "local":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        elif self.provider == "zhipu":
            import os
            from zhipuai import ZhipuAI
            self._model = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def _embed_raw(self, texts: list[str], is_query: bool) -> np.ndarray:
        self._load_model()
        if self.provider == "local":
            if is_query:
                texts = [BGE_QUERY_PREFIX + t for t in texts]
            vecs = self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        # zhipu embedding-3: the API takes at most 64 inputs per call — batch
        out = []
        for i in range(0, len(texts), 64):
            resp = self._model.embeddings.create(model="embedding-3",
                                                 input=texts[i:i + 64])
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        return arr

    # ---- cache ----

    def _cache_path(self):
        return KB_DIR / f"emb_cache_{self.provider}.json"

    def _load_cache(self) -> dict:
        p = self._cache_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self):
        KB_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path(), "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    # ---- public API ----

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed knowledge-base entries in batch (cache-backed)."""
        todo = [(i, t) for i, t in enumerate(texts) if self._key(t) not in self._cache]
        with self._lock:
            if todo:
                vecs = self._embed_raw([t for _, t in todo], is_query=False)
                for (i, t), v in zip(todo, vecs):
                    self._cache[self._key(t)] = v.tolist()
                self._save_cache()
        return np.asarray([self._cache[self._key(t)] for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query (query-side instruction prefix; uncached —
        queries almost never repeat)."""
        return self._embed_raw([text], is_query=True)[0]

    @property
    def dim(self) -> int:
        return int(self.embed_query("test").shape[0])
