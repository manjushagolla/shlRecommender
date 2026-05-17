"""
app/retriever.py
================
Singleton FAISS retriever. Loaded once at startup.
Agent calls retriever.retrieve() for every recommendation.
"""

import json, logging, os
from typing import Optional
import faiss
import numpy as np

log = logging.getLogger(__name__)

ROOT       = os.path.join(os.path.dirname(__file__), "..")
INDEX_PATH = os.path.join(ROOT, "catalog", "shl_index.faiss")
META_PATH  = os.path.join(ROOT, "catalog", "shl_meta.json")
MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Keywords that signal each test type — used for auto-filter inference
TEST_TYPE_SIGNALS = {
    "A": ["ability","aptitude","reasoning","numerical","verbal","inductive",
          "deductive","cognitive","spatial","mechanical","electrical","iq"],
    "B": ["situational","sjt","judgement","judgment","biodata","biographical","scenario"],
    "C": ["competenc","ucf","360","framework","behavioral interview"],
    "D": ["development","coaching","feedback","360"],
    "E": ["exercise","assessment centre","in-tray","role play","inbox"],
    "K": ["knowledge","skills","technical","java","python","sql","excel","javascript",
          "c#","c++","spring","react","angular","node","devops","aws","cloud",
          "cybersecurity","data science","machine learning","finance","accounting",
          "project management","html","css","seo","testing","mongodb","php","ruby"],
    "P": ["personality","opq","behaviour","behavior","motivation","motive","trait",
          "values","culture","character","hogan","preference","attitude"],
    "S": ["simulation","realistic","coding sim","automata","call center sim",
          "cashier","data entry","typing","interactive"],
}

JOB_LEVEL_SIGNALS = {
    "Entry-Level":                        ["entry","junior","fresh","intern","trainee","0-2 year"],
    "Graduate":                           ["graduate","grad","campus","early career","new grad"],
    "Supervisor":                         ["supervisor","team lead","lead"],
    "Front Line Manager":                 ["front line","frontline","floor manager"],
    "Manager":                            ["manager","management"],
    "Mid-Professional":                   ["mid","experienced","3 year","4 year","5 year"],
    "Professional Individual Contributor":["senior","specialist","expert","principal","ic"],
    "Director":                           ["director","vp","vice president","head of"],
    "Executive":                          ["executive","cxo","ceo","cto","cfo","c-level","c suite"],
    "General Population":                 ["general","all levels","any level","volume"],
}


class Retriever:
    def __init__(self):
        self._index: Optional[faiss.Index] = None
        self._meta:  list[dict] = []
        self._model = None

    # ── startup ──────────────────────────────────────────────────────────────
    def load(self):
        """Call once at app startup (FastAPI lifespan)."""
        if self._index is not None:
            return
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index missing: {INDEX_PATH}\n"
                "Run: python scripts/build_index.py"
            )
        self._index = faiss.read_index(INDEX_PATH)
        log.info("FAISS index: %d vectors, dim=%d", self._index.ntotal, self._index.d)

        with open(META_PATH, encoding="utf-8") as f:
            self._meta = json.load(f)
        log.info("Metadata: %d items", len(self._meta))

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(MODEL_NAME)
        log.info("Retriever ready ✅")

    # ── main search ──────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_test_types: Optional[list[str]] = None,
        filter_job_levels: Optional[list[str]]  = None,
        exclude_names: Optional[list[str]]       = None,
    ) -> list[dict]:
        """
        Semantic search with optional hard filters + soft boosts.

        Steps:
          1. Embed query
          2. FAISS search (over-fetch 5x)
          3. Hard filter by test type if specified
          4. Soft boost by job level overlap
          5. Sort, deduplicate, cap at top_k
        """
        assert self._index is not None, "Call retriever.load() first"

        # 1. embed
        vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # 2. over-fetch
        fetch_k  = min(self._index.ntotal, max(top_k * 6, 60))
        scores, indices = self._index.search(vec, fetch_k)

        exclude = set(exclude_names or [])
        candidates = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            item = self._meta[idx]
            if item["name"] in exclude:
                continue
            # hard filter
            if filter_test_types:
                if not any(t in filter_test_types for t in item.get("test_types", [])):
                    continue
            candidates.append({"item": item, "score": float(score)})

        # 3. soft job-level boost
        if filter_job_levels:
            wanted = set(filter_job_levels)
            for c in candidates:
                overlap = len(set(c["item"].get("job_levels", [])) & wanted)
                c["score"] += 0.04 * overlap

        # 4. sort, dedup, cap
        candidates.sort(key=lambda x: x["score"], reverse=True)
        seen, results = set(), []
        for c in candidates:
            name = c["item"]["name"]
            if name not in seen:
                seen.add(name)
                results.append(c["item"])
            if len(results) >= top_k:
                break

        return results

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_by_name(self, name: str) -> Optional[dict]:
        """Exact + fuzzy lookup by name (for compare queries)."""
        name_lower = name.lower()
        # exact
        for item in self._meta:
            if item["name"].lower() == name_lower:
                return item
        # substring
        for item in self._meta:
            if name_lower in item["name"].lower():
                return item
        return None

    def infer_test_types(self, text: str) -> list[str]:
        """Return test type letters that match signals in text."""
        t = text.lower()
        return [letter for letter, kws in TEST_TYPE_SIGNALS.items()
                if any(kw in t for kw in kws)]

    def infer_job_levels(self, text: str) -> list[str]:
        """Return job level names that match signals in text."""
        t = text.lower()
        return [level for level, kws in JOB_LEVEL_SIGNALS.items()
                if any(kw in t for kw in kws)]

    def all_names(self) -> list[str]:
        return [item["name"] for item in self._meta]


# global singleton
retriever = Retriever()