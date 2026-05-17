"""
scripts/build_index.py
======================
Embeds every assessment using sentence-transformers → saves FAISS index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --model all-MiniLM-L6-v2   # faster/smaller
    python scripts/build_index.py --model all-mpnet-base-v2  # better quality
"""

import argparse, json, os, sys, time, logging
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT         = os.path.join(os.path.dirname(__file__), "..")
CATALOG_PATH = os.path.join(ROOT, "catalog", "shl_catalog.json")
INDEX_PATH   = os.path.join(ROOT, "catalog", "shl_index.faiss")
META_PATH    = os.path.join(ROOT, "catalog", "shl_meta.json")

# all-MiniLM-L6-v2  → 384-dim, 80MB,  fast  ← good default
# all-mpnet-base-v2 → 768-dim, 420MB, better quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_catalog():
    if not os.path.exists(CATALOG_PATH):
        log.error("Catalog not found. Run scrape_catalog.py first.")
        sys.exit(1)
    with open(CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d catalog items", len(data))
    return data


def build_texts(data):
    texts = []
    for item in data:
        text = item.get("embedding_text") or (
            f"{item['name']}. "
            f"{item.get('description', '')}. "
            f"Keywords: {', '.join(item.get('keywords', []))}. "
            f"Job levels: {', '.join(item.get('job_levels', []))}."
        )
        texts.append(text.strip())
    return texts


def embed_texts(texts, model_name):
    from sentence_transformers import SentenceTransformer
    log.info("Loading model '%s' ...", model_name)
    model = SentenceTransformer(model_name)
    log.info("Embedding %d texts ...", len(texts))
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # enables cosine via inner product
    )
    log.info("Embedded in %.1fs  shape=%s", time.time() - t0, vectors.shape)
    return vectors.astype("float32")


def build_index(vectors):
    """
    IndexFlatIP = exact cosine search (vectors are normalized).
    Perfect for catalogs < 1000 items — no approximation error,
    no training step, instant results.
    """
    dim = vectors.shape[1]
    log.info("Building IndexFlatIP  dim=%d  n=%d", dim, len(vectors))
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    log.info("Index ready: %d vectors", index.ntotal)
    return index


def smoke_test(index, meta, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    queries = [
        "Java developer mid level backend",
        "personality assessment leadership manager",
        "numerical reasoning graduate finance",
        "customer service call center entry level",
        "data science machine learning Python",
        "cognitive ability screening volume hiring",
    ]
    log.info("── Smoke test ──────────────────────────────────────")
    for q in queries:
        vec = model.encode([q], normalize_embeddings=True).astype("float32")
        scores, idxs = index.search(vec, 3)
        results = [f"{meta[i]['name']} ({scores[0][j]:.3f})" for j, i in enumerate(idxs[0])]
        log.info("  Q: %-45s", q)
        log.info("     → %s", results[0])
    log.info("────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    data    = load_catalog()
    texts   = build_texts(data)
    vectors = embed_texts(texts, args.model)
    index   = build_index(vectors)

    # Lightweight metadata sidecar (one dict per item, no vectors)
    meta = [{
        "name":             item["name"],
        "url":              item["url"],
        "test_type":        item.get("test_type", ""),
        "test_types":       item.get("test_types", []),
        "description":      item.get("description", ""),
        "job_levels":       item.get("job_levels", []),
        "keywords":         item.get("keywords", []),
        "remote_testing":   item.get("remote_testing", False),
        "adaptive_irt":     item.get("adaptive_irt", False),
        "duration_minutes": item.get("duration_minutes", 0),
        "languages":        item.get("languages", []),
    } for item in data]

    faiss.write_index(index, INDEX_PATH)
    log.info("Saved FAISS index → %s", INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Saved metadata    → %s", META_PATH)

    if not args.skip_smoke:
        smoke_test(index, meta, args.model)

    print(f"\n✅  Index built successfully!")
    print(f"    Model   : {args.model}")
    print(f"    Items   : {index.ntotal}")
    print(f"    Index   : {INDEX_PATH}")
    print(f"    Metadata: {META_PATH}")
    print(f"\n    Next: build the retriever → python -m uvicorn app.main:app --reload")