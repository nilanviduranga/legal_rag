import faiss
import pickle
import numpy as np
from fastapi import HTTPException
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH, CANDIDATE_K, TOP_K

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")

store: dict = {}


def load_store() -> None:
    store["index"] = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        store["chunks"] = pickle.load(f)
    with open(BM25_CORPUS_PATH, "rb") as f:
        store["bm25"] = BM25Okapi(pickle.load(f))


def require_store() -> None:
    if not store:
        raise HTTPException(
            status_code=503,
            detail="Vector DB not ready. Call POST /vectordb/rebuild first.",
        )


def chunk_node_id(c):
    return c.get("node_id") if isinstance(c, dict) else None


def _chunk_text(c) -> str:
    return c["text"] if isinstance(c, dict) else c


def hybrid_search(query: str) -> list:
    query_vec = embedder.encode([query])
    _, faiss_indices = store["index"].search(np.array(query_vec), CANDIDATE_K)
    semantic_hits = set(faiss_indices[0].tolist())

    bm25_scores = store["bm25"].get_scores(query.lower().split())
    bm25_top = set(np.argsort(bm25_scores)[::-1][:CANDIDATE_K].tolist())

    candidates = [store["chunks"][i] for i in semantic_hits | bm25_top]
    pairs = [(query, _chunk_text(c)) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [c for _, c in ranked[:TOP_K]]
