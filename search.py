import os
import faiss
import pickle
import numpy as np
from fastapi import HTTPException
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH,
    CASELAW_INDEX_PATH, CASELAW_CHUNKS_PATH, CASELAW_BM25_PATH,
    CANDIDATE_K, TOP_K,
    CASELAW_CANDIDATE_K, CASELAW_TOP_K,
)

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")

store: dict = {}
caselaw_store: dict = {}

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_store() -> None:
    store["index"] = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        store["chunks"] = pickle.load(f)
    with open(BM25_CORPUS_PATH, "rb") as f:
        store["bm25"] = BM25Okapi(pickle.load(f))


def load_caselaw_store() -> None:
    if not (
        os.path.exists(CASELAW_INDEX_PATH)
        and os.path.exists(CASELAW_CHUNKS_PATH)
        and os.path.exists(CASELAW_BM25_PATH)
    ):
        return
    caselaw_store["index"] = faiss.read_index(CASELAW_INDEX_PATH)
    with open(CASELAW_CHUNKS_PATH, "rb") as f:
        caselaw_store["chunks"] = pickle.load(f)
    with open(CASELAW_BM25_PATH, "rb") as f:
        caselaw_store["bm25"] = BM25Okapi(pickle.load(f))


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


def _candidates(query: str, s: dict, k: int) -> list:
    query_vec = embedder.encode([_BGE_QUERY_PREFIX + query], normalize_embeddings=True)
    _, faiss_idx = s["index"].search(np.array(query_vec, dtype=np.float32), k)
    semantic_hits = set(faiss_idx[0].tolist())

    bm25_scores = s["bm25"].get_scores(query.lower().split())
    bm25_top    = set(np.argsort(bm25_scores)[::-1][:k].tolist())

    return [s["chunks"][i] for i in semantic_hits | bm25_top]


def hybrid_search(query: str) -> list:
    statute_candidates = _candidates(query, store, CANDIDATE_K)
    caselaw_candidates = _candidates(query, caselaw_store, CASELAW_CANDIDATE_K) if caselaw_store else []

    all_candidates = statute_candidates + caselaw_candidates
    if not all_candidates:
        return []

    pairs  = [(query, _chunk_text(c)) for c in all_candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, all_candidates), key=lambda x: x[0], reverse=True)

    statute_hits  = [c for _, c in ranked if not (isinstance(c, dict) and c.get("source") == "caselaw")]
    caselaw_hits  = [c for _, c in ranked if isinstance(c, dict) and c.get("source") == "caselaw"]

    return statute_hits[:TOP_K] + caselaw_hits[:CASELAW_TOP_K]
