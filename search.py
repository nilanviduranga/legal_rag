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
    ACT_INDEX_PATH, ACT_RECORDS_PATH,
    CANDIDATE_K, TOP_K,
    CASELAW_CANDIDATE_K, CASELAW_TOP_K,
    ACT_CANDIDATE_K, ACT_SCORE_THRESHOLD, EVIDENCE_TOP_K,
)

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")

store: dict = {}
caselaw_store: dict = {}
act_store: dict = {}

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Load / save helpers ──────────────────────────────────────────────────────

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


def load_act_store() -> None:
    if not (os.path.exists(ACT_INDEX_PATH) and os.path.exists(ACT_RECORDS_PATH)):
        return
    act_store["index"] = faiss.read_index(ACT_INDEX_PATH)
    with open(ACT_RECORDS_PATH, "rb") as f:
        act_store["records"] = pickle.load(f)


def build_act_store(metadata_records: list[dict]) -> None:
    """
    Build and persist the act-level FAISS index from AI metadata records.
    Each record is embedded as: title + summary + keywords + regulated_activities.
    """
    if not metadata_records:
        return

    texts = []
    for rec in metadata_records:
        parts = [rec.get("title", "")]
        if rec.get("summary"):
            parts.append(rec["summary"])
        kws = rec.get("keywords") or []
        acts = rec.get("regulated_activities") or []
        domains = rec.get("legal_domains") or []
        subjects = rec.get("subjects") or []
        parts.append(" ".join(kws + acts + domains + subjects))
        texts.append(" ".join(filter(None, parts)))

    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, ACT_INDEX_PATH)
    with open(ACT_RECORDS_PATH, "wb") as f:
        pickle.dump(metadata_records, f)

    act_store["index"]   = index
    act_store["records"] = metadata_records
    print(f"[search] Act-level FAISS built with {len(metadata_records)} acts")


# ── Runtime checks ───────────────────────────────────────────────────────────

def require_store() -> None:
    if not store:
        raise HTTPException(
            status_code=503,
            detail="Vector DB not ready. Call POST /vectordb/rebuild first.",
        )


# ── Chunk helpers ────────────────────────────────────────────────────────────

def chunk_node_id(c):
    return c.get("node_id") if isinstance(c, dict) else None


def _chunk_text(c) -> str:
    return c["text"] if isinstance(c, dict) else c


# ── Internal candidate retrieval ─────────────────────────────────────────────

def _candidates(query: str, s: dict, k: int) -> list:
    query_vec = embedder.encode([_BGE_QUERY_PREFIX + query], normalize_embeddings=True)
    _, faiss_idx = s["index"].search(np.array(query_vec, dtype=np.float32), k)
    semantic_hits = set(faiss_idx[0].tolist())

    bm25_scores = s["bm25"].get_scores(query.lower().split())
    bm25_top    = set(np.argsort(bm25_scores)[::-1][:k].tolist())

    return [s["chunks"][i] for i in semantic_hits | bm25_top]


# ── Hybrid search (existing explanation pipeline) ────────────────────────────

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


# ── Act-level discovery search ───────────────────────────────────────────────

def act_discovery_search(query: str) -> list[dict]:
    """
    Semantic search over the act-level FAISS index.

    Returns a list of act records (dicts with act_id, title, summary, …)
    ordered by relevance. All acts above ACT_SCORE_THRESHOLD are returned,
    with a hard cap of ACT_CANDIDATE_K.
    """
    if not act_store:
        return []

    query_vec = embedder.encode(
        [_BGE_QUERY_PREFIX + query], normalize_embeddings=True
    )
    query_vec = np.array(query_vec, dtype=np.float32)

    k = min(ACT_CANDIDATE_K, len(act_store["records"]))
    scores, indices = act_store["index"].search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) < ACT_SCORE_THRESHOLD:
            break
        rec = dict(act_store["records"][idx])
        rec["_score"] = float(score)
        results.append(rec)

    return results


def act_chunks_for_act(act_id: int) -> list:
    """Return all statute chunks that belong to a specific act_id."""
    if not store:
        return []
    return [c for c in store["chunks"] if isinstance(c, dict) and c.get("act_id") == act_id]


def evidence_for_act(query: str, act_id: int) -> list:
    """
    Retrieve the top EVIDENCE_TOP_K most relevant chunks for a specific act,
    re-ranked by CrossEncoder against the query.
    """
    candidates = act_chunks_for_act(act_id)
    if not candidates:
        return []

    pairs  = [(query, _chunk_text(c)) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:EVIDENCE_TOP_K]]
