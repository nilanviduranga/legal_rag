from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm import generate_answer
from ingest_api import build_index, INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH

import faiss
import pickle
import numpy as np
import os
import requests

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")

AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


def check_api_token() -> None:
    """Verify the API_TOKEN is valid against the external API before starting."""
    if not API_TOKEN:
        raise RuntimeError("API_TOKEN is not set in .env")
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/v1/nodes/leaf",
            headers=AUTH_HEADERS,
            timeout=10,
        )
        if resp.status_code == 401:
            raise RuntimeError(f"API_TOKEN is invalid — got 401 from {API_BASE_URL}")
        resp.raise_for_status()
        print(f"API token verified successfully against {API_BASE_URL}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not reach API to verify token: {e}")


check_api_token()

CANDIDATE_K = 10
TOP_K = 3

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# Mutable store so indexes can be hot-reloaded without restarting the server
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


if os.path.exists(INDEX_PATH):
    load_store()


class Question(BaseModel):
    question: str


def chunk_text(c) -> str:
    return c["text"] if isinstance(c, dict) else c


def chunk_node_id(c):
    return c.get("node_id") if isinstance(c, dict) else None


def fetch_full_law(node_id: str) -> str:
    try:
        url = f"{API_BASE_URL}/api/v1/nodes/{node_id}/law-path"
        resp = requests.get(url, headers=AUTH_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("content") or data.get("text") or data.get("law") or str(data)
    except Exception:
        return ""


def hybrid_search(query: str) -> list:
    query_vec = embedder.encode([query])
    _, faiss_indices = store["index"].search(np.array(query_vec), CANDIDATE_K)
    semantic_hits = set(faiss_indices[0].tolist())

    tokens = query.lower().split()
    bm25_scores = store["bm25"].get_scores(tokens)
    bm25_top = np.argsort(bm25_scores)[::-1][:CANDIDATE_K].tolist()
    keyword_hits = set(bm25_top)

    candidate_indices = list(semantic_hits | keyword_hits)
    candidates = [store["chunks"][i] for i in candidate_indices]

    pairs = [(query, chunk_text(c)) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [c for _, c in ranked[:TOP_K]]


@app.post("/ask")
def ask(question: Question):
    require_store()
    matched = hybrid_search(question.question)

    full_laws = []
    for c in matched:
        node_id = chunk_node_id(c)
        if node_id:
            law_text = fetch_full_law(node_id)
            if law_text:
                full_laws.append(law_text)

    answer = generate_answer(question.question, full_laws=full_laws)

    return {
        "answer": answer,
        "full_laws": full_laws,
    }


@app.post("/vectordb/rebuild")
def rebuild_vectordb():
    for path in [INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    store.clear()
    result = build_index(API_BASE_URL, embedder, auth_headers=AUTH_HEADERS)
    load_store()
    return {"status": "rebuilt", **result}
