import os
import pickle
import numpy as np
import faiss
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    INDEX_DIR,
    CASELAW_INDEX_PATH,
    CASELAW_CHUNKS_PATH,
    CASELAW_BM25_PATH,
    API_BASE_URL,
    AUTH_HEADERS,
)

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def build_caselaw_index(
    api_base_url: str,
    embedder: SentenceTransformer,
    auth_headers: dict | None = None,
) -> dict:
    headers = auth_headers or {}

    resp = requests.get(f"{api_base_url}/api/v1/case-laws", headers=headers)
    resp.raise_for_status()
    cases = resp.json()
    print(f"Found {len(cases)} active case laws")

    all_chunks: list[dict] = []

    for case in cases:
        header = f"[CASE LAW] {case['case_name']}"
        for chunk_text in _splitter.split_text(case["content"]):
            all_chunks.append({
                "text":        f"{header}\n\n{chunk_text}",
                "case_law_id": case["id"],
                "case_name":   case["case_name"],
                "source":      "caselaw",
            })

    print(f"Total caselaw chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks — caselaw index not written.")
        return {"chunks_indexed": 0}

    texts      = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(faiss_index, CASELAW_INDEX_PATH)

    with open(CASELAW_CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(CASELAW_BM25_PATH, "wb") as f:
        pickle.dump([c["text"].lower().split() for c in all_chunks], f)

    print("Caselaw index saved.")
    return {"chunks_indexed": len(all_chunks)}


if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    build_caselaw_index(API_BASE_URL, model, auth_headers=AUTH_HEADERS)
