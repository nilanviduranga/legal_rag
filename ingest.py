import os
import pickle
import numpy as np
import faiss
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH, API_BASE_URL, AUTH_HEADERS


def build_index(
    api_base_url: str,
    embedder: SentenceTransformer,
    auth_headers: dict | None = None,
    build_act_metadata: bool = True,
) -> dict:
    """
    Build the statute FAISS index from all leaf nodes.

    When build_act_metadata=True (default), also:
      - generate and store AI metadata for every act that lacks it
      - build the act-level FAISS index for discovery queries
    """
    headers = auth_headers or {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    resp = requests.get(f"{api_base_url}/api/v1/nodes/leaf", headers=headers)
    resp.raise_for_status()
    leaf_nodes = resp.json()
    print(f"Found {len(leaf_nodes)} leaf nodes")

    all_chunks = []
    for node in leaf_nodes:
        node_id = node["node_id"]
        act_id  = node.get("act_id")
        law_resp = requests.get(f"{api_base_url}/api/v1/nodes/{node_id}/law-path", headers=headers)
        law_resp.raise_for_status()
        law_text = law_resp.json()["law_text"]
        for chunk in splitter.split_text(law_text):
            all_chunks.append({
                "text":    f"[Node ID: {node_id}]\n\n{chunk}",
                "node_id": node_id,
                "act_id":  act_id,
            })

    print(f"Total chunks: {len(all_chunks)}")

    texts      = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(faiss_index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump([c["text"].lower().split() for c in all_chunks], f)

    print("Statute index saved")

    if build_act_metadata:
        _run_metadata_pipeline(embedder)

    return {"chunks_indexed": len(all_chunks)}


def _run_metadata_pipeline(embedder: SentenceTransformer) -> None:
    """Generate missing act metadata and rebuild the act-level FAISS index."""
    try:
        from metadata_gen import generate_and_store_all, fetch_all_metadata
        from search import build_act_store

        print("[ingest] Generating act-level AI metadata…")
        count = generate_and_store_all(skip_existing=True)
        print(f"[ingest] Metadata generated for {count} act(s)")

        print("[ingest] Building act-level FAISS index…")
        records = fetch_all_metadata()
        build_act_store(records)
        print(f"[ingest] Act FAISS built with {len(records)} record(s)")
    except Exception as e:
        print(f"[ingest] Act metadata pipeline failed (non-fatal): {e}")


if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    build_index(API_BASE_URL, model, auth_headers=AUTH_HEADERS)
