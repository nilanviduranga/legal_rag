import os
import requests
import pickle
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = "faiss_index"
INDEX_PATH = f"{INDEX_DIR}/legal.index"
CHUNKS_PATH = f"{INDEX_DIR}/chunks.pkl"
BM25_CORPUS_PATH = f"{INDEX_DIR}/bm25_corpus.pkl"


def build_index(api_base_url: str, embedder: SentenceTransformer) -> dict:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    response = requests.get(f"{api_base_url}/api/v1/nodes/leaf")
    response.raise_for_status()
    leaf_nodes = response.json()
    print(f"Found {len(leaf_nodes)} leaf nodes")

    all_chunks = []
    for node in leaf_nodes:
        node_id = node["node_id"]
        print(f"Processing node {node_id}...")
        law_resp = requests.get(f"{api_base_url}/api/v1/nodes/{node_id}/law-path")
        law_resp.raise_for_status()
        law_text = law_resp.json()["law_text"]
        header = f"[Node ID: {node_id}]"
        for chunk in splitter.split_text(law_text):
            all_chunks.append({"text": f"{header}\n\n{chunk}", "node_id": node_id})

    print(f"Total chunks: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(faiss_index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    bm25_corpus = [c["text"].lower().split() for c in all_chunks]
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(bm25_corpus, f)

    print("Index saved successfully")
    return {"chunks_indexed": len(all_chunks)}


if __name__ == "__main__":
    api_base = os.getenv("API_BASE_URL", "http://localhost/").rstrip("/")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    build_index(api_base, model)
