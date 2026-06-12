import os
import requests
import pickle
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost/").rstrip("/")


def get_leaf_nodes():
    response = requests.get(f"{API_BASE_URL}/api/v1/nodes/leaf")
    response.raise_for_status()
    return response.json()


def get_law_path(node_id):
    response = requests.get(f"{API_BASE_URL}/api/v1/nodes/{node_id}/law-path")
    response.raise_for_status()
    return response.json()


def build_path_label(ancestors):
    return " > ".join(a["label"] for a in ancestors)


splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

leaf_nodes = get_leaf_nodes()
print(f"Found {len(leaf_nodes)} leaf nodes")

all_chunks = []

for node in leaf_nodes:
    node_id = node["node_id"]
    print(f"Processing node {node_id}...")

    law_data = get_law_path(node_id)
    law_text = law_data["law_text"]
    # path_label = build_path_label(law_data["ancestors"])
    # header = f"[Node ID: {node_id} | {path_label}]"
    header = f"[Node ID: {node_id} ]"


    raw_chunks = splitter.split_text(law_text)

    for chunk in raw_chunks:
        all_chunks.append({
            "text": f"{header}\n\n{chunk}",
            "node_id": node_id,
            # "path": path_label,
        })

print(f"Total chunks: {len(all_chunks)}")

texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/legal.index")

with open("faiss_index/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("Index saved successfully")
