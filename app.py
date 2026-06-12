from fastapi import FastAPI
from pydantic import BaseModel
from llm import generate_answer

import faiss
import pickle
import numpy as np
import os
import requests

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")

# Load model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Load index
index = faiss.read_index("faiss_index/legal.index")

# Load chunks
with open("faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


class Question(BaseModel):
    question: str


#@app.post("/search")
#def search(question: Question):

#    query_embedding = model.encode([question.question])

#    distances, indices = index.search(
#        np.array(query_embedding),
#        k=4
#    )

#    results = []

#    for idx in indices[0]:
#        results.append(chunks[idx])

#    return {
#        "question": question.question,
#        "results": results
#    }


def fetch_full_law(node_id: str) -> str:
    try:
        url = f"{API_BASE_URL}/api/v1/nodes/{node_id}/law-path"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # extract text from common response field names
        return data.get("content") or data.get("text") or data.get("law") or str(data)
    except Exception:
        return ""


@app.post("/ask")
def ask(question: Question):

    query_embedding = model.encode([question.question])

    distances, indices = index.search(
        np.array(query_embedding),
        k=3
    )

    matched = [chunks[i] for i in indices[0]]

    # support both old (str) and new (dict) chunk formats
    # def chunk_text(c):
    #     return c["text"] if isinstance(c, dict) else c

    def chunk_node_id(c):
        return c.get("node_id") if isinstance(c, dict) else None


    # # fetch full law for each matched chunk via its node_id
    full_laws = []
    for c in matched:
        node_id = chunk_node_id(c)
        if node_id:
            law_text = fetch_full_law(node_id)
            if law_text:
                full_laws.append(law_text)

    answer = generate_answer(question.question, full_laws=full_laws)

    # sources = [
    #     {"node_id": c["node_id"], "path": c["path"], "text": c["text"]}
    #     if isinstance(c, dict) else c
    #     for c in matched
    # ]

    return {
        # "question": question.question,
        "answer": answer,
        # "sources": sources,
        "full_laws": full_laws,
    }