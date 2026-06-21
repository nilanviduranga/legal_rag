import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
CLIENT_BASE_URL = os.getenv("CLIENT_BASE_URL", "https://ludexora.live").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")
AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

CANDIDATE_K = 20
TOP_K = 3
CASELAW_CANDIDATE_K = 15
CASELAW_TOP_K = 3
SUMMARIZE_THRESHOLD = 5

INDEX_DIR = "faiss_index"

# Statute index
INDEX_PATH = f"{INDEX_DIR}/legal.index"
CHUNKS_PATH = f"{INDEX_DIR}/chunks.pkl"
BM25_CORPUS_PATH = f"{INDEX_DIR}/bm25_corpus.pkl"

# Case law index (separate store, separate search)
CASELAW_INDEX_PATH = f"{INDEX_DIR}/caselaw.index"
CASELAW_CHUNKS_PATH = f"{INDEX_DIR}/caselaw_chunks.pkl"
CASELAW_BM25_PATH = f"{INDEX_DIR}/caselaw_bm25.pkl"
