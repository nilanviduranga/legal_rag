import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from config import API_BASE_URL, AUTH_HEADERS, SUMMARIZE_THRESHOLD, INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH
from search import embedder, store, load_store, require_store, hybrid_search, chunk_node_id
from client import (
    check_api_token,
    fetch_full_law,
    create_session,
    update_session_title,
    store_chat,
    get_chats,
    clear_chats,
    fetch_chat_count,
    fetch_session_summary,
    fetch_unsummarized_chats,
    update_session_summary,
    mark_chats_summarized,
    get_history,
    run_summarize_job,
)
from llm import generate_answer
from ingest import build_index
API_TOKEN = os.getenv("API_TOKEN", "")

AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

check_api_token()

app = FastAPI()

if os.path.exists(INDEX_PATH):
    load_store()


class Question(BaseModel):
    question: str


class SessionCreate(BaseModel):
    user_id: str
    title: Optional[str] = None


class TitleUpdate(BaseModel):
    title: str


class ChatCreate(BaseModel):
    user_message: str
    ai_response: str


class SummaryUpdate(BaseModel):
    summary: str


class MarkSummarized(BaseModel):
    chat_ids: list[int]


_law_executor = ThreadPoolExecutor(max_workers=6)


def _resolve_laws(matched: list) -> list[str]:
    seen = set()
    unique_ids = []
    for c in matched:
        nid = chunk_node_id(c)
        if nid and nid not in seen:
            seen.add(nid)
            unique_ids.append(nid)

    if not unique_ids:
        return []

    futures = {_law_executor.submit(fetch_full_law, nid): nid for nid in unique_ids}
    laws = []
    for future in as_completed(futures):
        law = future.result()
        if law:
            laws.append(law)
    return laws


# ── Session management ──────────────────────────────────────────────────────

@app.post("/sessions")
def create_session_endpoint(body: SessionCreate):
    result = create_session(body.model_dump(exclude_none=True))
    if not result:
        raise HTTPException(status_code=502, detail="Failed to create session")
    return result


@app.patch("/sessions/{session_id}/title")
def update_title(session_id: str, body: TitleUpdate):
    result = update_session_title(session_id, body.title)
    return result


# ── AI ask ──────────────────────────────────────────────────────────────────

_io_executor = ThreadPoolExecutor(max_workers=4)


@app.post("/sessions/{session_id}/ask")
def session_ask(session_id: str, question: Question, background_tasks: BackgroundTasks):
    require_store()

    f_summary = _io_executor.submit(fetch_session_summary, session_id)
    f_chats = _io_executor.submit(fetch_unsummarized_chats, session_id)
    matched = hybrid_search(question.question)
    summary = f_summary.result()
    recent_chats = f_chats.result()

    full_laws = _resolve_laws(matched)
    answer = generate_answer(question.question, full_laws=full_laws, summary=summary, recent_chats=recent_chats)

    store_chat(session_id, question.question, answer)

    count = fetch_chat_count(session_id)
    if count is not None and count >= SUMMARIZE_THRESHOLD:
        background_tasks.add_task(run_summarize_job, session_id)

    return {"answer": answer, "full_laws": full_laws}


# ── Chat CRUD ───────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/chats")
def create_chat(session_id: str, body: ChatCreate):
    store_chat(session_id, body.user_message, body.ai_response)
    return {"status": "created"}


@app.get("/sessions/{session_id}/chats")
def list_chats(session_id: str):
    chats = get_chats(session_id)
    return {"session_id": session_id, "chats": chats}


@app.delete("/sessions/{session_id}/chats")
def delete_chats(session_id: str):
    clear_chats(session_id)
    return {"status": "cleared"}


@app.get("/sessions/{session_id}/count")
def session_chat_count(session_id: str):
    count = fetch_chat_count(session_id)
    if count is None:
        raise HTTPException(status_code=502, detail="Failed to fetch chat count")
    return {"session_id": session_id, "count": count}


# ── Summary ─────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/summary")
def get_summary(session_id: str):
    summary = fetch_session_summary(session_id)
    return {"session_id": session_id, "summary": summary}


@app.patch("/sessions/{session_id}/summary")
def patch_summary(session_id: str, body: SummaryUpdate):
    update_session_summary(session_id, body.summary)
    return {"status": "updated"}


# ── Mark summarized ─────────────────────────────────────────────────────────

@app.patch("/mark-summarized")
def mark_summarized(body: MarkSummarized):
    mark_chats_summarized(body.chat_ids)
    return {"status": "marked"}


# ── History ─────────────────────────────────────────────────────────────────

@app.get("/history/{user_id}")
def user_history(user_id: str):
    sessions = get_history(user_id)
    return {"user_id": user_id, "sessions": sessions}


# ── Admin ────────────────────────────────────────────────────────────────────

@app.post("/vectordb/rebuild")
def rebuild_vectordb():
    for path in [INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    store.clear()
    result = build_index(API_BASE_URL, embedder, auth_headers=AUTH_HEADERS)
    load_store()
    return {"status": "rebuilt", **result}
