import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from config import (
    API_BASE_URL, AUTH_HEADERS, SUMMARIZE_THRESHOLD,
    INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH,
    CASELAW_INDEX_PATH, CASELAW_CHUNKS_PATH, CASELAW_BM25_PATH,
)
from search import (
    embedder, store, caselaw_store,
    load_store, load_caselaw_store, require_store, hybrid_search,
    chunk_node_id, chunk_case_law_id,
)
from client import (
    check_api_token,
    fetch_full_law,
    fetch_law_with_context,
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
from ingest_caselaw import build_caselaw_index

API_TOKEN = os.getenv("API_TOKEN", "")
AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

check_api_token()

app = FastAPI()

if os.path.exists(INDEX_PATH):
    load_store()

load_caselaw_store()


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


def _resolve_laws(matched: list) -> list:
    """
    Resolve matched chunks to full context objects.
    Statute chunks: fetch law_text + cross-references from legal_admin.
    Case law chunks: return the chunk dict directly (already has all context needed).
    Returns a mixed list of str (statute) and dict (caselaw) for llm.py formatting.
    """
    statute_ids = []
    caselaw_chunks = []
    seen = set()

    for c in matched:
        if isinstance(c, dict) and c.get("source") == "caselaw":
            key = f"cl_{c.get('case_law_id')}_{c.get('section_type')}"
            if key not in seen:
                seen.add(key)
                caselaw_chunks.append(c)
        else:
            nid = chunk_node_id(c)
            if nid and nid not in seen:
                seen.add(nid)
                statute_ids.append(nid)

    results = []

    # Fetch statute law texts in parallel
    if statute_ids:
        futures = {_law_executor.submit(fetch_law_with_context, nid): nid for nid in statute_ids}
        primary_laws = []
        ref_laws = []
        seen_refs = set(seen)

        for future in as_completed(futures):
            ctx = future.result()
            if ctx.get("law_text"):
                primary_laws.append(ctx["law_text"])
            for ref in ctx.get("cross_references", []):
                ref_id = str(ref.get("node_id", ""))
                if ref_id and ref_id not in seen_refs and ref.get("law_text"):
                    seen_refs.add(ref_id)
                    ref_laws.append(ref["law_text"])

        results.extend(primary_laws + ref_laws)

    # Case law chunks come after statutes so LLM sees statutes first
    results.extend(caselaw_chunks)

    return results


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
    f_chats   = _io_executor.submit(fetch_unsummarized_chats, session_id)
    matched   = hybrid_search(question.question)
    summary   = f_summary.result()
    recent_chats = f_chats.result()

    full_laws = _resolve_laws(matched)
    answer    = generate_answer(question.question, full_laws=full_laws, summary=summary, recent_chats=recent_chats)

    store_chat(session_id, question.question, answer)

    count = fetch_chat_count(session_id)
    if count is not None and count >= SUMMARIZE_THRESHOLD:
        background_tasks.add_task(run_summarize_job, session_id)

    # Serialize caselaw dicts for the response so callers get structured data
    serialized_laws = []
    for item in full_laws:
        if isinstance(item, dict):
            serialized_laws.append({
                "source":       "caselaw",
                "case_name":    item.get("case_name"),
                "citation":     item.get("citation"),
                "section_type": item.get("section_type"),
                "text":         item.get("text", ""),
            })
        else:
            serialized_laws.append({"source": "statute", "text": item})

    return {"answer": answer, "full_laws": serialized_laws}


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
    """Rebuild both statute and case law FAISS indexes from scratch."""
    for path in [INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH,
                 CASELAW_INDEX_PATH, CASELAW_CHUNKS_PATH, CASELAW_BM25_PATH]:
        if os.path.exists(path):
            os.remove(path)
    store.clear()
    caselaw_store.clear()

    statute_result  = build_index(API_BASE_URL, embedder, auth_headers=AUTH_HEADERS)
    caselaw_result  = build_caselaw_index(API_BASE_URL, embedder, auth_headers=AUTH_HEADERS)

    load_store()
    load_caselaw_store()

    return {
        "status": "rebuilt",
        "statute_chunks":  statute_result.get("chunks_indexed", 0),
        "caselaw_chunks":  caselaw_result.get("chunks_indexed", 0),
    }


@app.post("/vectordb/rebuild-caselaws")
def rebuild_caselaw_vectordb():
    """Rebuild only the case law index without touching statute index."""
    for path in [CASELAW_INDEX_PATH, CASELAW_CHUNKS_PATH, CASELAW_BM25_PATH]:
        if os.path.exists(path):
            os.remove(path)
    caselaw_store.clear()

    result = build_caselaw_index(API_BASE_URL, embedder, auth_headers=AUTH_HEADERS)
    load_caselaw_store()

    return {"status": "rebuilt", "caselaw_chunks": result.get("chunks_indexed", 0)}
