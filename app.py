import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from config import (
    API_BASE_URL, AUTH_HEADERS, SUMMARIZE_THRESHOLD,
    INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH,
    CASELAW_INDEX_PATH, CASELAW_CHUNKS_PATH, CASELAW_BM25_PATH,
    ACT_INDEX_PATH, ACT_RECORDS_PATH,
)
from search import (
    embedder, store, caselaw_store, act_store,
    load_store, load_caselaw_store, load_act_store, require_store,
    build_act_store, hybrid_search, chunk_node_id,
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
    refresh_act_statistics,
)
from llm import generate_answer, generate_discovery_answer
from intent import classify_intent, is_discovery_intent, is_structural_intent
from prompt_enhancer import is_harmful_query, needs_history, get_refusal_response
from discovery import run_discovery
from legal_structure import run_structural_query, format_structural_answer
from ingest import build_index
from ingest_caselaw import build_caselaw_index

API_TOKEN = os.getenv("API_TOKEN", "")
AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

check_api_token()

app = FastAPI()

if os.path.exists(INDEX_PATH):
    load_store()

load_caselaw_store()
load_act_store()


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


def _post_answer(session_id: str, user_message: str, answer: str) -> None:
    """Background: persist chat then trigger summarisation if threshold reached."""
    store_chat(session_id, user_message, answer)
    count = fetch_chat_count(session_id)
    if count is not None and count >= SUMMARIZE_THRESHOLD:
        run_summarize_job(session_id)


@app.post("/sessions/{session_id}/ask")
def session_ask(session_id: str, question: Question, background_tasks: BackgroundTasks):
    require_store()

    if is_harmful_query(question.question):
        refusal = get_refusal_response()
        background_tasks.add_task(_post_answer, session_id, question.question, refusal)
        return {"answer": refusal, "intent": "harmful", "full_laws": []}

    intent = classify_intent(question.question)

    if is_structural_intent(intent):
        # ── Structural pipeline (count / list WITHIN a named Act) ───────────
        structural_result = run_structural_query(question.question, intent)
        answer = format_structural_answer(structural_result)

        background_tasks.add_task(_post_answer, session_id, question.question, answer)

        return {
            "answer":  answer,
            "intent":  intent,
            "full_laws": [
                {
                    "source":      "structural",
                    "type":        "STRUCTURAL_RESULT",
                    "act_id":      structural_result.act_id,
                    "act":         structural_result.act_title,
                    "short_title": structural_result.short_title,
                    "node_type":   structural_result.node_type,
                    "count":       structural_result.count,
                    "message":     structural_result.message,
                }
            ] if structural_result.found else [],
        }

    if is_discovery_intent(intent) and act_store:
        # ── Discovery pipeline (count / list) ──────────────────────────────
        discovery_result = run_discovery(question.question, intent)
        try:
            answer = generate_discovery_answer(question.question, discovery_result)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

        background_tasks.add_task(_post_answer, session_id, question.question, answer)

        serialized_acts = [
            {
                "source":      "act_discovery",
                "act_id":      a.act_id,
                "title":       a.title,
                "short_title": a.short_title,
                "summary":     a.summary,
                "score":       round(a.score, 4),
            }
            for a in discovery_result.acts
        ]
        return {
            "answer":    answer,
            "intent":    intent,
            "full_laws": serialized_acts,
        }

    # ── Standard explanation pipeline ──────────────────────────────────────
    use_history = needs_history(question.question)

    f_summary = _io_executor.submit(fetch_session_summary, session_id) if use_history else None
    f_chats   = _io_executor.submit(fetch_unsummarized_chats, session_id) if use_history else None
    matched   = hybrid_search(question.question)
    summary      = f_summary.result() if f_summary else None
    recent_chats = f_chats.result()   if f_chats   else None

    full_laws = _resolve_laws(matched)
    try:
        answer = generate_answer(question.question, full_laws=full_laws, summary=summary, recent_chats=recent_chats)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    background_tasks.add_task(_post_answer, session_id, question.question, answer)

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

    return {"answer": answer, "intent": intent, "full_laws": serialized_laws}


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

@app.post("/vectordb/rebuild-statutes")
def rebuild_statutes():
    """Rebuild only the statute FAISS index. Does not touch case laws or act metadata."""
    for path in [INDEX_PATH, CHUNKS_PATH, BM25_CORPUS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    store.clear()

    result = build_index(API_BASE_URL, embedder, auth_headers=AUTH_HEADERS, build_act_metadata=False)
    load_store()

    return {"status": "rebuilt", "statute_chunks": result.get("chunks_indexed", 0)}


@app.post("/vectordb/rebuild-caselaws")
def rebuild_caselaw_vectordb():
    """Rebuild only the case law index without touching statute or act-metadata indexes."""
    for path in [CASELAW_INDEX_PATH, CASELAW_CHUNKS_PATH, CASELAW_BM25_PATH]:
        if os.path.exists(path):
            os.remove(path)
    caselaw_store.clear()

    result = build_caselaw_index(API_BASE_URL, embedder, auth_headers=AUTH_HEADERS)
    load_caselaw_store()

    return {"status": "rebuilt", "caselaw_chunks": result.get("chunks_indexed", 0)}


@app.post("/vectordb/warm-structure-cache")
def warm_structure_cache():
    """
    Trigger legal_admin to (re)compute act_statistics for every active act.
    Call this after a vectordb rebuild to pre-warm the structural query cache.
    """
    from client import search_acts_by_title
    import requests as _req

    # Fetch all active acts from legal_admin then refresh each
    try:
        from config import API_BASE_URL, AUTH_HEADERS as _hdrs
        resp = _req.get(f"{API_BASE_URL}/api/v1/acts", headers=_hdrs, timeout=30)
        resp.raise_for_status()
        acts = resp.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    refreshed = 0
    for act in acts:
        result = refresh_act_statistics(act["act_id"])
        if result:
            refreshed += 1

    return {"status": "warmed", "acts_refreshed": refreshed}


@app.post("/vectordb/rebuild-act-metadata")
def rebuild_act_metadata():
    """
    Regenerate AI metadata for all acts and rebuild only the act-level FAISS index.
    Does not touch the statute or case-law indexes.
    """
    for path in [ACT_INDEX_PATH, ACT_RECORDS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    act_store.clear()

    from metadata_gen import generate_and_store_all, fetch_all_metadata

    count   = generate_and_store_all(skip_existing=False)
    records = fetch_all_metadata()
    build_act_store(records)
    load_act_store()

    return {
        "status":          "rebuilt",
        "acts_generated":  count,
        "acts_indexed":    len(act_store.get("records", [])),
    }
