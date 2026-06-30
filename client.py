import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import API_BASE_URL, CLIENT_BASE_URL, API_TOKEN, AUTH_HEADERS
from llm import generate_summary

_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=Retry(total=2, backoff_factor=0.3, status_forcelist=[502, 503, 504]),
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_session.headers.update(AUTH_HEADERS)


def check_api_token() -> None:
    if not API_TOKEN:
        raise RuntimeError("API_TOKEN is not set in .env")
    try:
        resp = _session.get(f"{API_BASE_URL}/api/v1/nodes/leaf", timeout=10)
        if resp.status_code == 401:
            raise RuntimeError(f"API_TOKEN is invalid — got 401 from {API_BASE_URL}")
        resp.raise_for_status()
        print(f"API token verified against {API_BASE_URL}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not reach API to verify token: {e}")


def fetch_full_law(node_id: str) -> str:
    try:
        resp = _session.get(
            f"{API_BASE_URL}/api/v1/nodes/{node_id}/law-path",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("law_text", "")
    except Exception:
        return ""


def fetch_law_with_context(node_id: str) -> dict:
    """Fetch a node's law_text plus every cross-referenced node's law_text."""
    try:
        resp = _session.get(
            f"{API_BASE_URL}/api/v1/nodes/{node_id}/law-context",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def create_session(payload: dict) -> dict:
    try:
        resp = _session.post(
            f"{CLIENT_BASE_URL}/api/chat/sessions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warn] create_session: {e}")
        return {}


def update_session_title(session_id: str, title: str) -> dict:
    try:
        resp = _session.patch(
            f"{CLIENT_BASE_URL}/api/chat/sessions/{session_id}/title",
            json={"title": title},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warn] update_session_title {session_id}: {e}")
        return {}


def store_chat(session_id: str, user_message: str, ai_response: str) -> None:
    try:
        resp = _session.post(
            f"{CLIENT_BASE_URL}/api/chat/sessions/{session_id}/chats",
            json={"chat_session_id": session_id, "user_message": user_message, "ai_response": ai_response},
            timeout=(5, 10),
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[warn] store_chat {session_id}: {e}")


def get_chats(session_id: str) -> list:
    try:
        resp = _session.get(
            f"{CLIENT_BASE_URL}/api/chat/sessions/{session_id}/chats",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("chats", [])
    except Exception as e:
        print(f"[warn] get_chats {session_id}: {e}")
        return []


def clear_chats(session_id: str) -> None:
    try:
        resp = _session.delete(
            f"{CLIENT_BASE_URL}/api/chat/sessions/{session_id}/chats",
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[warn] clear_chats {session_id}: {e}")


def fetch_chat_count(session_id: str):
    try:
        resp = _session.get(
            f"{CLIENT_BASE_URL}/api/chat/sessions/{session_id}/count",
            timeout=(5, 10),
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("count", data)
    except Exception as e:
        print(f"[warn] fetch_chat_count {session_id}: {e}")
        return None


def get_history(user_id: str) -> list:
    try:
        resp = _session.get(
            f"{CLIENT_BASE_URL}/api/chat/history/{user_id}",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("sessions", [])
    except Exception as e:
        print(f"[warn] get_history {user_id}: {e}")
        return []


def fetch_session_summary(session_id: str):
    try:
        resp = _session.get(
            f"{CLIENT_BASE_URL}/api/chat/sessions/{session_id}/summary",
            timeout=30,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get("summary") if isinstance(data, dict) else None
    except Exception as e:
        print(f"[warn] fetch_session_summary {session_id}: {e}")
        return None


def fetch_unsummarized_chats(session_id: str) -> list:
    try:
        resp = _session.get(
            f"{CLIENT_BASE_URL}/api/chat/sessions/{session_id}/chats",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("chats", [])
    except Exception as e:
        print(f"[warn] fetch_unsummarized_chats {session_id}: {e}")
        return []


def update_session_summary(session_id: str, summary: str) -> None:
    try:
        resp = _session.patch(
            f"{CLIENT_BASE_URL}/api/chat/sessions/{session_id}/summary",
            json={"summary": summary},
            timeout=(5, 10),
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[warn] patch_session_summary {session_id}: {e}")


def mark_chats_summarized(chat_ids: list) -> None:
    if not chat_ids:
        return
    try:
        resp = _session.patch(
            f"{CLIENT_BASE_URL}/api/chat/mark-summarized",
            json={"chat_ids": chat_ids},
            timeout=(5, 10),
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[warn] mark_chats_summarized {chat_ids}: {e}")


# ── Legal structure API (structural intent queries) ──────────────────────────

def search_acts_by_title(query: str) -> list[dict]:
    """
    Search legal_admin for acts whose title or short_title matches query.
    Returns up to 5 matches ordered by relevance.
    """
    try:
        resp = _session.get(
            f"{API_BASE_URL}/api/v1/acts/search",
            params={"q": query},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warn] search_acts_by_title '{query}': {e}")
        return []


def fetch_act_structure_stats(act_id: int) -> dict:
    """
    Fetch precomputed structure statistics (counts by node type) for an act.
    Returns the full stats dict from legal_admin, or {} on failure.
    """
    try:
        resp = _session.get(
            f"{API_BASE_URL}/api/v1/acts/{act_id}/structure-stats",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warn] fetch_act_structure_stats act_id={act_id}: {e}")
        return {}


def fetch_act_nodes_by_type(act_id: int, node_type: str) -> dict:
    """
    Fetch an ordered list of nodes of the given type for an act.
    Returns the full response dict with 'nodes' and 'count', or {} on failure.
    """
    try:
        resp = _session.get(
            f"{API_BASE_URL}/api/v1/acts/{act_id}/nodes-by-type",
            params={"node_type": node_type},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warn] fetch_act_nodes_by_type act_id={act_id} type={node_type}: {e}")
        return {}


def refresh_act_statistics(act_id: int) -> dict:
    """Force-recompute and cache structure statistics for the act in legal_admin."""
    try:
        resp = _session.post(
            f"{API_BASE_URL}/api/v1/acts/{act_id}/statistics/refresh",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warn] refresh_act_statistics act_id={act_id}: {e}")
        return {}


def run_summarize_job(session_id: str) -> None:
    print(f"[summarize] Starting for session {session_id}")
    old_summary = fetch_session_summary(session_id)
    chats = fetch_unsummarized_chats(session_id)
    if not chats:
        return
    chat_ids = [c.get("id") for c in chats if c.get("id")]
    new_summary = generate_summary(old_summary, chats)
    update_session_summary(session_id, new_summary)
    mark_chats_summarized(chat_ids)
    print(f"[summarize] Done for session {session_id} — {len(chats)} chat(s)")
