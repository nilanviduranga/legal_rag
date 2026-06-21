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
        data = resp.json()
        return data.get("content") or data.get("text") or data.get("law") or str(data)
    except Exception:
        return ""


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
            timeout=30,
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
            timeout=30,
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
            timeout=15,
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
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[warn] mark_chats_summarized {chat_ids}: {e}")


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
