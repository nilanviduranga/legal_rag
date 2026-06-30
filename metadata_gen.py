"""
Act-level AI metadata generation.

Called once per Act during index build — never during user queries.

Workflow:
  1. Fetch acts list from legal_admin          → fetch_acts()
  2. Fetch a text sample for each act          → fetch_act_text_sample()
  3. Call LLM to produce structured metadata   → generate_act_metadata()
  4. POST the result back to legal_admin        → store_act_metadata()
  5. Fetch all stored metadata for FAISS build → fetch_all_metadata()
"""

import json

import requests

from config import API_BASE_URL, AUTH_HEADERS
from llm import llm_call

_SYSTEM = (
    "You are a legal metadata extractor.\n"
    "Given an Act title and text excerpt, output ONLY a valid JSON object:\n"
    "{\n"
    '  "summary": "One sentence describing what this Act regulates.",\n'
    '  "keywords": ["word1", "word2"],\n'
    '  "legal_domains": ["Commercial Law"],\n'
    '  "regulated_activities": ["selling", "buying"],\n'
    '  "subjects": ["consumers", "employers"]\n'
    "}\n"
    "No explanation. No markdown. Only the JSON object."
)


def fetch_acts() -> list[dict]:
    resp = requests.get(f"{API_BASE_URL}/api/v1/acts", headers=AUTH_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_act_text_sample(act_id: int) -> dict:
    resp = requests.get(
        f"{API_BASE_URL}/api/v1/acts/{act_id}/text-sample",
        headers=AUTH_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def generate_act_metadata(act_id: int, title: str, text_sample: str) -> dict:
    prompt = f"Act: {title}\n\nText excerpt:\n{text_sample[:3000]}"
    try:
        raw = llm_call(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        ).strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception as e:
        print(f"[metadata_gen] LLM failed for act {act_id} ({title}): {e}")
        return {
            "summary":              f"Regulates matters covered by {title}.",
            "keywords":             title.lower().split(),
            "legal_domains":        [],
            "regulated_activities": [],
            "subjects":             [],
        }


def store_act_metadata(act_id: int, metadata: dict) -> bool:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/acts/{act_id}/metadata",
            json=metadata,
            headers=AUTH_HEADERS,
            timeout=30,
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        print(f"[metadata_gen] store failed for act {act_id}: {e}")
        return False


def fetch_all_metadata() -> list[dict]:
    """
    Returns all acts that have stored AI metadata.
    Each record has: act_id, title, short_title, summary, keywords,
    legal_domains, regulated_activities, subjects.
    """
    resp = requests.get(
        f"{API_BASE_URL}/api/v1/acts/all-metadata",
        headers=AUTH_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def generate_and_store_all(skip_existing: bool = True) -> int:
    """
    Iterate over every active act, generate metadata if missing, and store it.
    Returns the count of acts processed.
    """
    acts = fetch_acts()
    processed = 0

    for act in acts:
        act_id = act["act_id"]
        title  = act.get("title", "")

        if skip_existing:
            check = requests.get(
                f"{API_BASE_URL}/api/v1/acts/{act_id}/metadata",
                headers=AUTH_HEADERS,
                timeout=10,
            )
            if check.status_code == 200:
                print(f"[metadata_gen] act {act_id} already has metadata — skipping")
                continue

        print(f"[metadata_gen] generating for act {act_id}: {title}")
        try:
            sample_data = fetch_act_text_sample(act_id)
            text_sample = sample_data.get("text_sample", title)
        except Exception as e:
            print(f"[metadata_gen] text sample fetch failed for act {act_id}: {e}")
            text_sample = title

        metadata = generate_act_metadata(act_id, title, text_sample)
        ok = store_act_metadata(act_id, metadata)
        if ok:
            processed += 1
            print(f"[metadata_gen] stored metadata for act {act_id}")
        else:
            print(f"[metadata_gen] failed to store metadata for act {act_id}")

    return processed
