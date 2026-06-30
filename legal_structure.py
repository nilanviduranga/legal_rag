"""
Legal Structure Query Pipeline — deterministic DB-backed answers.

Used for structural intent queries:
  "How many sections are in the Consumer Affairs Authority Act?"
  "List all sections of the Labour Act"
  "How many provisions does the Penal Code have?"

Flow:
  1. Extract act name from question (regex)
  2. Resolve act_id via legal_admin search API
  3. Fetch structure stats OR node list from legal_admin DB (no FAISS, no LLM)
  4. Return a deterministic StructuralResult dict ready for the API response
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from client import (
    search_acts_by_title,
    fetch_act_structure_stats,
    fetch_act_nodes_by_type,
)

# ── Node type alias map ──────────────────────────────────────────────────────
# Maps user-facing words to legal_nodes.node_type enum values

_NODE_TYPE_ALIASES: dict[str, str] = {
    "section":     "SECTION",
    "sections":    "SECTION",
    "provision":   "SECTION",
    "provisions":  "SECTION",
    "clause":      "SUBSECTION",
    "clauses":     "SUBSECTION",
    "subsection":  "SUBSECTION",
    "subsections": "SUBSECTION",
    "paragraph":   "PARAGRAPH",
    "paragraphs":  "PARAGRAPH",
    "part":        "PART",
    "parts":       "PART",
    "chapter":     "CHAPTER",
    "chapters":    "CHAPTER",
    "item":        "ITEM",
    "items":       "ITEM",
    "schedule":    "SCHEDULE",
    "schedules":   "SCHEDULE",
}

# Pattern to extract act name from a natural-language question.
# Captures text that ends with Act/Law/Ordinance/Code/Statute.
_ACT_NAME_PATTERN = re.compile(
    r"""
    (?:in|of|under|within|about)\s+(?:the\s+)?
    ([A-Z][^?.,!;\n]+?(?:Act|Law|Ordinance|Code|Statute)\b[^?,!;\n]*)
    |
    ([A-Z][^?.,!;\n]+?(?:Act|Law|Ordinance|Code|Statute)\b[^?,!;\n]*)
    """,
    re.VERBOSE,
)

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class StructuralResult:
    query:       str
    intent:      str             # "structural"
    found:       bool = False
    act_id:      int | None = None
    act_title:   str = ""
    short_title: str | None = None
    node_type:   str = "SECTION"  # what was asked about
    count:       int = 0
    nodes:       list[dict] = field(default_factory=list)
    stats:       dict = field(default_factory=dict)
    message:     str = ""
    error:       str = ""


# ── Public entry point ───────────────────────────────────────────────────────

def run_structural_query(question: str, intent: str) -> StructuralResult:
    """
    Main entry point for structural intent queries.
    Returns a StructuralResult — never calls FAISS or LLM.
    """
    result = StructuralResult(query=question, intent=intent)

    act_name = _extract_act_name(question)
    if not act_name:
        result.error = (
            "I could not identify a specific Act name in your question. "
            "Please mention the full Act name, for example: "
            "'How many sections are in the Consumer Affairs Authority Act?'"
        )
        return result

    node_type = _extract_node_type(question)
    result.node_type = node_type

    # Resolve act_id via legal_admin title search
    matches = search_acts_by_title(act_name)
    if not matches:
        result.error = (
            f"No Act matching '{act_name}' was found in the legal database. "
            "Please check the Act name and try again."
        )
        return result

    act = matches[0]
    result.found      = True
    result.act_id     = act["act_id"]
    result.act_title  = act["title"]
    result.short_title = act.get("short_title")

    # Determine whether to count or list
    is_list_question = _is_list_question(question)

    if is_list_question:
        _populate_list(result)
    else:
        _populate_count(result)

    return result


# ── Response helpers ─────────────────────────────────────────────────────────

def _populate_count(result: StructuralResult) -> None:
    """Fetch and format a count answer using structure-stats endpoint."""
    stats = fetch_act_structure_stats(result.act_id)
    if not stats:
        result.error = f"Could not retrieve structure statistics for '{result.act_title}'."
        return

    result.stats = stats
    type_key = _stats_key_for_type(result.node_type)
    result.count = stats.get(type_key, 0)

    label = result.node_type.lower() + ("s" if result.count != 1 else "")
    act_label = result.short_title or result.act_title
    result.message = (
        f"The {act_label} contains {result.count} {label}."
    )


def _populate_list(result: StructuralResult) -> None:
    """Fetch and format a list answer using nodes-by-type endpoint."""
    data = fetch_act_nodes_by_type(result.act_id, result.node_type)
    if not data:
        result.error = f"Could not retrieve {result.node_type.lower()}s for '{result.act_title}'."
        return

    result.count = data.get("count", 0)
    result.nodes = data.get("nodes", [])

    act_label = result.short_title or result.act_title
    label = result.node_type.lower() + ("s" if result.count != 1 else "")
    result.message = (
        f"The {act_label} has {result.count} {label}."
    )


# ── Format for API response ──────────────────────────────────────────────────

def format_structural_answer(result: StructuralResult) -> str:
    """
    Build a human-readable answer string from a StructuralResult.
    No LLM is involved — this is purely deterministic text formatting.
    """
    if result.error:
        return result.error

    if not result.found:
        return "I could not find the requested Act in the legal database."

    if result.nodes:
        lines = [result.message, ""]
        for node in result.nodes:
            no      = node.get("node_no") or ""
            heading = node.get("heading") or ""
            label   = f"Section {no}" if no else result.node_type.capitalize()
            line    = f"• {label}" + (f": {heading}" if heading else "")
            lines.append(line)
        return "\n".join(lines)

    return result.message


# ── Private helpers ──────────────────────────────────────────────────────────

def _extract_act_name(question: str) -> str | None:
    """
    Extract a specific Act name from the question using regex.
    Returns None if no Act name pattern is found.
    """
    for match in _ACT_NAME_PATTERN.finditer(question):
        candidate = (match.group(1) or match.group(2) or "").strip()
        # Require at least two words so we don't capture bare "Act"
        if candidate and len(candidate.split()) >= 2:
            return candidate
    return None


def _extract_node_type(question: str) -> str:
    """Detect which node type the question is about; defaults to SECTION."""
    q_lower = question.lower()
    for alias, db_type in _NODE_TYPE_ALIASES.items():
        if re.search(r'\b' + alias + r'\b', q_lower):
            return db_type
    return "SECTION"


def _is_list_question(question: str) -> bool:
    """Return True if the question asks to list/enumerate rather than just count."""
    triggers = [
        r'\blist\b', r'\bshow\b', r'\benumerate\b', r'\bwhat are\b',
        r'\bname all\b', r'\bgive me all\b', r'\ball sections\b',
        r'\ball provisions\b', r'\ball clauses\b',
    ]
    q_lower = question.lower()
    return any(re.search(t, q_lower) for t in triggers)


def _stats_key_for_type(node_type: str) -> str:
    """Map node_type enum value to the act_statistics column name."""
    return {
        "SECTION":    "sections_count",
        "SUBSECTION": "subsections_count",
        "PART":       "parts_count",
        "CHAPTER":    "chapters_count",
        "PARAGRAPH":  "paragraphs_count",
        "ITEM":       "items_count",
    }.get(node_type, "sections_count")
