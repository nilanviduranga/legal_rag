"""
Discovery pipeline for count/list intent queries.

Flow:
  1. act_discovery_search(query)  → ranked list of matching Acts (semantic)
  2. For each Act, evidence_for_act(query, act_id) → top relevant chunks
  3. Return structured DiscoveryResult for the LLM

This module never calls the LLM — it only gathers evidence.
"""

from dataclasses import dataclass, field
from search import act_discovery_search, evidence_for_act


@dataclass
class ActEvidence:
    act_id:               int
    title:                str
    short_title:          str | None
    summary:              str | None
    legal_domains:        list[str]
    regulated_activities: list[str]
    keywords:             list[str]
    score:                float
    evidence_chunks:      list[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    query:   str
    intent:  str
    acts:    list[ActEvidence] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.acts)


def run_discovery(query: str, intent: str) -> DiscoveryResult:
    """
    Main entry point for discovery queries.
    Returns a DiscoveryResult containing all matching acts with evidence chunks.
    """
    matched_acts = act_discovery_search(query)
    result = DiscoveryResult(query=query, intent=intent)

    for rec in matched_acts:
        act_id = rec.get("act_id")
        if act_id is None:
            continue

        chunks    = evidence_for_act(query, act_id)
        chunk_texts = [c["text"] if isinstance(c, dict) else c for c in chunks]

        result.acts.append(ActEvidence(
            act_id               = act_id,
            title                = rec.get("title", ""),
            short_title          = rec.get("short_title"),
            summary              = rec.get("summary"),
            legal_domains        = rec.get("legal_domains") or [],
            regulated_activities = rec.get("regulated_activities") or [],
            keywords             = rec.get("keywords") or [],
            score                = rec.get("_score", 0.0),
            evidence_chunks      = chunk_texts,
        ))

    return result
