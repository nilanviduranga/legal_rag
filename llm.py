import os
from openai import OpenAI

_PROVIDERS = {
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
    },
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-flash-latest",
    },
}

_PROVIDER: str = ""
_MODEL: str = ""
client = None        # OpenAI client — set for groq only
_gemini_client = None


def _init():
    global _PROVIDER, _MODEL, client, _gemini_client

    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        for name in ("groq", "gemini"):
            if os.getenv(_PROVIDERS[name]["key_env"]):
                provider = name
                break

    if not provider:
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY or GEMINI_API_KEY in .env"
        )

    cfg = _PROVIDERS.get(provider)
    if cfg is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}'. Supported: {', '.join(_PROVIDERS)}"
        )

    api_key = os.getenv(cfg["key_env"])
    if not api_key:
        raise RuntimeError(f"LLM_PROVIDER={provider} but {cfg['key_env']} is not set")

    _PROVIDER = provider
    _MODEL = os.getenv("LLM_MODEL", cfg["default_model"])

    if provider == "gemini":
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    print(f"[llm] provider={_PROVIDER} model={_MODEL}")


_init()


def llm_call(messages: list[dict], temperature: float, max_tokens: int) -> str:
    if _PROVIDER == "gemini":
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        config: dict = {"temperature": temperature, "max_output_tokens": max_tokens}
        if system:
            config["system_instruction"] = system

        resp = _gemini_client.models.generate_content(
            model=_MODEL,
            contents=contents,
            config=config,
        )
        return resp.text or ""
    else:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


_SYSTEM_PROMPT = (
    "You are a precise legal assistant specialising in Sri Lankan Consumer Protection and Labour laws.\n"
    "Answer ONLY from the legal context provided.\n"
    "If the answer cannot be determined from the provided context, say so explicitly — do not guess or fabricate.\n\n"
    "Context blocks are labelled [STATUTE] or [CASE LAW]:\n"
    "• Lead your answer with the relevant STATUTE provision (the rule).\n"
    "• Use CASE LAW to show how courts have interpreted or applied that rule in practice.\n"
    "• Cite the Act name and section for statutes; cite the case name and citation for case law.\n"
    "Be concise and direct."
)

_DISCOVERY_SYSTEM_PROMPT = (
    "You are a precise legal assistant specialising in Sri Lankan law.\n"
    "You have been given a complete list of Acts retrieved from the legal database.\n"
    "Answer ONLY from the Acts and evidence provided — do NOT invent additional Acts.\n"
    "For count questions: state the exact number first, then list the Acts.\n"
    "For list questions: list every Act provided, with a one-line description from its summary.\n"
    "Format each Act as: '<number>. <Act Title> — <summary or domain>'\n"
    "Be concise and factual."
)


def _format_block(chunk: dict | str) -> str:
    if isinstance(chunk, dict) and chunk.get("source") == "caselaw":
        return chunk.get("text", "")
    text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
    return f"[STATUTE]\n{text}"


def generate_answer(
    question: str,
    full_laws: list | None = None,
    summary: str | None = None,
    recent_chats: list | None = None,
) -> str:
    context_parts = []

    if summary:
        context_parts.append(f"[Conversation Summary]\n{summary}")

    if recent_chats:
        lines = []
        for chat in recent_chats:
            lines.append(f"User: {chat.get('user_message', '')}")
            lines.append(f"Assistant: {chat.get('ai_response', '')}")
        context_parts.append("[Recent Conversation]\n" + "\n".join(lines))

    if full_laws:
        context_parts.append("[Relevant Legal Context]\n\n---\n\n".join(_format_block(c) for c in full_laws))

    user_content = "\n\n".join(context_parts)
    user_content += f"\n\n[Question]\n{question}" if user_content else f"[Question]\n{question}"

    return llm_call(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=1024,
    )


def generate_discovery_answer(question: str, discovery_result) -> str:
    acts = discovery_result.acts

    if not acts:
        return (
            "I could not find any Acts in the database that are relevant to your question. "
            "The legal database may not yet contain Acts covering this topic."
        )

    act_lines = []
    for i, act in enumerate(acts, 1):
        desc = act.summary or (", ".join(act.legal_domains) if act.legal_domains else "")
        act_lines.append(f"{i}. {act.title}" + (f" — {desc}" if desc else ""))

    acts_block = "\n".join(act_lines)

    evidence_parts = []
    for act in acts:
        if act.evidence_chunks:
            evidence_parts.append(f"[{act.title}]\n{act.evidence_chunks[0]}")

    evidence_block = "\n\n---\n\n".join(evidence_parts) if evidence_parts else "(no section text retrieved)"

    user_content = (
        f"[Relevant Acts — {len(acts)} total]\n"
        f"{acts_block}\n\n"
        f"[Supporting Section Evidence]\n"
        f"{evidence_block}\n\n"
        f"[Question]\n{question}"
    )

    return llm_call(
        messages=[
            {"role": "system", "content": _DISCOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=1024,
    )


def generate_summary(old_summary, chats) -> str:
    chats_text = "\n".join(
        f"User: {c.get('user_message', '')}\nAssistant: {c.get('ai_response', '')}"
        for c in chats
    )
    prior_section = f"[Previous Summary]\n{old_summary}\n\n" if old_summary else ""

    system = (
        "You are a legal session summariser. Produce a concise factual summary (max 250 words) "
        "that captures the key legal questions asked and the key advice given. "
        "Preserve specific legal provisions and case citations mentioned."
    )
    user = f"{prior_section}[New Exchanges to include]\n{chats_text}"

    return llm_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=400,
    )
