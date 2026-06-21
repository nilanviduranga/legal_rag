import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

_MODEL = "llama-3.1-8b-instant"

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

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


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

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content
