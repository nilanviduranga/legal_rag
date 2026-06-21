import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

_MODEL = "llama-3.1-8b-instant"


_SYSTEM_PROMPT = (
    "You are a precise legal assistant specialising in Sri Lankan Consumer Protection and Labour laws. "
    "Answer ONLY from the legal context provided. "
    "If the answer cannot be determined from the provided context, say so explicitly — do not guess or fabricate. "
    "When citing a legal provision, mention the relevant section or act name. "
    "Be concise and direct."
)


def generate_answer(question: str, full_laws=None, summary=None, recent_chats=None) -> str:
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
        context_parts.append("[Relevant Legal Provisions]\n" + "\n\n---\n\n".join(full_laws))

    user_content = "\n\n".join(context_parts)
    if user_content:
        user_content += f"\n\n[Question]\n{question}"
    else:
        user_content = f"[Question]\n{question}"

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
        "This will be used as context for future questions — preserve specific legal provisions mentioned."
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
