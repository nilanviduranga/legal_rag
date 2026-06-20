import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

_MODEL = "llama-3.1-8b-instant"


def generate_answer(question: str, full_laws=None, summary=None, recent_chats=None) -> str:
    laws_section = ""
    if full_laws:
        laws_section = "\nReferenced Laws:\n" + "\n\n---\n\n".join(full_laws)

    history_section = ""
    if summary:
        history_section += f"\nConversation Summary (earlier context):\n{summary}\n"
    if recent_chats:
        lines = []
        for chat in recent_chats:
            lines.append(f"User: {chat.get('user_message', '')}")
            lines.append(f"Assistant: {chat.get('ai_response', '')}")
        history_section += "\nRecent Conversation:\n" + "\n".join(lines) + "\n"

    prompt = (
        "You are a legal assistant for Consumer Protection laws.\n"
        "Use ONLY the legal context provided below. If prior conversation history is present, "
        "use it to understand the full context of the current question.\n"
        f"{history_section}{laws_section}\n"
        f"Question: {question}\n\n"
        "Answer clearly:"
    )

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_summary(old_summary, chats) -> str:
    chats_text = "\n".join(
        f"User: {c.get('user_message', '')}\nAssistant: {c.get('ai_response', '')}"
        for c in chats
    )
    prior_section = f"Previous Summary:\n{old_summary}\n\n" if old_summary else ""

    prompt = (
        "You are summarising a legal consultation chat for a Consumer Protection law assistant.\n"
        "Create a concise summary that preserves key legal questions asked and the advice given.\n"
        "This summary will be used as context for future questions in this session.\n\n"
        f"{prior_section}New Conversations to include:\n{chats_text}\n\n"
        "Provide a concise, factual summary (max 300 words):"
    )

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
