import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def generate_answer(question, full_laws=None):
    laws_section = ""
    if full_laws:
        joined = "\n\n---\n\n".join(full_laws)
        laws_section = f"\n\nFull Referenced Laws:\n{joined}"

    prompt = f"""
You are a legal assistant for Consumer Protection laws.

Use ONLY the context below.

Context:
{laws_section}

Question:
{question}

Answer clearly:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
