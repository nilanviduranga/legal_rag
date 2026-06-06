from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

def generate_answer(context, question):

    prompt = f"""
You are a legal assistant for Consumer Protection laws.

Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer clearly:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content