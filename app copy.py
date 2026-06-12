# from fastapi import FastAPI
# from pydantic import BaseModel
# from llm import generate_answer


# import faiss
# import pickle
# import numpy as np

# from sentence_transformers import SentenceTransformer

# app = FastAPI()

# # Load model
# model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# # Load index
# index = faiss.read_index("faiss_index/legal.index")

# # Load chunks
# with open("faiss_index/chunks.pkl", "rb") as f:
#     chunks = pickle.load(f)


# class Question(BaseModel):
#     question: str


# #@app.post("/search")
# #def search(question: Question):

# #    query_embedding = model.encode([question.question])

# #    distances, indices = index.search(
# #        np.array(query_embedding),
# #        k=4
# #    )

# #    results = []

# #    for idx in indices[0]:
# #        results.append(chunks[idx])

# #    return {
# #        "question": question.question,
# #        "results": results
# #    }


# @app.post("/ask")
# def ask(question: Question):

#     query_embedding = model.encode([question.question])

#     distances, indices = index.search(
#         np.array(query_embedding),
#         k=3
#     )

#     matched = [chunks[i] for i in indices[0]]

#     # support both old (str) and new (dict) chunk formats
#     def chunk_text(c):
#         return c["text"] if isinstance(c, dict) else c

#     context = "\n\n".join(chunk_text(c) for c in matched)

#     answer = generate_answer(context, question.question)

#     sources = [
#         {"node_id": c["node_id"], "path": c["path"], "text": c["text"]}
#         if isinstance(c, dict) else c
#         for c in matched
#     ]

#     return {
#         "question": question.question,
#         "answer": answer,
#         "sources": sources,
#     }