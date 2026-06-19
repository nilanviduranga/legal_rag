from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

reader = PdfReader("data/consumer_protection.pdf")
text = "".join(page.extract_text() + "\n" for page in reader.pages)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks")

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(chunks, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "faiss_index/legal.index")

with open("faiss_index/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

bm25_corpus = [chunk.lower().split() for chunk in chunks]
with open("faiss_index/bm25_corpus.pkl", "wb") as f:
    pickle.dump(bm25_corpus, f)

print("Index saved successfully")
