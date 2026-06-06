from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Read PDF
reader = PdfReader("data/consumer_protection.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

# Chunk text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks")

# Embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "faiss_index/legal.index")

# Save chunks
with open("faiss_index/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Index saved successfully")