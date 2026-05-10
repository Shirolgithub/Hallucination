from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Knowledge Base
documents = [
    "CERN discovered the Higgs boson in 2012.",
    "Photosynthesis is the process plants use to make food.",
    "Artificial Intelligence refers to machines performing human-like tasks.",
    "Java is used for backend development.",
    "Mawsynram in India receives more rainfall than Seattle."
]

# Load embedding model
model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# Create embeddings
embeddings = model.encode(documents)

# Convert to float32
embeddings = np.array(
    embeddings,
    dtype=np.float32
)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# User query
query = "What is photosynthesis?"

# Encode query
query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding,
    dtype=np.float32
)

# Search
k = 2

distances, indices = index.search(
    query_embedding,
    k
)

print("\nTop Results:\n")

for idx in indices[0]:
    print(documents[idx])