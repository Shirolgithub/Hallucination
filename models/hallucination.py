from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -----------------------------
# Load GPT2 Model
# -----------------------------
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# -----------------------------
# Load Embedding Model
# -----------------------------
embed_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# -----------------------------
# Knowledge Base
# -----------------------------
knowledge_base = [
    "CERN discovered the Higgs boson in 2012.",
    "Photosynthesis is the process plants use to make food.",
    "Artificial Intelligence refers to machines performing human-like tasks.",
    "Java is used for backend development.",
    "Mawsynram in India receives more rainfall than Seattle.",
    "The Earth revolves around the Sun.",
    "Quicksort is often faster in practice because of cache efficiency.",
    "Penguins cannot fly.",
    "Penguins are birds that cannot fly.",
    "The Sun is a star.",
    "The Sun is not a planet.",
    "Machine learning is a subset of artificial intelligence.",
    "Machine learning enables systems to learn from data.",
    "Water boils at 100 degrees Celsius.",
    "Python is a popular programming language.",
    "Python is used for artificial intelligence and web development."
]

# -----------------------------
# Create Embeddings
# -----------------------------
kb_embeddings = embed_model.encode(
    knowledge_base
)

kb_embeddings = np.array(
    kb_embeddings,
    dtype=np.float32
)

# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = kb_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(kb_embeddings)

# -----------------------------
# Semantic Retrieval
# -----------------------------
def retrieve(query, top_k=1):

    query_embedding = embed_model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(
            knowledge_base[idx]
        )

    return results

# -----------------------------
# Generate Answer
# -----------------------------
def generate_answer(query):

    prompt = f"Question: {query}\nAnswer:"

    result = generator(
        prompt,
        max_new_tokens=25,
        do_sample=False,
        temperature=0.1
    )

    answer = result[0]["generated_text"]

    answer = answer.replace(
        prompt,
        ""
    ).strip()

    return answer

# -----------------------------
# Confidence Score
# -----------------------------
def confidence_score(answer, context):

    answer_embedding = embed_model.encode([answer])

    context_embedding = embed_model.encode(context)

    answer_embedding = np.array(
        answer_embedding,
        dtype=np.float32
    )

    context_embedding = np.array(
        context_embedding,
        dtype=np.float32
    )

    similarities = []

    for ctx in context_embedding:

        similarity = np.dot(
            answer_embedding[0],
            ctx
        ) / (
            np.linalg.norm(answer_embedding[0])
            *
            np.linalg.norm(ctx)
        )

        similarities.append(similarity)

    max_similarity = max(similarities)

    return round(
        float(max_similarity),
        3
    )

# -----------------------------
# Severity Level
# -----------------------------
def severity_level(score):

    if score >= 0.90:
        return "Low Hallucination Risk"

    elif score >= 0.70:
        return "Medium Hallucination Risk"

    elif score >= 0.50:
        return "High Hallucination Risk"

    else:
        return "Critical Hallucination Detected"

# -----------------------------
# Main Framework
# -----------------------------
def hallucination_guard(query):

    # Step 1: Generate Answer
    answer = generate_answer(query)

    # Step 2: Retrieve Context
    context = retrieve(query)

    # Step 3: Confidence Scoring
    score = confidence_score(
        answer,
        context
    )

    # Step 4: Severity Detection
    severity = severity_level(score)

    # Step 5: Smart Correction
    if score < 0.90:

        final_answer = context[0]

    else:

        final_answer = answer

    return {

        "llm_answer": answer,

        "context": context,

        "confidence": score,

        "severity": severity,

        "final_answer": final_answer
    }