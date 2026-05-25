from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# =========================================
# Load Embedding Model
# =========================================
embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================
# Load Knowledge Base
# =========================================
with open(
    "data/knowledge_base.txt",
    "r",
    encoding="utf-8"
) as file:

    knowledge_base = [

        line.strip()

        for line in file

        if line.strip()
    ]

# =========================================
# Create Embeddings
# =========================================
kb_embeddings = embed_model.encode(
    knowledge_base
)

kb_embeddings = np.array(
    kb_embeddings,
    dtype=np.float32
)

# =========================================
# Build FAISS Index
# =========================================
dimension = kb_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(kb_embeddings)

# =========================================
# Semantic Retrieval
# =========================================
def retrieve(query, top_k=3):

    query_embedding = embed_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_contexts = []

    for idx in indices[0]:

        retrieved_contexts.append(
            knowledge_base[idx]
        )

    return retrieved_contexts

# =========================================
# Generate Simulated LLM Answer
# =========================================
def generate_answer(query):

    retrieved = retrieve(query)

    # intentionally imperfect response
    answer = retrieved[0]

    # simulate hallucination sometimes
    if "Artificial Intelligence" in answer:

        return "AI refers to intelligent computer systems."

    elif "Machine learning" in answer:

        return "Machine learning allows systems to learn automatically."

    elif "Photosynthesis" in answer:

        return "Plants make food using sunlight."

    return answer

# =========================================
# Confidence Score
# =========================================
def confidence_score(answer, context):

    answer_embedding = embed_model.encode(
        [answer]
    )

    context_embedding = embed_model.encode(
        context
    )

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

        similarities.append(
            similarity
        )

    max_similarity = max(
        similarities
    )

    return round(
        float(max_similarity),
        3
    )

# =========================================
# Severity Detection
# =========================================
def severity_level(score):

    if score >= 0.90:

        return "Low Hallucination Risk"

    elif score >= 0.70:

        return "Medium Hallucination Risk"

    elif score >= 0.50:

        return "High Hallucination Risk"

    else:

        return "Critical Hallucination Detected"

# =========================================
# Main Framework
# =========================================
def hallucination_guard(query):

    # Step 1
    llm_answer = generate_answer(
        query
    )

    # Step 2
    retrieved_context = retrieve(
        query
    )

    # Step 3
    score = confidence_score(
        llm_answer,
        retrieved_context
    )

    # Step 4
    severity = severity_level(
        score
    )

    # Step 5
    if score < 0.90:

        final_answer = retrieved_context[0]

    else:

        final_answer = llm_answer

    return {

        "llm_answer":
        llm_answer,

        "context":
        retrieved_context,

        "confidence":
        score,

        "severity":
        severity,

        "final_answer":
        final_answer
    }