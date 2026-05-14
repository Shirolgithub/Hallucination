from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -----------------------------
# Load Lightweight GPT Model
# -----------------------------
generator = pipeline(
    "text-generation",
    model="distilgpt2"
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

"Photosynthesis is the process plants use to make food.",
"Artificial Intelligence refers to machines performing human-like tasks.",
"Java is used for backend development.",
"Mawsynram in India receives more rainfall than Seattle.",
"Penguins cannot fly.",
"The Sun is a star.",
"Machine learning is a subset of artificial intelligence.",
"Python supports artificial intelligence and web development.",
"Water boils at 100 degrees Celsius.",
"The Earth revolves around the Sun.",

"FAISS is a vector similarity search library.",
"NLP stands for Natural Language Processing.",
"Deep learning is a subset of machine learning.",
"A transformer model processes sequential data using attention mechanisms.",
"GPT2 is a transformer-based language model.",
"RAG stands for Retrieval-Augmented Generation.",
"Hallucination refers to factually incorrect AI-generated information.",
"Semantic search retrieves information based on meaning.",
"Flask is a lightweight Python web framework.",
"NumPy is a numerical computing library for Python.",

"TensorFlow is a machine learning framework.",
"PyTorch is a deep learning framework.",
"A chatbot is an AI system designed for conversation.",
"OpenAI is an artificial intelligence research organization.",
"An embedding is a vector representation of data.",
"Cosine similarity measures similarity between vectors.",
"FAISS is used for efficient vector similarity search.",
"Backend development handles server-side logic.",
"Frontend development focuses on user interfaces.",
"Cloud computing provides computing services over the internet.",

"Cybersecurity protects systems and data from attacks.",
"Data science extracts insights from data.",
"Computer vision enables machines to interpret images.",
"Reinforcement learning learns using rewards and penalties.",
"Supervised learning uses labeled training data.",
"Unsupervised learning finds patterns in unlabeled data.",
"Overfitting occurs when a model memorizes training data.",
"Underfitting occurs when a model fails to learn patterns.",
"An API enables communication between software systems.",
"GitHub is a platform for version control and collaboration.",

"Linux is an open-source operating system.",
"Windows is an operating system developed by Microsoft.",
"HTML is used to structure web pages.",
"CSS is used for styling web pages.",
"JavaScript adds interactivity to web pages.",
"SQL is used to manage databases.",
"A database stores organized information.",
"MongoDB is a NoSQL database.",
"MySQL is a relational database system.",
"Docker is a containerization platform.",

"Kubernetes manages containerized applications.",
"Virtualization creates virtual computing environments.",
"IoT stands for Internet of Things.",
"Blockchain is a decentralized digital ledger.",
"Cryptocurrency is a digital currency.",
"Bitcoin is a decentralized cryptocurrency.",
"Ethereum is a blockchain platform.",
"Data mining extracts patterns from data.",
"Big data refers to extremely large datasets.",
"Edge computing processes data near the source.",

"DevOps combines software development and operations.",
"Automation performs tasks with minimal human intervention.",
"Robotics involves designing intelligent machines.",
"An operating system manages computer hardware and software.",
"RAM is temporary memory used by computers.",
"ROM stores permanent system instructions.",
"CPU is the central processing unit.",
"GPU processes graphics and parallel computations.",
"Cache memory stores frequently used data.",
"A compiler converts source code into machine code.",

"An interpreter executes code line by line.",
"Object-oriented programming organizes software using objects and classes.",
"Polymorphism allows methods to behave differently.",
"Inheritance enables code reuse between classes.",
"Encapsulation hides internal implementation details.",
"Abstraction simplifies complex systems.",
"Recursion occurs when a function calls itself.",
"An algorithm is a step-by-step problem-solving method.",
"A data structure organizes data efficiently.",
"Binary search finds elements in sorted arrays.",

"Linear search checks elements sequentially.",
"Sorting arranges data in a specific order.",
"Quicksort is a divide-and-conquer sorting algorithm.",
"Merge sort divides and merges arrays.",
"A neural network mimics the human brain.",
"A dataset is a collection of data.",
"Training data teaches machine learning models.",
"Testing data evaluates model performance.",
"Accuracy measures correct predictions.",
"Precision measures positive prediction correctness.",

"Recall measures detection of actual positives.",
"F1 score balances precision and recall.",
"A confusion matrix evaluates classification performance.",
"Evaluation measures model effectiveness.",
"Bias in AI causes unfair outcomes.",
"Ethical AI ensures fairness and transparency.",
"Generative AI creates new content.",
"ChatGPT is a conversational AI model.",
"Prompt engineering designs effective AI prompts.",
"Semantic similarity measures meaning closeness."

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
        temperature=0.1,
        pad_token_id=50256
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