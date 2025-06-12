import os
from io import StringIO

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groqcloud import GroqCloud
import PyPDF2
import networkx as nx

# ——— Configuration ———
API_KEY = "YOUR_GROQCLOUD_API_KEY"
MODEL_NAME = "chatgroq-7b"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 100
TOP_K = 3

# ——— Initialize clients & models ———
gc = GroqCloud(api_key=API_KEY)
embedder = SentenceTransformer(EMBED_MODEL)

def load_pdf_text(path: str) -> str:
    reader = PyPDF2.PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def chunk_text(text: str, size: int, overlap: int):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def build_faiss_index(passages: list):
    embeddings = embedder.encode(passages, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve(query: str, passages: list, index, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, ids = index.search(q_emb, top_k)
    return [passages[i] for i in ids[0]]


def chat_with_context(docs: list, question: str, history=None) -> str:
    prompt = (
        "Use the following context to answer the question:\n\n"
        + "\n".join(f"- {d}" for d in docs)
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    messages = (history or []) + [{"role": "user", "content": prompt}]
    resp = gc.chat.create(model=MODEL_NAME, messages=messages)
    return resp.choices[0].message.content

# ——— RAG Architectures ———

def corrective_rag(q: str, passages: list, index) -> str:
    docs = retrieve(q, passages, index)
    initial = chat_with_context(docs, q)
    verify = (
        f"You answered:\n“{initial}”\n\n"
        "Check against the provided context. Correct any factual errors and restate."
    )
    resp = gc.chat.create(model=MODEL_NAME, messages=[{"role": "user", "content": verify}])
    return resp.choices[0].message.content


def speculative_rag(q: str, passages: list, index) -> str:
    # Draft with retrieval then verify with corrective
    _ = retrieve(q, passages, index)
    return corrective_rag(q, passages, index)


def agentic_rag(q: str, passages: list, index) -> str:
    use_vector = len(q) % 2 == 0
    docs = retrieve(q, passages, index) if use_vector else ["<keyword-retrieved-doc>"]
    return chat_with_context(docs, q)


def single_router_rag(q: str, passages: list, index) -> str:
    mode = "vector" if any(x in q.lower() for x in ["define", "what is"]) else "keyword"
    docs = retrieve(q, passages, index) if mode == "vector" else ["<keyword-doc>"]
    return chat_with_context(docs, q)


def multi_agent_rag(q: str, passages: list, index) -> str:
    v_docs = retrieve(q, passages, index, top_k=2)
    k_docs = ["<keyword-doc>"]
    w_docs = ["<web-scraped-doc>"]
    all_docs = v_docs + k_docs + w_docs
    return chat_with_context(all_docs, q)


def self_reflective_rag(q: str, passages: list, index) -> str:
    docs = retrieve(q, passages, index, top_k=5)
    filter_prompt = (
        "From these passages, select the 3 most relevant to the question:\n\n"
        + "\n".join(f"{i+1}. {d}" for i, d in enumerate(docs))
        + f"\n\nQuestion: {q}\nRelevant passages (by number):"
    )
    pick = gc.chat.create(model=MODEL_NAME, messages=[{"role": "user", "content": filter_prompt}])
    picks = [int(n) - 1 for n in pick.choices[0].message.content.split() if n.isdigit()]
    selected = [docs[i] for i in picks[:3] if 0 <= i < len(docs)]
    return chat_with_context(selected, q)


def self_route_rag(q: str, passages: list, index) -> str:
    small_docs = retrieve(q, passages, index, top_k=2)
    small_ans = chat_with_context(small_docs, q)
    check_prompt = (
        f"You answered:\n“{small_ans}”\n\n"
        "Is the context sufficient? Answer 'yes' or 'no'."
    )
    verdict = gc.chat.create(model=MODEL_NAME, messages=[{"role": "user", "content": check_prompt}])
    if verdict.choices[0].message.content.strip().lower().startswith("no"):
        full_docs = retrieve(q, passages, index, top_k=5)
        return chat_with_context(full_docs, q)
    return small_ans


def graph_rag(q: str, passages: list, index) -> str:
    # simple KG: co-occurrence graph
    G = nx.Graph()
    for i, doc in enumerate(passages):
        G.add_node(i, text=doc)
    for i in range(len(passages)):
        for j in range(i+1, len(passages)):
            G.add_edge(i, j)
    q_emb = embedder.encode([q], convert_to_numpy=True)
    sims = (embedder.encode(passages, convert_to_numpy=True) @ q_emb.T).flatten()
    top_idx = sims.argsort()[-2:]
    sub_nodes = set(top_idx)
    for idx in top_idx:
        sub_nodes.update(G.neighbors(idx))
    sub_docs = [passages[i] for i in sub_nodes]
    return chat_with_context(sub_docs, q)

# ——— Main flow ———
if __name__ == "__main__":
    pdf_path = input("Enter path to your PDF file: ").strip()
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        exit(1)

    print("Loading and indexing PDF—this may take a moment...")
    full_text = load_pdf_text(pdf_path)
    passages = chunk_text(full_text, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP)
    index = build_faiss_index(passages)
    print(f"Indexed {len(passages)} chunks from PDF.\n")

    rag_funcs = {
        "corrective": corrective_rag,
        "speculative": speculative_rag,
        "agentic": agentic_rag,
        "single-router": single_router_rag,
        "multi-agent": multi_agent_rag,
        "self-reflect": self_reflective_rag,
        "self-route": self_route_rag,
        "graph": graph_rag,
    }

    print("Available RAG modes:", ", ".join(rag_funcs.keys()), "\n")
    while True:
        mode = input("Choose RAG mode (or 'exit'): ").strip().lower()
        if mode in ("exit", "quit"): break
        if mode not in rag_funcs:
            print("Invalid mode. Try again.")
            continue
        question = input("Enter your question: ").strip()
        answer = rag_funcs[mode](question, passages, index)
        print(f"\n📝 Answer ({mode} RAG):\n", answer, "\n")
