"""Core RAG logic: retrieve relevant chunks, build a grounded prompt, generate."""

import chromadb

import config
import ollama_client

SYSTEM_PROMPT = """You are a biomedical research assistant answering questions about \
the p53 tumor suppressor, using ONLY the provided excerpts from PubMed papers.

Rules:
- Base every claim on the provided context. Do not use outside knowledge.
- Cite supporting papers inline using their [PMID] tags.
- If the context does not contain the answer, say so plainly.
- Be precise and concise; prefer mechanistic detail when present."""


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


def retrieve(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """Embed the question and return the top_k most similar chunks with metadata."""
    collection = get_collection()
    q_emb = ollama_client.embed_one(question)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def build_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        m = h["meta"]
        header = f"[PMID {m['pmid']}] {m['title']} ({m.get('journal','')}, {m.get('year','')})"
        blocks.append(f"{header}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, top_k: int = config.TOP_K, stream: bool = False):
    """Retrieve context and generate an answer. Returns (response, hits).

    When stream=True, the first element is a token generator.
    """
    hits = retrieve(question, top_k)
    context = build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = ollama_client.chat(messages, stream=stream)
    return response, hits
