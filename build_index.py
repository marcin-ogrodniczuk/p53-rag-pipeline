"""Chunk paper abstracts, embed them with Ollama, and store in ChromaDB."""

import json

import chromadb

import config
import ollama_client


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, preferring sentence breaks."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        window = text[start:end]
        # Try to end on a sentence boundary for cleaner chunks.
        if end < len(text):
            last_period = window.rfind(". ")
            if last_period > size * 0.5:
                end = start + last_period + 1
                window = text[start:end]
        chunks.append(window.strip())
        start = end - overlap
    return [c for c in chunks if c]


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def main():
    papers = json.loads(config.PAPERS_FILE.read_text())
    print(f"Loaded {len(papers)} papers.")

    collection = get_collection(reset=True)

    ids, documents, metadatas = [], [], []
    for paper in papers:
        # Embed title + abstract together; title gives strong topical signal.
        full_text = f"{paper['title']}\n\n{paper['abstract']}"
        chunks = chunk_text(full_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for j, chunk in enumerate(chunks):
            ids.append(f"{paper['pmid']}_{j}")
            documents.append(chunk)
            metadatas.append({
                "pmid": paper["pmid"],
                "title": paper["title"],
                "journal": paper.get("journal", ""),
                "year": paper.get("year", ""),
                "url": paper["url"],
                "chunk": j,
            })

    print(f"Created {len(documents)} chunks. Embedding via Ollama ({config.EMBED_MODEL})...")

    # Embed in batches with progress output.
    batch = 32
    embeddings = []
    for i in range(0, len(documents), batch):
        embeddings.extend(ollama_client.embed(documents[i:i + batch]))
        print(f"  embedded {min(i + batch, len(documents))}/{len(documents)} chunks")

    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {collection.count()} chunks into ChromaDB at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
