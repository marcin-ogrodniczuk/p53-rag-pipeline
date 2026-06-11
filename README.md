# p53 Tumor Suppressor — Local RAG Pipeline

A fully local retrieval-augmented generation (RAG) system over 100 PubMed papers on
the **p53 tumor suppressor**. Embeddings and generation both run on **Ollama**.

## Architecture

```
PubMed (Entrez E-utilities)
        │  fetch_pubmed.py
        ▼
   data/papers.json   ── 100 abstracts + metadata
        │  build_index.py  (chunk → embed with nomic-embed-text)
        ▼
   chroma_db/         ── persistent vector store (ChromaDB, cosine)
        │  rag.py  (retrieve top-k → grounded prompt)
        ▼
   Ollama llama3.1:8b ── cited answer
        ▲
   cli.py  /  app.py (Streamlit chat)
```

## Prerequisites

- [Ollama](https://ollama.com) installed and running (`ollama serve` happens automatically on macOS).
- Python 3.10+.

Pull the two models (once):

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

Install Python deps:

```bash
pip install -r requirements.txt
```

## Usage

Run these from inside the `p53_rag/` directory, in order:

```bash
# 1. Download 100 p53 abstracts from PubMed
python fetch_pubmed.py

# 2. Chunk, embed, and index them into ChromaDB
python build_index.py

# 3a. Ask questions from the command line
python cli.py "How does p53 trigger apoptosis after DNA damage?"

# 3b. ...or launch the chat UI
streamlit run app.py
```

## Configuration

All knobs live in `config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `PUBMED_QUERY` | `p53 tumor suppressor` | PubMed search term |
| `NUM_PAPERS` | `100` | How many papers to fetch |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHAT_MODEL` | `llama3.1:8b` | Ollama generation model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Character-based chunking |
| `TOP_K` | `5` | Chunks retrieved per query |
| `NCBI_API_KEY` | `""` | Optional — raises PubMed rate limit |

## Notes

- This indexes **abstracts**, which are reliably available for all 100 papers. To use
  full text, you'd fetch the PubMed Central open-access subset (a subset of papers) and
  point the ingest at those XML/PDF files instead.
- Answers are grounded: the model is instructed to answer only from retrieved context
  and to cite `[PMID]` tags. It will say when the context doesn't cover a question.
- Re-running `build_index.py` rebuilds the collection from scratch.
```
