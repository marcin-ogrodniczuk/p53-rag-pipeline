"""Shared configuration for the p53 RAG pipeline."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PAPERS_FILE = DATA_DIR / "papers.json"
CHROMA_DIR = ROOT / "chroma_db"

# --- PubMed search ---
PUBMED_QUERY = "p53 tumor suppressor"
NUM_PAPERS = 100
# NCBI asks that you identify your tool/email for E-utilities. Polite, not required.
ENTREZ_EMAIL = "user@example.com"
ENTREZ_TOOL = "p53-rag-pipeline"
# Set an NCBI API key here to raise the rate limit from 3 to 10 req/s (optional).
NCBI_API_KEY = ""

# --- Ollama ---
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"

# --- Chunking ---
CHUNK_SIZE = 1200       # characters
CHUNK_OVERLAP = 200     # characters

# --- Retrieval ---
COLLECTION_NAME = "p53_papers"
TOP_K = 5

DATA_DIR.mkdir(exist_ok=True)
