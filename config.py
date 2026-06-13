"""Shared configuration for the p53 RAG pipeline."""

from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PAPERS_FILE = DATA_DIR / "papers.json"
CHROMA_DIR = ROOT / "chroma_db"

# --- pubmed search ---
PUBMED_QUERY = "p53 tumor suppressor"
NUM_PAPERS = 100
# NCBI identify tool/email for E-utilities
ENTREZ_EMAIL = "user@example.com"
ENTREZ_TOOL = "p53-rag-pipeline"

NCBI_API_KEY = ""

# --- Ollama ---
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text" #embedding model
CHAT_MODEL = "llama3.1:8b" # chatbot model

# --- chunking ---
CHUNK_SIZE = 1200       # characters
CHUNK_OVERLAP = 200     # characters, produces 154 chunks from 100 abstracts 

# --- retrieval ---
COLLECTION_NAME = "p53_papers" # name of chromaDB table 
TOP_K = 5

DATA_DIR.mkdir(exist_ok=True)
