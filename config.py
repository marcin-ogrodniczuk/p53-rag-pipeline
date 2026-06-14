"""configuration for the p53 RAG pipeline."""

from pathlib import Path

# --- paths --- tell the file where to live on disk,
ROOT = Path(__file__).resolve().parent # figures out the project folder 
DATA_DIR = ROOT / "data"
PAPERS_FILE = DATA_DIR / "papers.json"
CHROMA_DIR = ROOT / "chroma_db"

# --- pubmed search --- # what data enters the system
PUBMED_QUERY = "p53 tumor suppressor" # what to search for 
NUM_PAPERS = 100 # how many to fetch 
# NCBI identify tool/email for E-utilities
ENTREZ_EMAIL = "user@example.com" # polite ID for NCBI 
ENTREZ_TOOL = "p53-rag-pipeline" # required for NCBI API 

NCBI_API_KEY = ""

# --- Ollama --- # Which AI models section 
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text" #embedding model
CHAT_MODEL = "llama3.1:8b" # chatbot model

# --- chunking --- # tradeoff between precision and completeness 
CHUNK_SIZE = 1200       # characters, median of the abstract length = 1254 characters
CHUNK_OVERLAP = 200     # characters, produces 154 chunks from 100 abstracts 

# --- retrieval ---  # label for vector table (both build_index.pu and rag.py use to read/write data
COLLECTION_NAME = "p53_papers" # name of chromaDB table 
TOP_K = 5

DATA_DIR.mkdir(exist_ok=True)
