import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
LLM_MODEL = "gpt2" 
CHROMA_PERSIST_DIR = DB_DIR
