import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = "app/backend/chroma_langchain_db"
OLLAMA_EMBED_MODEL = "nomic-embed-text:v1.5"
GEMINI_MODEL = "gemini-2.0-flash"
COLLECTION_NAME = "web_docs"
