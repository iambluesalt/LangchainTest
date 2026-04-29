# FastAPI PDF RAG (Retrieval-Augmented Generation) application
# Run with: uvicorn app.backend.main:app --reload (development) or uvicorn app.backend.main:app (production)
# Dependencies: pip install fastapi uvicorn python-dotenv
# Environment: Create .env file with required API keys (e.g., OPENAI_API_KEY, PINECONE_API_KEY)

import os
import shutil
import tempfile
from dotenv import load_dotenv; load_dotenv()
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.backend.semantic_functions import load_and_split, index_documents, search, clear_collection

app = FastAPI(title="PDF RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = load_and_split(tmp_path)
        count = index_documents(chunks)
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "chunks_indexed": count}


@app.delete("/collection")
def delete_collection():
    deleted = clear_collection()
    return {"deleted_chunks": deleted}


@app.post("/search")
def search_docs(req: SearchRequest):
    results = search(req.query, k=req.k)
    return {
        "query": req.query,
        "results": [
            {
                "score": round(score, 4),
                "page": doc.metadata.get("page_label", doc.metadata.get("page", "?")),
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "content": doc.page_content,
            }
            for doc, score in results
        ],
    }
