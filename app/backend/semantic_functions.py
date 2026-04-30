import os
import re
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_langchain_db")
COLLECTION_NAME = "pdf_collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text:v1.5")


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def _fix_char_spacing(text: str) -> str:
    tokens = text.split(" ")
    single_char_ratio = sum(1 for t in tokens if len(t) == 1) / max(len(tokens), 1)
    if single_char_ratio > 0.4:
        # Collapse intra-word spaces (single spaces between non-whitespace chars)
        text = re.sub(r"(?<=[^\s]) (?=[^\s])", "", text)
        # Normalize leftover multi-spaces back to a single space (were inter-word gaps)
        text = re.sub(r" {2,}", " ", text)
    return text


def load_and_split(file_path: str) -> List[Document]:
    docs = PyPDFLoader(file_path).load()
    for doc in docs:
        doc.page_content = _fix_char_spacing(doc.page_content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    return splitter.split_documents(docs)


def index_documents(chunks: List[Document]) -> int:
    store = get_vector_store()
    batch_size = 10
    ids = []
    for i in range(0, len(chunks), batch_size):
        ids.extend(store.add_documents(documents=chunks[i : i + batch_size]))
    return len(ids)


def search(query: str, k: int = 4) -> List[Tuple[Document, float]]:
    store = get_vector_store()
    return store.similarity_search_with_relevance_scores(query, k=k)


def clear_collection() -> int:
    store = get_vector_store()
    ids = store.get()["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)
