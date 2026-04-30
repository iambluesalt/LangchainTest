import torch
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"[reranker] Loading {MODEL_NAME} on {DEVICE.upper()}")
        _model = CrossEncoder(MODEL_NAME, device=DEVICE)
    return _model


def rerank(
    query: str,
    results: List[Tuple[Document, float]],
    top_k: int = 4,
) -> List[Tuple[Document, float]]:
    if not results:
        return results

    model = _get_model()
    pairs = [(query, doc.page_content) for doc, _ in results]
    scores = model.predict(pairs)

    reranked = sorted(zip([doc for doc, _ in results], scores), key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in reranked[:top_k]]
