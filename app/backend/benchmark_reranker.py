"""
Compare retrieval results with and without the re-ranker.

Usage:
    python -m app.backend.benchmark_reranker "your query here"
    python -m app.backend.benchmark_reranker  # runs default test queries
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv; load_dotenv()
from app.backend.semantic_functions import search
from app.backend.reranker import rerank

CANDIDATES = 20
TOP_K = 4
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.txt")

DEFAULT_QUERIES = [
    "What is the attention mechanism?",
    "How does multi-head attention work?",
    "How does scaled dot-product attention work?",
    "What is positional encoding and why is it needed?",
    "Why did the authors replace recurrence with attention?",
    "What are the encoder and decoder components of the transformer?",
    "What BLEU scores did the transformer achieve on translation tasks?",
]

SEPARATOR = "-" * 72


def _truncate(text: str, max_len: int = 180) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def run(query: str, out) -> None:
    def write(line: str = ""):
        print(line)
        out.write(line + "\n")

    write(f"\n{'=' * 72}")
    write(f"QUERY: {query}")
    write('=' * 72)

    baseline = search(query, k=TOP_K)
    write(f"\n[WITHOUT re-ranker]  top-{TOP_K} from Chroma\n{SEPARATOR}")
    for i, (doc, score) in enumerate(baseline, 1):
        src = doc.metadata.get("source", "?").split("\\")[-1].split("/")[-1]
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        write(f"  {i}. score={score:.4f}  [{src}  p.{page}]")
        write(f"     {_truncate(doc.page_content)}\n")

    candidates = search(query, k=CANDIDATES)
    reranked = rerank(query, candidates, top_k=TOP_K)
    write(f"[WITH re-ranker]     top-{TOP_K} from {len(candidates)} candidates\n{SEPARATOR}")
    for i, (doc, score) in enumerate(reranked, 1):
        src = doc.metadata.get("source", "?").split("\\")[-1].split("/")[-1]
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        write(f"  {i}. score={score:.4f}  [{src}  p.{page}]")
        write(f"     {_truncate(doc.page_content)}\n")

    baseline_ids = [id(doc) for doc, _ in baseline]
    reranked_ids = [id(doc) for doc, _ in reranked]
    if baseline_ids == reranked_ids:
        write("  >> Order unchanged — re-ranker agrees with embedding search.")
    else:
        write("  >> Order changed — re-ranker surfaced different top results.")


if __name__ == "__main__":
    queries = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUERIES

    with open(RESULTS_FILE, "w", encoding="utf-8") as out:
        header = f"Benchmark run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(header)
        out.write(header + "\n")
        for q in queries:
            run(q, out)

    print(f"\nResults saved to {RESULTS_FILE}")
