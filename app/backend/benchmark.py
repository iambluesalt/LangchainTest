"""
Embedding Model Benchmark Template
===================================
Configure everything in the CONFIG section below, then run:
    python benchmark.py

All knobs in one place — models, PDF, questions, chunking, concurrency, thresholds.
"""

import time, shutil, asyncio
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ║ CONFIGURE HERE ║

# --- Models to compare (add/remove as many as you want) ---
MODELS = [
    "nomic-embed-text:v1.5",
    # "qwen3-embedding:0.6b",
    # "mxbai-embed-large",
    # "snowflake-arctic-embed2",
    # "bge-m3",
]

# --- Your PDF ---
PDF_PATH = "./../../public/NIPS-2017-attention-is-all-you-need-Paper.pdf"

# --- Your test questions (add/remove as many as you want) ---
QUESTIONS = [
    "What is the main architecture proposed in the paper?",
    "How does self-attention work in the Transformer model?",
    "What is multi-head attention and why is it used?",
    "What are the encoder and decoder components of the Transformer?",
    "What is positional encoding and why is it necessary?",
    "What BLEU score did the model achieve on English-to-German translation?",
    "How does the Transformer compare to recurrent neural networks?",
    "What is the scaled dot-product attention mechanism?",
    "How many parameters does the base Transformer model have?",
    "What training data and hardware were used for the experiments?",
]

# --- Knobs ---
CHUNK_SIZE    = 1000    # Try: 300, 500, 1000, 1500
CHUNK_OVERLAP = 150     # Try: 50, 100, 150, 200 (keep 10-20% of chunk_size)
BATCH_SIZE    = 10      # Chunks per embed batch (higher = faster, more memory)
TOP_K         = 1       # Results per query
CONCURRENCY   = 5       # Max parallel queries during search phase
SEARCH_TYPE   = "similarity"  # "similarity" or "mmr"
MMR_FETCH_K   = 10      # Only used when SEARCH_TYPE = "mmr"

# --- Score thresholds ---
GOOD_THRESHOLD = 0.70
OKAY_THRESHOLD = 0.50

# ║ END OF CONFIGURATION ║


def load_and_split():
    print(f"\n  Loading: {PDF_PATH}")
    docs = PyPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    splits = splitter.split_documents(docs)
    print(f"  Pages: {len(docs)} → {len(splits)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return splits


def embed_documents(model_name, splits):
    safe = model_name.replace(":", "_").replace("/", "_")
    db_path = f"./chroma_bench_{safe}"

    emb = OllamaEmbeddings(model=model_name)
    store = Chroma(collection_name=safe, embedding_function=emb, persist_directory=db_path)
    store.reset_collection()

    total = len(splits)
    start = time.time()
    for i in range(0, total, BATCH_SIZE):
        store.add_documents(documents=splits[i:i + BATCH_SIZE])
        print(f"\r  Embedding: {min(i + BATCH_SIZE, total)}/{total}", end="", flush=True)
    elapsed = time.time() - start
    print(f"\r  Embedded {total} chunks in {elapsed:.1f}s           ")
    return store, elapsed, db_path


async def search_all(store, questions):
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def limited_search(q):
        async with semaphore:
            if SEARCH_TYPE == "mmr":
                docs = await store.amax_marginal_relevance_search(q, k=TOP_K, fetch_k=MMR_FETCH_K)
                # MMR doesn't return scores, so compute them manually
                scored = await store.asimilarity_search_with_relevance_scores(q, k=TOP_K)
                return scored
            return await store.asimilarity_search_with_relevance_scores(q, k=TOP_K)

    return await asyncio.gather(*[limited_search(q) for q in questions])


def label(score):
    if score >= GOOD_THRESHOLD: return "\033[92mGOOD\033[0m", "GOOD"
    if score >= OKAY_THRESHOLD: return "\033[93mOKAY\033[0m", "OKAY"
    return "\033[91m BAD\033[0m", "BAD"


def benchmark_model(model_name, splits):
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")

    store, embed_time, db_path = embed_documents(model_name, splits)
    results = asyncio.run(search_all(store, QUESTIONS))

    scores, pages = [], []
    good = okay = bad = 0

    for q, res in zip(QUESTIONS, results):
        doc, score = res[0]
        page = doc.metadata.get("page_label", "?")
        colored, tag = label(score)
        scores.append(score)
        pages.append(page)
        if tag == "GOOD": good += 1
        elif tag == "OKAY": okay += 1
        else: bad += 1
        print(f"  [{colored}] {score:.4f} | Pg {page:>3} | {q[:58]}")

    avg = sum(scores) / len(scores)
    print(f"\n  {good} GOOD | {okay} OKAY | {bad} BAD | Avg: {avg:.4f} | Embed: {embed_time:.1f}s")

    shutil.rmtree(db_path, ignore_errors=True)
    return {"model": model_name, "scores": scores, "pages": pages,
            "avg": avg, "time": embed_time, "good": good, "okay": okay, "bad": bad}


def compare(all_results):
    if len(all_results) < 2:
        return

    print(f"\n{'='*70}")
    print(f"{'HEAD-TO-HEAD':^70}")
    print(f"{'='*70}")

    shorts = {r["model"]: r["model"].split(":")[0][-14:] for r in all_results}

    header = f"  {'Question':<40}"
    for r in all_results:
        header += f" {shorts[r['model']]:>14}"
    header += "  Best"
    print(header)
    print("  " + "-" * (len(header) - 2))

    wins = {r["model"]: 0 for r in all_results}
    for i, q in enumerate(QUESTIONS):
        row = f"  {q[:38]:<40}"
        best = max(r["scores"][i] for r in all_results)
        for r in all_results:
            s = r["scores"][i]
            marker = "*" if s >= best - 0.01 else " "
            row += f" {s:>13.4f}{marker}"
        winner = [r["model"] for r in all_results if r["scores"][i] >= best - 0.01]
        if len(winner) == len(all_results):
            row += "  tie"
        else:
            row += f"  {shorts[winner[0]]}"
            wins[winner[0]] += 1
        print(row)

    # Final summary
    col = max(len(shorts[r["model"]]) for r in all_results) + 8
    print(f"\n{'='*70}")
    print(f"{'RESULTS':^70}")
    print(f"{'='*70}")
    print(f"  {'':>18}", end="")
    for r in all_results:
        print(f" {shorts[r['model']]:>{col}}", end="")
    print()
    for metric, key, fmt in [
        ("Avg Score", "avg", ".4f"), ("Embed Time", "time", ".1f"),
        ("GOOD", "good", "d"), ("OKAY", "okay", "d"), ("BAD", "bad", "d"),
    ]:
        print(f"  {metric:>18}", end="")
        for r in all_results:
            val = f"{r[key]:{fmt}}" + ("s" if key == "time" else "")
            print(f" {val:>{col}}", end="")
        print()
    print(f"  {'Wins':>18}", end="")
    for r in all_results:
        print(f" {wins[r['model']]:>{col}}", end="")
    print()

    best = max(all_results, key=lambda r: r["avg"])
    fastest = min(all_results, key=lambda r: r["time"])
    print(f"\n  Best quality : {best['model']} (avg {best['avg']:.4f})")
    print(f"  Fastest      : {fastest['model']} ({fastest['time']:.1f}s)")
    if best["model"] != fastest["model"]:
        print(f"  Trade-off    : {fastest['model']} is {best['time']/fastest['time']:.1f}x faster but {best['avg']-fastest['avg']:+.4f} avg score")


def main():
    print(f"\n{'#'*70}")
    print(f"{'EMBEDDING BENCHMARK':^70}")
    print(f"{'#'*70}")
    print(f"  Models      : {', '.join(MODELS)}")
    print(f"  PDF         : {PDF_PATH}")
    print(f"  Questions   : {len(QUESTIONS)}")
    print(f"  Chunk       : {CHUNK_SIZE} / {CHUNK_OVERLAP}")
    print(f"  Search      : {SEARCH_TYPE} (top_k={TOP_K})")
    print(f"  Concurrency : {CONCURRENCY}")
    print(f"  Thresholds  : GOOD >= {GOOD_THRESHOLD}, OKAY >= {OKAY_THRESHOLD}")

    splits = load_and_split()
    all_results = [benchmark_model(m, splits) for m in MODELS]
    compare(all_results)
    print()


if __name__ == "__main__":
    main()
