# Semantic Search Tips

## Chunking

| Setting | Small (300) | Medium (500) | Large (1000) | XL (1500) |
|---------|------------|--------------|--------------|-----------|
| Precision | High | Good | Moderate | Low |
| Context | Thin | Balanced | Rich | Too noisy |
| Chunk count | Many | Moderate | Few | Very few |
| Best for | FAQs, definitions | General use | Long-form docs | Books, legal |

- **Overlap** should be 10-20% of chunk size (e.g., 500 chunk -> 50-100 overlap)
- Too little overlap = lost context at boundaries
- Too much overlap = redundant embeddings, wasted compute

## Embedding Models (Ollama, Free)

| Model | Params | Dims | Best For |
|-------|--------|------|----------|
| `nomic-embed-text` | 137M | 768 | Fast, lightweight, good baseline |
| `mxbai-embed-large` | 335M | 1024 | Better accuracy, still fast |
| `snowflake-arctic-embed2` | 568M | 1024 | Strong English retrieval |
| `bge-m3` | 567M | 1024 | Multilingual documents |
| `qwen3-embedding:0.6b` | 600M | 2048 | Newest, general purpose |
| `qwen3-embedding:4b` | 4B | 2560 | Heavy but highest capacity |

> Bigger model != always better scores. Test with your actual data.

## Search Strategies

- **Similarity** -- default, returns closest matches by cosine distance
- **MMR** (Maximal Marginal Relevance) -- balances relevance + diversity, avoids redundant results
  ```python
  retriever = store.as_retriever(
      search_type="mmr",
      search_kwargs={"k": 3, "fetch_k": 10}
  )
  ```

## Score Thresholds

| Score | Quality | Meaning |
|-------|---------|---------|
| 0.7+  | GOOD    | Chunk directly answers the query |
| 0.5-0.7 | OKAY | Related content, may not be precise |
| < 0.5 | BAD     | Likely irrelevant or wrong chunk |

## Common Problems & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Wrong page retrieved | Chunks too large, keyword overlap | Reduce `chunk_size` to 300-500 |
| Low scores across the board | Weak embedding model | Try `mxbai-embed-large` or `snowflake-arctic-embed2` |
| Stale/duplicate results | Old data in Chroma | `vector_store.reset_collection()` before re-adding |
| Slow embedding | Large model + many chunks | Batch inserts, use lighter model |
| Math/tables retrieved poorly | PDF text extraction loses structure | Consider markdown-based loaders or OCR preprocessing |
| Redundant top-k results | All chunks from same section | Switch to MMR search |

## Speed Tips

- Batch document inserts in groups of 10-20
- Use `asyncio.gather` for concurrent queries
- Smaller embedding model = faster at the cost of some accuracy
- Persist Chroma to disk (`persist_directory`) to avoid re-embedding on restart
