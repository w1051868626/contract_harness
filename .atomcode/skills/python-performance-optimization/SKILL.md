---
name: python-performance-optimization
description: Profile and optimize Python hot paths — cProfile/py-spy sampling, batch LLM calls, regex precompilation, deterministic hashing, vectorized NumPy, and async I/O. Use when editing performance-sensitive modules (batch_check, embed_batch, RRF fusion, Multi-Agent coordination) or after profiling reveals bottlenecks.
---

# Python Performance Optimization

You are a Python performance optimization specialist. Apply these practices when editing hot paths in this contract-review Agent system.

## Core Workflow

1. **Measure first** — never optimize blind. Reach for `cProfile` for CPU, `py-spy record` for sampling in prod, `memory_profiler` for allocations.
2. **Identify the bottleneck** — is it LLM I/O, embedding API calls, regex compilation, dict/list reshuffles, or ChromaDB queries?
3. **Apply the smallest fix that moves the needle** — batch, cache, precompile, vectorize, or parallelize.
4. **Re-measure** — confirm the win and that nothing regressed.

## Patterns for This Codebase

### Batch LLM calls
`ComplianceChecker.batch_check` and `RiskAnalyzer._parse_batch_response` already merge per-clause calls into one. When adding new LLM-driven checks, batch them — N round-trips at ~1s each dominates total latency.

### Regex precompilation
`harness/rag/chunking.py` compiles heading patterns at module load. Never call `re.compile` inside a hot loop; hoist to module scope or `__init__`.

### Deterministic hashing
`hash_embed` uses deterministic hashing so identical inputs embed identically (cache-friendly). Don't introduce `random()` or `uuid4()` into embedding/cache keys.

### Embedding rate limiting
`OpenAIEmbeddingProvider` integrates `RateLimiter` (RPM + TPM sliding window). New providers must respect `EMBEDDING_MAX_RPM` / `EMBEDDING_MAX_TPM` or cost will spike.

### RRF fusion
`rrf_fuse` merges dense + BM25 sparse results. Keep `rrf_k` configurable; the default 60 fits most corpora but legal text with long clauses may need tuning.

### ChromaDB on Windows
`ChromaVectorStore.close()` must release the sqlite3 connection (ChromaDB 1.5+ `client.close()`) or Windows `TemporaryDirectory` cleanup throws `PermissionError [WinError 32]`.

## Anti-patterns to Flag

- `for x in items: llm.chat(...)` — not batched
- `re.compile(...)` inside a function called per-chunk
- `embedding(text)` without checking the rate limiter
- `time.sleep()` as a rate-limit strategy (use `RateLimiter`)
- N+1 ChromaDB queries where one batched `get` would do
- `list.index()` / `in` scans on large lists — suggest `dict`/`set`

## Verification

After any optimization, run:
```bash
pytest tests/ -x -q          # no regressions
ruff check harness/ tests/   # style clean
pyright harness/             # types clean
```

If the change touches RAG retrieval or eval scoring, also run the relevant `test_rag.py` / `test_eval.py` cases explicitly.
