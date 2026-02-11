# Caching Responses RAG

This module demonstrates response caching in a RAG pipeline using three backends:

- `a_in_memory/with_openai.py`
- `b_postgres/with_openai.py`
- `c_redis/with_openai.py`

## Prerequisites

- `OPENAI_API_KEY` is set
- Dependencies installed:

```bash
pip install -r requirements.txt
```

## Run Examples

### 1) In-Memory Cache

```bash
python3 5_caching_responses_rag/a_in_memory/with_openai.py
```

### 2) PostgreSQL Cache

Set:

- `POSTGRES_CACHE_URL` (SQLAlchemy format, example: `postgresql+psycopg://user:pass@localhost:5432/dbname`)
- Optional: `CACHE_KEY_PREFIX` (default: `rag_cache`)

Run:

```bash
export POSTGRES_CACHE_URL='postgresql+psycopg://user:pass@localhost:5432/dbname'
python3 5_caching_responses_rag/b_postgres/with_openai.py
```

### 3) Redis Cache

Set:

- `REDIS_CACHE_URL` (example: `redis://localhost:6379/0`)
- Optional: `CACHE_KEY_PREFIX` (default: `rag_cache`)
- Optional: `REDIS_TTL_SECONDS` (default: `0`, no expiration)

Run:

```bash
export REDIS_CACHE_URL='redis://localhost:6379/0'
export REDIS_TTL_SECONDS=3600
python3 5_caching_responses_rag/c_redis/with_openai.py
```

## What to Observe

- First query: full retrieval + generation latency
- Repeated query with same retrieved context: `[CACHED]` result and lower total latency
