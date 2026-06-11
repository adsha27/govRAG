# govRAG

RAG pipeline for government and policy documents. Upload PDFs, ask questions, get answers grounded in the source text.

Built to work with dense regulatory documents where hallucination is not acceptable - every answer cites the chunk it came from.

## What it does

- Ingest PDF documents (direct text extraction + OCR fallback for scanned docs)
- Chunk and embed with sentence-transformers
- Store vectors in pgvector (Postgres) for persistent, queryable storage
- Answer questions via a local LLM (Ollama) with source citations
- Stream responses token by token
- REST API + Swagger UI

## Stack

- Python 3.12, FastAPI
- pgvector for vector storage (persistent, filterable)
- sentence-transformers for embeddings (all-MiniLM-L6-v2, fully local)
- Ollama for LLM inference (qwen3:8b default, swap to anything)
- PyMuPDF for PDF parsing, pytesseract for OCR fallback

## Quick start (local)

```bash
# Start Postgres with pgvector + the API
docker compose up

# Or manually:
pip install -r requirements.txt
docker run -d -e POSTGRES_PASSWORD=govrag -p 5432:5432 pgvector/pgvector:pg16
ollama pull qwen3:8b
uvicorn api.main:app --reload
```

Open http://localhost:8000/docs for the Swagger UI.

## Deploy to Railway (cloud)

Railway supports Docker and managed Postgres with pgvector — the right fit for this stack.

1. Fork this repo and create a new project at [railway.app](https://railway.app)
2. Add a **Postgres** plugin — Railway sets `DATABASE_URL` automatically
3. Set these environment variables in your Railway service:

```
LLM_PROVIDER=groq
GROQ_API_KEY=<your key from console.groq.com — free tier works>
```

4. Deploy. Railway picks up `railway.json` and uses the Dockerfile.

> Ollama can't run on Railway (no GPU, no persistent process). Groq gives you llama-3.3-70b for free via API — same quality, zero setup.

## API

```
POST /ingest/              Upload a PDF
POST /ask/                 Ask a question, get answer + source chunks
GET  /ask/stream/          Same but streams the response
GET  /docs/                List ingested documents
DELETE /docs/{id}/         Remove a document and its chunks
POST /generate_mcq/        Original MCQ generation feature (still works)
```

### Example

```bash
curl -X POST http://localhost:8000/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the eligibility criteria?"}'
```

```json
{
  "answer": "Based on the policy documents...",
  "sources": [
    {"chunk": "...", "doc_name": "policy.pdf", "page": 3, "score": 0.91}
  ]
}
```

## Why pgvector over FAISS

FAISS is in-memory - index resets on restart. pgvector persists to Postgres, supports metadata filtering by document or date, and runs alongside your existing stack. If you already run Postgres, pgvector is zero extra infra.

## Project structure

```
api/             FastAPI app
chunking/        Sentence-aware chunker
embeddings/      Embedding wrapper
ingestion/       PDF ingest with OCR fallback
preprocessing/   Text extraction
storage/         pgvector store
question_generation/  MCQ generator (original)
tests/
docker-compose.yml
```

## Background

Built while working on a production RAG system over 13 government policy PDFs at Right Walk Foundation, where the same pipeline serves real user queries in 5 languages. This is the open, standalone version.
