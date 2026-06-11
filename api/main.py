import os
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from api import llm
from ingestion.pdf_ingest import ingest_pdf
from chunking.chunker import chunk_text_sentences, chunk_text
from embeddings.embedder import Embedder
from storage.pgvector_store import (
    init_db, add_document, add_chunks, search,
    list_documents, delete_document
)
from question_generation.mcq_generator import MCQGenerator

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

app = FastAPI(
    title="govRAG",
    description="RAG over government and policy documents. Upload PDFs, ask questions, get cited answers.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = Embedder()
mcq_generator = MCQGenerator(model_name=OLLAMA_MODEL)


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0", "docs": "/docs"}


@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Upload a PDF file")

    tmp_path = f"/tmp/upload_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        result = ingest_pdf(tmp_path)
        if isinstance(result, tuple):
            text, pages = result
        else:
            text, pages = result, None
    finally:
        os.remove(tmp_path)

    doc_id = add_document(file.filename)

    try:
        chunks, chunk_pages = chunk_text_sentences(text, pages)
    except Exception:
        chunks = chunk_text(text)
        chunk_pages = [0] * len(chunks)

    embeddings = embedder.encode(chunks)
    add_chunks(doc_id, file.filename, chunks, embeddings, chunk_pages)

    return {"doc_id": doc_id, "doc_name": file.filename, "num_chunks": len(chunks)}


@app.get("/docs/")
def list_docs():
    return list_documents()


@app.delete("/docs/{doc_id}/")
def delete_doc(doc_id: int):
    delete_document(doc_id)
    return {"deleted": doc_id}


class AskRequest(BaseModel):
    question: str
    doc_id: int | None = None
    k: int = 5


@app.post("/ask/")
async def ask(req: AskRequest):
    query_emb = embedder.encode([req.question])[0]
    chunks = search(query_emb, k=req.k, doc_id=req.doc_id)

    if not chunks:
        raise HTTPException(404, "No relevant chunks found. Ingest documents first.")

    context = "\n\n".join(
        f"[{c.doc_name}, p.{c.page}]\n{c.text}" for c in chunks
    )
    prompt = _build_prompt(req.question, context)
    answer = await _call_llm(prompt)

    return {
        "answer": answer,
        "sources": [
            {"chunk": c.text[:200], "doc_name": c.doc_name, "page": c.page, "score": round(c.score, 3)}
            for c in chunks
        ],
    }


@app.get("/ask/stream/")
async def ask_stream(question: str, doc_id: int | None = None, k: int = 5):
    query_emb = embedder.encode([question])[0]
    chunks = search(query_emb, k=k, doc_id=doc_id)

    if not chunks:
        raise HTTPException(404, "No relevant chunks found")

    context = "\n\n".join(f"[{c.doc_name}, p.{c.page}]\n{c.text}" for c in chunks)
    prompt = _build_prompt(question, context)

    async def generate() -> AsyncGenerator[str, None]:
        async for token in llm.stream(prompt):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/generate_mcq/")
def generate_mcq(prompt: str):
    mcq = mcq_generator.generate_mcq(prompt)
    return {"prompt": prompt, "mcq": mcq}


def _build_prompt(question: str, context: str) -> str:
    return f"""Answer based only on the document excerpts below.
If the answer is not in the excerpts, say that directly.
Do not make up information. Cite source and page when relevant.

Excerpts:
{context}

Question: {question}

Answer:"""


async def _call_llm(prompt: str) -> str:
    return await llm.generate(prompt)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
