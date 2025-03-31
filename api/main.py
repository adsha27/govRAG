from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os

# FastAPI CORS middleware
from fastapi.middleware.cors import CORSMiddleware

# Local imports for your modules
from ingestion.pdf_ingest import ingest_pdf
from chunking.chunker import chunk_text
from embeddings.embedder import Embedder
from index.indexer import VectorIndexer
from question_generation.mcq_generator import MCQGenerator

# Initialize FastAPI app
app = FastAPI(
    title="GovRAG API",
    description="API for document ingestion, semantic search, and MCQ generation.",
)

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load backend components
embedder = Embedder()
indexer = VectorIndexer(embedding_dim=384)
mcq_generator = MCQGenerator(model_name="llama3")


@app.get("/")
def root():
    return {"message": "✅ GovRAG API is live. Visit /docs for Swagger."}


@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    text = ingest_pdf(temp_file)
    os.remove(temp_file)

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)
    indexer.add_bulk(embeddings, chunks)

    return {
        "message": "Document ingested and indexed successfully.",
        "num_chunks": len(chunks)
    }


@app.get("/search/")
def search(query: str, k: int = 5):
    query_embedding = embedder.encode([query])[0]
    distances, results = indexer.search(query_embedding, k)
    return {
        "query": query,
        "results": results,
        "distances": distances.tolist()
    }


@app.post("/generate_mcq/")
def generate_mcq(prompt: str):
    mcq = mcq_generator.generate_mcq(prompt)
    return {"prompt": prompt, "mcq": mcq}


@app.post("/upload_and_generate_mcqs/")
async def upload_and_generate_mcqs(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    text = ingest_pdf(temp_file)
    os.remove(temp_file)

    chunks = chunk_text(text)

    mcq_list = []
    for i, chunk in enumerate(chunks):
        if i >= 10:
            break

        chunk = chunk[:1000]  # Limit for faster response + shorter prompt
        print(f"[INFO] Generating MCQ {i + 1}/10...")
        mcq = mcq_generator.generate_mcq(chunk)
        mcq_list.append({"chunk_index": i, "mcq": mcq})

    return {
        "num_chunks": len(chunks),
        "mcqs": mcq_list
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)