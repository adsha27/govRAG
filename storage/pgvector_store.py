"""
pgvector-backed vector store.
Replaces FAISS. Persists across restarts, supports metadata filtering.
"""

import json
import os
from dataclasses import dataclass

import psycopg2
import psycopg2.extras

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:govrag@localhost:5432/govrag")


@dataclass
class Chunk:
    id: int
    doc_id: int
    doc_name: str
    page: int
    text: str
    score: float = 0.0


def _connect():
    return psycopg2.connect(DB_URL)


def init_db():
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    doc_name TEXT NOT NULL,
                    page INTEGER DEFAULT 0,
                    text TEXT NOT NULL,
                    embedding vector(384)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
        conn.commit()


def add_document(name: str) -> int:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (name) VALUES (%s) RETURNING id", (name,))
            doc_id = cur.fetchone()[0]
        conn.commit()
    return doc_id


def add_chunks(doc_id: int, doc_name: str, chunks: list[str], embeddings, pages: list[int] | None = None):
    if pages is None:
        pages = [0] * len(chunks)

    with _connect() as conn:
        with conn.cursor() as cur:
            for text, emb, page in zip(chunks, embeddings, pages):
                emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                cur.execute(
                    "INSERT INTO chunks (doc_id, doc_name, page, text, embedding) VALUES (%s, %s, %s, %s, %s)",
                    (doc_id, doc_name, page, text, json.dumps(emb_list)),
                )
        conn.commit()


def search(query_embedding, k: int = 5, doc_id: int | None = None) -> list[Chunk]:
    emb_list = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
    emb_str = json.dumps(emb_list)

    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if doc_id is not None:
                cur.execute("""
                    SELECT id, doc_id, doc_name, page, text,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE doc_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (emb_str, doc_id, emb_str, k))
            else:
                cur.execute("""
                    SELECT id, doc_id, doc_name, page, text,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (emb_str, emb_str, k))

            rows = cur.fetchall()

    return [
        Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            doc_name=row["doc_name"],
            page=row["page"],
            text=row["text"],
            score=float(row["score"]),
        )
        for row in rows
    ]


def list_documents() -> list[dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT d.id, d.name, d.created_at,
                       COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.doc_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
            """)
            return [dict(r) for r in cur.fetchall()]


def delete_document(doc_id: int):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()
