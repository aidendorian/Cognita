import json
import psycopg
from config.env import db_url
from rag.embeddings import embed_text

def _vector_search(cur, query_vector: list[float], project_id: int, top_k: int) -> list[tuple]:
    vector_str = json.dumps(query_vector)
    cur.execute(
        """
        SELECT id, chunk_text, source
        FROM embeddings
        WHERE project_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,(project_id, vector_str, top_k),
    )
    return cur.fetchall()

def _bm25_search(cur, query: str, project_id: int, top_k: int) -> list[tuple]:
    cur.execute(
        """
        SELECT id, chunk_text, source
        FROM embeddings
        WHERE project_id = %s
          AND chunk_tsv @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(chunk_tsv, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """,(project_id, query, query, top_k),
    )
    return cur.fetchall()

def _rrf(vector_rows: list[tuple], bm25_rows: list[tuple], top_k: int, k: int = 60) -> list[dict]:
    scores: dict[int, float] = {}
    texts: dict[int, str] = {}
    sources: dict[int, str] = {}

    for rank, (chunk_id, chunk_text, source) in enumerate(vector_rows):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + k)
        texts[chunk_id] = chunk_text
        sources[chunk_id] = source or ""

    for rank, (chunk_id, chunk_text, source) in enumerate(bm25_rows):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + k)
        texts[chunk_id] = chunk_text
        sources[chunk_id] = source or ""

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "chunk_id": chunk_id,
            "chunk_text": texts[chunk_id],
            "source": sources[chunk_id],
            "score": round(score, 6),
        }
        for chunk_id, score in ranked[:top_k]
    ]


def retrieve(query: str, project_id: int, top_k: int = 5) -> list[dict]:
    query_vector = embed_text(query, task_type="RETRIEVAL_QUERY")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            vector_rows = _vector_search(cur, query_vector, project_id, top_k * 2)
            bm25_rows = _bm25_search(cur, query, project_id, top_k * 2)

    return _rrf(vector_rows, bm25_rows, top_k)