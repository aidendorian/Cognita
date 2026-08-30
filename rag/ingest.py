import json
import psycopg
import trafilatura

from config.env import db_url
from rag.chunking import chunk_text
from rag.embeddings import embed_text

def _detect_type(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        return "url"
    if source.lower().endswith(".pdf"):
        return "pdf"
    return "text"

def _extract_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch URL: {url}")
    text = trafilatura.extract(downloaded)
    if not text:
        raise ValueError(f"Could not extract content from: {url}")
    return text

def _extract_pdf(path: str) -> str:
    try:
        from paddleocr import PPStructureV3
        pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        pages = []
        for res in pipeline.predict(path):
            if hasattr(res, "res") and isinstance(res.res, dict):
                for block in res.res.get("layout_result", {}).get("boxes", []):
                    text = block.get("text", "")
                    if text:
                        pages.append(text)
            elif hasattr(res, "to_markdown"):
                pages.append(res.to_markdown())

        text = "\n\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en")
    result = ocr.ocr(path)
    lines = []
    for page in result:
        if page:
            for line in page:
                lines.append(line[1][0])
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError(f"PaddleOCR extracted no text from: {path}")
    return text

def _store_chunks(chunks: list[str], project_id: int, source: str) -> int:
    with psycopg.connect(db_url) as conn: #type: ignore
        with conn.cursor() as cur:
            for chunk in chunks:
                vector = embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")
                vector_str = json.dumps(vector)
                cur.execute(
                    """
                    INSERT INTO embeddings (project_id, chunk_text, embedding, source)
                    VALUES (%s, %s, %s::vector, %s)
                    """,
                    (project_id, chunk, vector_str, source),
                )
        conn.commit()
    return len(chunks)

def ingest(source: str, project_id: int, source_label:str | None = None, source_type: str = "auto") -> int:
    if source_type == "auto":
        source_type = _detect_type(source)

    if source_type == "url" and source_label is None:
        text = _extract_url(source)
        label = source
    elif source_type == "url" and source_label is not None:
        text = source
        label = source_label
    elif source_type == "pdf":
        text = _extract_pdf(source)
        label = source
    elif source_type == "text":
        text = source
        label = "raw_text"
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"No usable chunks extracted from: {source!r}")

    stored = _store_chunks(chunks, project_id, label)
    print(f"[ingest] stored {stored} chunks from {label!r}")
    return stored