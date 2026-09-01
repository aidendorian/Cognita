import json
import psycopg
import trafilatura
from config.env import db_url
from rag.chunking import chunk_text
from rag.embeddings import embed_text

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
        print(f"[ingest] PPStructureV3 returned empty output for {path!r}, trying PaddleOCR")
    except ImportError:
        print("[ingest] PPStructureV3 not available, falling back to PaddleOCR")
    except Exception as exc:
        print(f"[ingest] PPStructureV3 failed for {path!r}: {exc!r}, falling back to PaddleOCR")

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

def _store_chunks(chunks: list[str], project_id: int, source: str, force: bool = False) -> int:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            if not force:
                cur.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE project_id = %s AND source = %s",
                    (project_id, source),
                )
                if cur.fetchone()[0] > 0:  # type: ignore
                    print(f"[ingest] skipping {source!r} — already ingested for project {project_id}")
                    return 0

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


def ingest_url(url: str, project_id: int, force: bool = False) -> int:
    text = _extract_url(url)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"No usable chunks extracted from URL: {url!r}")
    stored = _store_chunks(chunks, project_id, source=url, force=force)
    print(f"[ingest] stored {stored} chunks from {url!r}")
    return stored


def ingest_url_content(content: str, url: str, project_id: int, force: bool = False) -> int:
    chunks = chunk_text(content)
    if not chunks:
        raise ValueError(f"No usable chunks in content for {url!r}")
    stored = _store_chunks(chunks, project_id, source=url, force=force)
    print(f"[ingest] stored {stored} chunks from pre-fetched content {url!r}")
    return stored

def ingest_pdf(path: str, project_id: int, force: bool = False) -> int:
    text = _extract_pdf(path)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"No usable chunks extracted from PDF: {path!r}")
    stored = _store_chunks(chunks, project_id, source=path, force=force)
    print(f"[ingest] stored {stored} chunks from {path!r}")
    return stored

def ingest_text(text: str, project_id: int, source_label: str, force: bool = False) -> int:
    if not source_label:
        raise ValueError("source_label is required for ingest_text — used for deduplication")
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"No usable chunks in text for {source_label!r}")
    stored = _store_chunks(chunks, project_id, source=source_label, force=force)
    print(f"[ingest] stored {stored} chunks from {source_label!r}")
    return stored

"""def ingest(source: str, project_id: int, source_label: str | None = None, source_type: str = "auto", force: bool = False) -> int:

    if source_type == "auto":
        if source.startswith("http://") or source.startswith("https://"):
            source_type = "url"
        elif source.lower().endswith(".pdf"):
            source_type = "pdf"
        else:
            source_type = "text"

    if source_type == "url":
        if source_label is not None:
            return ingest_url_content(source, url=source_label, project_id=project_id, force=force)
        return ingest_url(source, project_id=project_id, force=force)

    if source_type == "url_content":
        if source_label is None:
            raise ValueError("source_label (the URL) is required for source_type='url_content'")
        return ingest_url_content(source, url=source_label, project_id=project_id, force=force)

    if source_type == "pdf":
        return ingest_pdf(source, project_id=project_id, force=force)

    if source_type == "text":
        label = source_label or "raw_text"
        return ingest_text(source, project_id=project_id, source_label=label, force=force)

    raise ValueError(f"Unknown source_type: {source_type!r}")"""