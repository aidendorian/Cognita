def chunk_text(text: str, chunk_size: int = 2048, overlap: int = 200) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) >= 100:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks