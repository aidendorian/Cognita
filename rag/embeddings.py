from google import genai
from config.env import api_key
from google.genai import types

EMBEDDING_MODEL = "gemini-embedding-001"

client = genai.Client(api_key=api_key)

def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )
    return result.embeddings[0].values # type: ignore