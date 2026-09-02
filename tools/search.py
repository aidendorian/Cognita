import logging
from typing import Any
from tavily import TavilyClient
from config.env import tavily_key

logger = logging.getLogger(__name__)

_client: TavilyClient | None = None

def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not tavily_key:
            raise RuntimeError("Tavily API key is not configured.")
        _client = TavilyClient(api_key=tavily_key)
    return _client

def search(query: str, project_id: int, run_id: str | None = None, max_results: int = 5) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if project_id <= 0:
        raise ValueError("project_id must be positive")

    if max_results <= 0:
        raise ValueError("max_results must be positive")

    if run_id is not None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string when provided")

    response = _get_client().search(
        query=query.strip(),
        max_results=max_results,
    )

    results = response.get("results", [])

    if not isinstance(results, list):
        logger.warning("Tavily returned an invalid results payload.")
        return []

    unique_results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for result in results:
        if not isinstance(result, dict):
            continue
        url = result.get("url", "")

        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)

        unique_results.append(result)
    return unique_results