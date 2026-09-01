from tavily import TavilyClient
from config.env import tavily_key
from rag.ingest import ingest_url_content

client = TavilyClient(api_key=tavily_key)

def search(query: str, project_id: int, max_results: int = 5) -> list[dict]:
    response = client.search(query, max_results=max_results)
    results = response["results"]

    seen_urls: set[str] = set()
    for result in results:
        url = result.get("url", "")
        content = result.get("content", "")
        if url and content and url not in seen_urls:
            seen_urls.add(url)
            try:
                ingest_url_content(content, url=url, project_id=project_id)
            except Exception as e:
                print(f"[search] failed to ingest {url}: {e}")
    return results