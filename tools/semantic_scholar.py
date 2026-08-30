import httpx

URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_papers(query: str, limit: int = 5) -> list[dict]:
    
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,citationCount,authors,externalIds,paperId"
    }
    try:
        response = httpx.get(url=URL, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        papers = data.get("data", [])

        return [
            {
                "title": p.get("title", ""),
                "abstract": p.get("abstract") or "",
                "year": p.get("year"),
                "citation_count": p.get("citationCount", 0),
                "authors": ", ".join(a["name"] for a in p.get("authors", [])),
                "url": f"https://www.semanticscholar.org/paper/{p['paperId']}",
            }
            for p in papers
        ]
    
    except Exception as e:
        print(f"[semantic_scholar] search failed: {e}")
        return []