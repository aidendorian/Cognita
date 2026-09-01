import asyncio
from datetime import datetime, timezone
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from config.env import neo4j_uri, neo4j_user, neo4j_password

def _get_graphiti() -> Graphiti:
    return Graphiti(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )

def setup_graphiti() -> None:
    async def _setup():
        g = _get_graphiti()
        await g.build_indices_and_constraints()
        await g.close()
        print("[graphiti] indices and constraints built")

    asyncio.run(_setup())

def add_research_episode(project_id: int, content: str, source: str = "researcher", task: str = "") -> None:
    if not content or not content.strip():
        return

    async def _add():
        g = _get_graphiti()
        try:
            await g.add_episode(
                name=f"project_{project_id}_{source}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                episode_body=content,
                source=EpisodeType.text,
                reference_time=datetime.now(timezone.utc),
                source_description=f"project_id={project_id} source={source} task={task[:100]}",
                group_id=str(project_id),
            )
        finally:
            await g.close()

    try:
        asyncio.run(_add())
        print(f"[graphiti] episode added for project {project_id} from {source}")
    except Exception as e:
        print(f"[graphiti] failed to add episode: {e}")

def search_knowledge_graph(query: str, project_id: int, num_results: int = 5) -> list[dict]:
    async def _search() -> list:
        g = _get_graphiti()
        try:
            results = await g.search(
                query=query,
                num_results=num_results,
                group_ids=[str(project_id)],
            )
            return results
        finally:
            await g.close()

    try:
        raw_results = asyncio.run(_search())
        if not raw_results:
            return []
            
        return [
            {
                "fact": getattr(r, "fact", str(r)),
                "context": getattr(r, "source_description", ""),
                "created_at": str(getattr(r, "created_at", "")),
            }
            for r in raw_results
        ]
    except Exception as e:
        print(f"[graphiti] search failed: {e}")
        return []