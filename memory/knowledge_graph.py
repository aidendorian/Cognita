import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from config.env import neo4j_uri, neo4j_user, neo4j_password, api_key, llm_model
from graphiti_core.nodes import EpisodeType

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_graphiti = None

def _start_background_loop() -> asyncio.AbstractEventLoop:
    global _loop_thread
    loop = asyncio.new_event_loop()

    def run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    thread.start()
    _loop_thread = thread
    return loop

def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = _start_background_loop()
    return _loop

def _run_async(coro):
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=60)

def _get_graphiti():
    global _graphiti
    if _graphiti is not None:
        return _graphiti

    llm_config = LLMConfig(
        api_key=api_key,
        model=llm_model,
        temperature=0.0,
    )

    embedder_config = GeminiEmbedderConfig(
        api_key=api_key,
        embedding_model="gemini-embedding-001",
    )

    _graphiti = Graphiti(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        llm_client=GeminiClient(config=llm_config),
        embedder=GeminiEmbedder(config=embedder_config),
        cross_encoder=GeminiRerankerClient(config=llm_config)
    )
    print("[graphiti] initialized with native Gemini backend")
    return _graphiti

def setup_graphiti() -> None:
    async def _setup():
        g = _get_graphiti()
        await g.build_indices_and_constraints()
        print("[graphiti] indices and constraints built")

    try:
        _run_async(_setup())
    except Exception as e:
        print(f"[graphiti] setup failed: {e}")
        raise


def add_research_episode(project_id: int, content: str, source: str = "researcher", task: str = "") -> None:
    if not content or not content.strip():
        return

    async def _add():
        g = _get_graphiti()
        await g.add_episode(
            name=f"project_{project_id}_{source}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            episode_body=content,
            source=EpisodeType.text,
            reference_time=datetime.now(timezone.utc),
            source_description=f"source={source} task={task[:100]}",
            group_id=f"project_{project_id}",  # project isolation
        )

    try:
        _run_async(_add())
        print(f"[graphiti] episode added — project {project_id}, source={source}")
    except Exception as e:
        print(f"[graphiti] add_episode failed (non-fatal): {e}")

def search_knowledge_graph(query: str, project_id: int, num_results: int = 5) -> list[dict]:
    async def _search() -> list:
        g = _get_graphiti()
        results = await g.search(
            query=query,
            num_results=num_results,
            group_ids=[f"project_{project_id}"],
        )
        return results

    try:
        raw = _run_async(_search())
        return [
            {
                "fact": getattr(r, "fact", str(r)),
                "name": getattr(r, "name", ""),
                "uuid": str(getattr(r, "uuid", "")),
                "source_node_uuid": str(getattr(r, "source_node_uuid", "")),
                "target_node_uuid": str(getattr(r, "target_node_uuid", "")),
                "valid_at": getattr(r, "valid_at", None),
                "invalid_at": getattr(r, "invalid_at", None),
            }
            for r in raw
        ]
    except Exception as e:
        print(f"[graphiti] search failed (non-fatal): {e}")
        return []

def shutdown_graphiti() -> None:
    global _loop, _graphiti
    if _loop is not None and not _loop.is_closed():
        _loop.call_soon_threadsafe(_loop.stop)
        time.sleep(0.5)
    if _graphiti is not None:
        try:
            _run_async(_graphiti.close())
        except Exception:
            pass
    print("[graphiti] shut down gracefully")