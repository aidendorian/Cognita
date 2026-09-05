import uuid
import threading
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from psycopg_pool import ConnectionPool
from memory.conversation import create_chat, save_messages
from memory.project import save_project_memory
from app.dependencies import validate_api_key, limiter
from graphs.checkpointer import get_checkpointer
from memory.knowledge_graph import shutdown_graphiti
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from memory.knowledge_graph import setup_graphiti, shutdown_graphiti
from app.dependencies import get_pool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_active_threads: dict[str, threading.Thread] = {}
_active_threads_lock = threading.Lock()

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_MIME_TYPES = {"application/pdf"}

_pool: Optional[ConnectionPool] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_checkpointer()
    get_pool()
    setup_graphiti()
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE runs SET status = 'interrupted', error = 'Server restarted'
                WHERE status IN ('running', 'pending')
            """)
        conn.commit()
    try:
        yield
    finally:
        shutdown_graphiti()
        global _pool
        if _pool is not None:
            _pool.close()
            _pool = None

app = FastAPI(
    title="ResearchPlatform API",
    lifespan=lifespan,
    description="Multi-agent research platform — create projects, run workflows, get reports",
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) #type: ignore

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="app"), name="app")

@app.get("/ui")
async def ui():
    return FileResponse("app/index.html")

class ProjectCreate(BaseModel):
    name: str

class RunRequest(BaseModel):
    task: str

def _project_exists(project_id: int) -> bool:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM projects WHERE id = %s",
                (project_id,),
            )
            return cur.fetchone() is not None

def _create_run(project_id: int, task: str) -> tuple[str, str]:
    """Create the durable run record before starting the worker."""
    run_id = str(uuid.uuid4())
    thread_id = f"run-{run_id}"

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (id, project_id, thread_id, task, current_agent, status, config) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (run_id, project_id, thread_id, task, "initializing", "pending", "{}"),
            )
        conn.commit()

    return run_id, thread_id

def _update_run(run_id: str, *, status: Optional[str] = None, current_agent: Optional[str] = None, final_output: Optional[str] = None, error: Optional[str] = None) -> None:
    """Update a run using the current run-centric schema."""
    updates = []
    params = []

    if status is not None:
        updates.append("status = %s")
        params.append(status)

        if status == "running":
            updates.append("started_at = COALESCE(started_at, NOW())")

        if status in {"completed", "failed", "cancelled"}:
            updates.append("completed_at = COALESCE(completed_at, NOW())")

    if current_agent is not None:
        updates.append("current_agent = %s")
        params.append(current_agent)

    if final_output is not None:
        updates.append("final_output = %s")
        params.append(final_output)

    if error is not None:
        updates.append("error = %s")
        params.append(error)

    if not updates:
        return

    updates.append("updated_at = NOW()")
    params.append(run_id)

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE runs
                SET {", ".join(updates)}
                WHERE id = %s
                """,
                tuple(params),
            )
        conn.commit()

def _get_run_by_id(run_id: str) -> Optional[dict]:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_id, thread_id, task, current_agent, config, status, final_output, error, created_at, started_at, completed_at, updated_at
                FROM runs
                WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()

    if not row:
        return None

    return {
        "run_id": str(row[0]),
        "project_id": row[1],
        "thread_id": row[2],
        "task": row[3],
        "current_agent": row[4],
        "config": row[5],
        "status": row[6],
        "final_output": row[7],
        "error": row[8],
        "created_at": str(row[9]),
        "started_at": str(row[10]) if row[10] else None,
        "completed_at": str(row[11]) if row[11] else None,
        "updated_at": str(row[12]),
    }


def _get_latest_run(project_id: int) -> Optional[dict]:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM runs
                WHERE project_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (project_id,),
            )
            row = cur.fetchone()

    if not row:
        return None

    return _get_run_by_id(str(row[0]))

def _run_workflow_background(project_id: int, run_id: str, thread_id: str, task: str) -> None:
    try:
        from graphs.graph import get_app

        _update_run(
            run_id,
            status="running",
            current_agent="initializing",
        )

        graph_app = get_app()

        initial_state = {
            "project_id": project_id,
            "run_id": run_id,
            "task": task,
            "plan": None,
            "current_step": None,
            "research_output": None,
            "research_summary": None,
            "code_output": None,
            "code_result": None,
            "analysis": None,
            "analysis_summary": None,
            "draft": None,
            "review_notes": None,
            "final_output": None,
            "next_agent": None,
            "messages": [],
            "status": None,
            "revision_count": 0,
            "output_files": None,
            "novelty_report": None,
            "task_mode": None,
        }

        result = graph_app.invoke(
            initial_state,  # type: ignore[arg-type]
            config={
                "recursion_limit": 50,
                "configurable": {"thread_id": thread_id},
            },
        )

        messages = result.get("messages", [])
        if messages:
            chat_id = create_chat(project_id)
            save_messages(chat_id, messages)

        save_project_memory(project_id, result)  # type: ignore[arg-type]

        final_output = result.get("final_output")
        graph_status = result.get("status")

        run = _get_run_by_id(run_id)
        if run and run["status"] != "cancelled":
            terminal_status = "needs_review" if graph_status == "needs_review" else "completed"
            _update_run(
                run_id,
                status=terminal_status,
                current_agent="completed",
                final_output=final_output,
            )

    except Exception as exc:
        try:
            run = _get_run_by_id(run_id)
            if run and run["status"] != "cancelled":
                _update_run(
                    run_id,
                    status="failed",
                    current_agent="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
        except Exception:
            pass
    with _active_threads_lock:
        _active_threads.pop(run_id, None)

@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    return {
        "name": "ResearchPlatform API",
        "version": "0.1.0",
        "status": "running",
    }

@app.post("/projects", status_code=201)
@limiter.limit("5/minute")
async def create_project(request: Request, body: ProjectCreate, api_key: str = Depends(validate_api_key)):
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO projects (name)
                VALUES (%s)
                RETURNING id, name, created_at
                """,
                (body.name,),
            )
            row = cur.fetchone()
        conn.commit()

    return {
        "id": row[0], #type: ignore
        "name": row[1], #type: ignore
        "created_at": str(row[2]), #type: ignore
    }

@app.get("/projects")
@limiter.limit("30/minute")
async def list_projects(request: Request, api_key: str = Depends(validate_api_key)):
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, created_at
                FROM projects
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "name": row[1],
            "created_at": str(row[2]),
        }
        for row in rows
    ]

@app.get("/projects/{project_id}/runs")
@limiter.limit("30/minute")
async def list_runs(request: Request, project_id: int, api_key: str = Depends(validate_api_key)):
    if not _project_exists(project_id):
        raise HTTPException(404, "Project not found")

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, task, status, current_agent, error, created_at, started_at, completed_at
                FROM runs
                WHERE project_id = %s
                ORDER BY created_at DESC
                """,
                (project_id,),
            )
            rows = cur.fetchall()

    return [
        {
            "run_id": str(row[0]),
            "task": row[1],
            "status": row[2],
            "current_agent": row[3],
            "error": row[4],
            "created_at": str(row[5]),
            "started_at": str(row[6]) if row[6] else None,
            "completed_at": str(row[7]) if row[7] else None,
        }
        for row in rows
    ]

@app.post("/projects/{project_id}/run", status_code=202)
@limiter.limit("2/minute")
async def run_workflow(request: Request, project_id: int, body: RunRequest, api_key: str = Depends(validate_api_key)):
    if not _project_exists(project_id):
        raise HTTPException(404, "Project not found")

    task = body.task.strip()
    if not task:
        raise HTTPException(400, "Task cannot be empty")

    run_id, thread_id = _create_run(project_id, task)

    thread = threading.Thread(
        target=_run_workflow_background,
        args=(project_id, run_id, thread_id, task),
        name=f"research-run-{run_id[:8]}",
        daemon=True,
    )
    
    with _active_threads_lock:
            _active_threads[run_id] = thread
    
    thread.start()
    
    return {
        "run_id": run_id,
        "project_id": project_id,
        "status": "pending",
    }
    
@app.post("/runs/{run_id}/cancel", status_code=200)
@limiter.limit("10/minute")
async def cancel_run(request: Request, run_id: str, api_key: str = Depends(validate_api_key)):
    run = _get_run_by_id(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    if run["status"] not in {"pending", "running"}:
        raise HTTPException(400, f"Run is already {run['status']} — cannot cancel")

    _update_run(run_id, status="cancelled", error="Cancelled by user")

    with _active_threads_lock:
        _active_threads.pop(run_id, None)

    return {"run_id": run_id, "status": "cancelled"}

@app.get("/runs/{run_id}")
@limiter.limit("30/minute")
async def get_run_status(request: Request, run_id: str, api_key: str = Depends(validate_api_key)):
    run = _get_run_by_id(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.get("/projects/{project_id}/status")
@limiter.limit("30/minute")
async def get_project_status(request: Request, project_id: int, api_key: str = Depends(validate_api_key), run_id: Optional[str] = None):
    if not _project_exists(project_id):
        raise HTTPException(404, "Project not found")

    if run_id:
        run = _get_run_by_id(run_id)

        if not run or run["project_id"] != project_id:
            raise HTTPException(404, "Run not found for this project")

        return run

    run = _get_latest_run(project_id)

    if not run:
        raise HTTPException(404, "No runs found for this project")

    return run

def _report_path(project_id: int, run_id: str) -> Path:
    return (
        Path("data")
        / "projects"
        / str(project_id)
        / "runs"
        / run_id
        / "outputs"
        / "final_report.md"
    )


def _draft_paths(project_id: int, run_id: str) -> list[str]:
    output_dir = _report_path(project_id, run_id).parent

    if not output_dir.is_dir():
        return []

    return [
        str(path.relative_to(Path("data") / "projects" / str(project_id) / "runs" / run_id))
        for path in sorted(output_dir.glob("draft_v*.md"))
        if path.is_file()
    ]

@app.get("/runs/{run_id}/report")
@limiter.limit("20/minute")
async def get_run_report(request: Request, run_id: str, api_key: str = Depends(validate_api_key)):
    run = _get_run_by_id(run_id)

    if not run:
        raise HTTPException(404, "Run not found")

    project_id = run["project_id"]
    report_path = _report_path(project_id, run_id)

    if not report_path.is_file():
        drafts = _draft_paths(project_id, run_id)

        if drafts:
            return {
                "project_id": project_id,
                "run_id": run_id,
                "status": "in_progress",
                "message": "No final report yet",
                "drafts_available": drafts,
            }

        if run["status"] in {"pending", "running"}:
            raise HTTPException(404, "No report yet — workflow is still in progress")

        raise HTTPException(404, "No report found for this run")

    return FileResponse(
        path=str(report_path),
        media_type="text/markdown",
        filename=f"report_run_{run_id}.md",
    )

@app.get("/projects/{project_id}/report")
@limiter.limit("20/minute")
async def get_latest_project_report(request: Request, project_id: int, api_key: str = Depends(validate_api_key)):
    if not _project_exists(project_id):
        raise HTTPException(404, f"Project {project_id} not found")

    run = _get_latest_run(project_id)

    if not run:
        raise HTTPException(404, "No runs found for this project")

    if run["status"] != "completed":
        raise HTTPException(
            409,
            f"Latest run is not completed (status: {run['status']})",
        )

    return await get_run_report(
        request=request,
        run_id=run["run_id"],
        api_key=api_key,
    )

@app.post("/projects/{project_id}/ingest")
@limiter.limit("10/minute")
async def ingest_source(request: Request, project_id: int, api_key: str = Depends(validate_api_key), url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    if not _project_exists(project_id):
        raise HTTPException(404, f"Project {project_id} not found")

    if url and file:
        raise HTTPException(400, "Pass either url or file, not both")

    if not url and not file:
        raise HTTPException(400, "Pass either url or file")

    if url:
        from rag.ingest import ingest_url

        try:
            chunks = ingest_url(
                url=url.strip(),
                project_id=project_id,
            )
            return {
                "project_id": project_id,
                "source": url,
                "chunks_stored": chunks,
            }
        except Exception as exc:
            raise HTTPException(
                500,
                f"Ingestion failed: {type(exc).__name__}: {exc}",
            ) from exc

    assert file is not None

    filename = Path(file.filename or "upload.pdf").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are supported")

    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            415,
            f"Unsupported MIME type. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            413,
            f"File too large. Max size: {MAX_FILE_SIZE} bytes",
        )

    from rag.ingest import ingest_pdf
    from tools.filesystem import FileSystem
    
    fs = FileSystem(project_id=project_id)
    relative_path = f"sources/pdfs/{filename}"

    try:
        fs.write_bytes(relative_path, content)

        full_path = str(
            Path("data")
            / "projects"
            / str(project_id)
            / relative_path
        )

        chunks = ingest_pdf(
            path=full_path,
            project_id=project_id,
        )

        return {
            "project_id": project_id,
            "source": relative_path,
            "chunks_stored": chunks,
        }

    except Exception as exc:
        raise HTTPException(
            500,
            f"PDF ingestion failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        await file.close()

@app.get("/projects/{project_id}/memory")
@limiter.limit("30/minute")
async def get_memory(request: Request, project_id: int, api_key: str = Depends(validate_api_key)):
    if not _project_exists(project_id):
        raise HTTPException(404, "Project not found")

    from memory.project import load_project_memory

    prior = load_project_memory(project_id)

    return {
        "project_id": project_id,
        "research_summaries": len(prior["research_summaries"]),
        "analysis_summaries": len(prior["analysis_summaries"]),
        "prior_reports": len(prior["prior_reports"]),
        "critic_reports": len(prior["critic_reports"]),
        "most_recent_research": (
            prior["research_summaries"][-1][:300]
            if prior["research_summaries"]
            else None
        ),
        "most_recent_analysis": (
            prior["analysis_summaries"][-1][:300]
            if prior["analysis_summaries"]
            else None
        ),
    }