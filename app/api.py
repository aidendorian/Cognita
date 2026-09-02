import uuid
import threading
from typing import Optional
from pathlib import Path
from memory.conversation import create_chat, save_messages
from memory.project import save_project_memory
import psycopg
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.runs import upsert_run, get_run
from config.env import db_url
from tools.filesystem import FileSystem
from contextlib import asynccontextmanager
from graphs.checkpointer import get_checkpointer
from memory.knowledge_graph import shutdown_graphiti

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_MIME_TYPES = {"application/pdf"}

@asynccontextmanager
async def lifespan(app):
    get_checkpointer()
    yield
    shutdown_graphiti()

app = FastAPI(
    title="ResearchPlatform API",
    lifespan=lifespan,
    description="Multi-agent research platform — create projects, run workflows, get reports",
    version="0.1.0",
)

class ProjectCreate(BaseModel):
    name: str

class RunRequest(BaseModel):
    task: str

class IngestURLRequest(BaseModel):
    url: str

def _get_conn():
    return psycopg.connect(db_url) #type: ignore

def _project_exists(project_id: int) -> bool:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM projects WHERE id = %s", (project_id,))
            return cur.fetchone() is not None


def _run_workflow_background(project_id: int, task: str) -> None:
    """Runs in a background thread — updates _runs dict as it progresses."""

    try:
        from graphs.graph import get_app
        graph_app = get_app()
        
        upsert_run(project_id, "running", task=task)
        thread_id = f"project-{project_id}-{uuid.uuid4().hex[:8]}"

        initial_state = {
            "project_id": project_id,
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
            initial_state, #type: ignore
            config={"recursion_limit": 20, "configurable": {"thread_id": thread_id}},
        )

        chat_id = create_chat(project_id)
        save_messages(chat_id, result["messages"])
        save_project_memory(project_id, result) #type: ignore

        final_output = result.get("final_output")
        upsert_run(project_id, "complete", task=task, final_output=final_output)

    except Exception as e:
        upsert_run(project_id, "error", task=task, error=str(e))
        
@app.get("/")
async def root():
    return {"name": "ResearchPlatform API", "version": "0.1.0", "status": "running"}

@app.post("/projects", status_code=201)
async def create_project(body: ProjectCreate):
    """Create a new research project."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO projects (name) VALUES (%s) RETURNING id, name, created_at",
                (body.name,),
            )
            row = cur.fetchone()
        conn.commit()
    return {"id": row[0], "name": row[1], "created_at": str(row[2])} #type: ignore

@app.get("/projects")
async def list_projects():
    """List all projects."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, created_at FROM projects ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [{"id": r[0], "name": r[1], "created_at": str(r[2])} for r in rows]

@app.post("/projects/{project_id}/run")
async def run_workflow(project_id: int, body: RunRequest):
    if not _project_exists(project_id):
        raise HTTPException(404, "Project not found")
    
    thread = threading.Thread(
        target=_run_workflow_background,
        args=(project_id, body.task),
        daemon=True
    )
    thread.start()
    
    return {
        "project_id": project_id,
        "status": "started"
    }

@app.get("/projects/{project_id}/status")
async def get_status(project_id: int):
    run = get_run(project_id)
    if not run:
        raise HTTPException(404, "No run found for this project")
    return run

@app.get("/projects/{project_id}/report")
async def get_report(project_id: int):
    """
    Get the final approved report for a project.
    Returns the markdown report file if it exists.
    """
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    report_path = Path(f"data/projects/{project_id}/outputs/final_report.md")

    if not report_path.exists():
        fs = FileSystem(project_id=project_id)
        files = fs.list_files("outputs")
        drafts = [f for f in files if f.startswith("outputs/draft")]

        if drafts:
            return {
                "project_id": project_id,
                "status": "in_progress",
                "message": "No final report yet — workflow still in progress",
                "drafts_available": drafts,
            }

        raise HTTPException(status_code=404, detail="No report found for this project")

    return FileResponse(
        path=str(report_path),
        media_type="text/markdown",
        filename=f"report_project_{project_id}.md",
    )


@app.post("/projects/{project_id}/ingest")
async def ingest_source(project_id: int, url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    """
    Ingest a URL or PDF file into the project's knowledge base.
    Pass either `url` (form field) or `file` (multipart upload), not both.
    """
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    if url and file:
        raise HTTPException(status_code=400, detail="Pass either url or file, not both")

    if not url and not file:
        raise HTTPException(status_code=400, detail="Pass either url or file")

    from rag.ingest import ingest_url, ingest_pdf

    if url:
        try:
            chunks = ingest_url(url=url, project_id=project_id)
            return {"project_id": project_id, "source": url, "chunks_stored": chunks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    if file:
        fs = FileSystem(project_id=project_id)
        filename = f"sources/pdfs/{file.filename}"
        content = await file.read()
        fs.write_bytes(filename, content)
        full_path = str(Path(f"data/projects/{project_id}") / filename)
        try:
            chunks = ingest_pdf(path=full_path, project_id=project_id)
            return {"project_id": project_id, "source": filename, "chunks_stored": chunks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {e}")

@app.get("/projects/{project_id}/memory")
async def get_memory(project_id: int):
    """Get accumulated project memory — prior research and analysis summaries."""
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    from memory.project import load_project_memory

    prior = load_project_memory(project_id)

    return {
        "project_id": project_id,
        "research_summaries": len(prior["research_summaries"]),
        "analysis_summaries": len(prior["analysis_summaries"]),
        "prior_reports": len(prior["prior_reports"]),
        "most_recent_research": prior["research_summaries"][-1][:300] if prior["research_summaries"] else None,
        "most_recent_analysis": prior["analysis_summaries"][-1][:300] if prior["analysis_summaries"] else None,
    }