from typing import Any
from psycopg import sql
import psycopg
from psycopg.rows import dict_row
from config.env import db_url
from psycopg.types import json
from app.dependencies import get_pool

VALID_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "needs_review",
}

def _validate_status(status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid run status: {status!r}. "
            f"Expected one of: {sorted(VALID_STATUSES)}"
        )

def create_run(run_id: str, project_id: int, thread_id: str, task: str, *, config: dict[str, Any] | None = None) -> None:
    """Create a new durable run record."""
    with get_pool().connection() as conn:  # type: ignore[arg-type]
        with conn.cursor() as cur:
            cur.execute(    
                """
                INSERT INTO runs (id, project_id, thread_id, task, current_agent, config, status) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    run_id,
                    project_id,
                    thread_id,
                    task,
                    "initializing",
                    json.Jsonb(config or {}),
                    "pending",
                ),
            )

        conn.commit()

def update_run(run_id: str, *, status: str | None = None, current_agent: str | None = None, final_output: str | None = None, error: str | None = None) -> None:
    """Update fields on an existing run."""
    if status is not None:
        _validate_status(status)

    updates: list[str] = []
    params: list[Any] = []

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

    query = sql.SQL("UPDATE runs SET {fields} WHERE id = %s").format(
        fields=sql.SQL(", ").join(updates)
    )
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
        conn.commit()

def get_run(run_id: str) -> dict[str, Any] | None:
    """Fetch a run by its unique run ID."""

    with get_pool().connection() as conn:  # type: ignore[arg-type]
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, project_id, thread_id, task, current_agent, config, status, final_output, error, created_at, started_at, completed_at, updated_at
                FROM runs
                WHERE id = %s
                """,
                (run_id,),
            )

            row = cur.fetchone()

            return dict(row) if row else None


def get_latest_run(project_id: int) -> dict[str, Any] | None:
    """Fetch the most recently created run for a project."""

    with get_pool().connection() as conn:  # type: ignore[arg-type]
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, project_id, thread_id, task, current_agent, config, status, final_output, error, created_at, started_at, completed_at, updated_at
                FROM runs
                WHERE project_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (project_id,),
            )

            row = cur.fetchone()
            return dict(row) if row else None

def delete_run(run_id: str) -> bool:
    """Delete a specific run by ID."""
    with psycopg.connection() as conn:  # type: ignore[arg-type]
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM runs
                WHERE id = %s
                """,
                (run_id,),
            )

            deleted = cur.rowcount > 0
        conn.commit()
    return deleted