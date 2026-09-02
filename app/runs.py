import psycopg
from psycopg.rows import dict_row
from config.env import db_url

def upsert_run(project_id: int, status: str, task: str | None = None, final_output: str | None= None, error: str | None = None):
    """Insert or update the run record for a project."""
    with psycopg.connect(db_url) as conn: #type: ignore
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO runs (project_id, status, task, final_output, error, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (project_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    task = COALESCE(EXCLUDED.task, runs.task),
                    final_output = COALESCE(EXCLUDED.final_output, runs.final_output),
                    error = COALESCE(EXCLUDED.error, runs.error),
                    updated_at = NOW()
            """, (project_id, status, task, final_output, error))
            conn.commit()

def get_run(project_id: int) -> dict | None:
    """Fetch the run record for a project."""
    with psycopg.connect(db_url) as conn: #type: ignore
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM runs WHERE project_id = %s", (project_id,))
            row = cur.fetchone()
            return dict(row) if row else None

def delete_run(project_id: int):
    """Optional: delete a run record if needed."""
    with psycopg.connect(db_url) as conn: #type: ignore
        with conn.cursor() as cur:
            cur.execute("DELETE FROM runs WHERE project_id = %s", (project_id,))
            conn.commit()