import psycopg
from config.env import db_url
from graphs.state import State

MAX_SUMMARIES_PER_AGENT = 5

def _prune_oldest(cur, project_id: int, agent: str) -> None:
    """Delete the oldest row for this agent if we're at the cap."""
    cur.execute(
        "SELECT COUNT(*) FROM summaries WHERE project_id = %s AND agent = %s",
        (project_id, agent),
    )
    count = cur.fetchone()[0]
    if count >= MAX_SUMMARIES_PER_AGENT:
        cur.execute(
            """
            DELETE FROM summaries
            WHERE id = (
                SELECT id FROM summaries
                WHERE project_id = %s AND agent = %s
                ORDER BY created_at ASC
                LIMIT 1
            )
            """,
            (project_id, agent),
        )

def save_project_memory(project_id: int, state: State) -> None:

    to_save = [
        ("researcher", state.get("research_summary"), state.get("research_output")),
        ("analyst",    state.get("analysis_summary"), state.get("analysis")),
        ("critic",     state.get("novelty_report"),   state.get("novelty_report")),
    ]

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            for agent, summary, raw in to_save:
                if summary:
                    _prune_oldest(cur, project_id, agent)
                    cur.execute(
                        """
                        INSERT INTO summaries (project_id, agent, content, raw_length)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (project_id, agent, summary, len(raw) if raw else 0),
                    )

            if state.get("final_output"):
                _prune_oldest(cur, project_id, "writer")
                cur.execute(
                    """
                    INSERT INTO summaries (project_id, agent, content, raw_length)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        project_id,
                        "writer",
                        state["final_output"],
                        len(str(state["final_output"])),
                    ),
                )
        conn.commit()

def load_project_memory(project_id: int) -> dict:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent, content FROM summaries
                WHERE project_id = %s
                ORDER BY created_at ASC
                """,
                (project_id,),
            )
            rows = cur.fetchall()

    result: dict[str, list[str]] = {
        "research_summaries": [],
        "analysis_summaries": [],
        "prior_reports": [],
        "critic_reports": [],
    }

    for agent, content in rows:
        if agent == "researcher":
            result["research_summaries"].append(content)
        elif agent == "analyst":
            result["analysis_summaries"].append(content)
        elif agent == "writer":
            result["prior_reports"].append(content)
        elif agent == "critic":
            result["critic_reports"].append(content)

    return result