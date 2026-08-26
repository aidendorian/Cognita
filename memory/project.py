import psycopg
from config.env import db_url
from graphs.state import State

def save_project_memory(project_id: int, state: State) -> None:

    to_save = [("researcher", state.get("research_summary"), state.get("research_output")),
               ("analyst",    state.get("analysis_summary"),  state.get("analysis"))]

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            for agent, summary, raw in to_save:
                if summary:
                    cur.execute(
                        """
                        INSERT INTO summaries (project_id, agent, content, raw_length)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (project_id, agent, summary, len(raw) if raw else 0),
                    )

            if state.get("final_output"):
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
    }

    for agent, content in rows:
        if agent == "researcher":
            result["research_summaries"].append(content)
        elif agent == "analyst":
            result["analysis_summaries"].append(content)
        elif agent == "writer":
            result["prior_reports"].append(content)

    return result