import psycopg
from config.env import db_url

def create_chat(project_id: int) -> int:
    with psycopg.connect(db_url) as conn: #type: ignore
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (project_id) VALUES (%s) RETURNING id",
                (project_id,),
            )
            chat_id = cur.fetchone()[0] #type: ignore
        conn.commit()
    return chat_id

def save_messages(chat_id: int, messages: list) -> None:
    if not messages:
        return
    with psycopg.connect(db_url) as conn: #type: ignore
        with conn.cursor() as cur:
            for msg in messages:
                if hasattr(msg, "type"):
                    role = "user" if msg.type == "human" else "assistant"
                    content = msg.content
                else:
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "")
                if content:
                    cur.execute(
                        "INSERT INTO messages (chat_id, role, content) VALUES (%s, %s, %s)",
                        (chat_id, role, content),
                    )
        conn.commit()

def load_messages(chat_id: int) -> list[dict]:
    with psycopg.connect(db_url) as conn: #type: ignore
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content FROM messages
                WHERE chat_id = %s
                ORDER BY created_at ASC
                """,
                (chat_id,),
            )
            return [{"role": row[0], "content": row[1]} for row in cur.fetchall()]