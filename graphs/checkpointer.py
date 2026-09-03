import psycopg_pool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from config.env import db_url

_pool = None
_checkpointer = None

def get_checkpointer():
    global _pool, _checkpointer
    if _checkpointer is None:
        _pool = psycopg_pool.ConnectionPool(
            conninfo=db_url, #type: ignore
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
        _checkpointer = PostgresSaver(_pool) #type: ignore
        _checkpointer.setup()
    return _checkpointer