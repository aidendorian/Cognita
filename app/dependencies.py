# app/dependencies.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from config.env import fastapi_api_key
from psycopg_pool import ConnectionPool
from config.env import db_url

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != fastapi_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key"
        )
    return api_key

_pool: ConnectionPool | None = None

def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(db_url, min_size=1, max_size=10, open=True)
    return _pool