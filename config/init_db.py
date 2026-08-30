import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg
from config.env import db_url

schema_path = os.path.join(os.path.dirname(__file__), "..", "config", "schema.sql")
with open(schema_path) as f:
    schema_sql = f.read()

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()

print("Schema applied successfully.")