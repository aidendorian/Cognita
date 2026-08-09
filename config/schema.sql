CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE projects(id SERIAL PRIMARY KEY, name TEXT, created_at TIMESTAMP DEFAULT now());
CREATE TABLE chats(id SERIAL PRIMARY KEY, project_id INT REFERENCES projects(id), created_at TIMESTAMP DEFAULT now());
CREATE TABLE messages(id SERIAL PRIMARY KEY, chat_id INT REFERENCES chats(id), role TEXT, content TEXT, created_at TIMESTAMP DEFAULT now());
CREATE TABLE embeddings(id SERIAL PRIMARY KEY, project_id INT REFERENCES projects(id), chunk_text TEXT, embedding VECTOR(768), source TEXT);
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);