CREATE EXTENSION IF NOT EXISTS vector;

-- Create a sample vector table (optional)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);

-- Create an index for similarity search (optional)
CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);