CREATE EXTENSION IF NOT EXISTS vector;

-- Create the embeddings table for PrivateGPT
-- Dimension 768 is for Gemini's text-embedding-004
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(768)
);

-- Create an HNSW index for high-performance similarity search
-- Dist method vector_cosine_ops is recommended for cosine similarity
CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);