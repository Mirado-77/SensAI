-- R1: chunks of the indexed codebase and their embeddings (float32, normalized).
CREATE TABLE code_chunks (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    sha TEXT NOT NULL,
    model TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX idx_code_chunks_path ON code_chunks(path);
