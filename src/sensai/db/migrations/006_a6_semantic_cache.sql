-- A6: answers served again for a close enough question, keyed by the embedding of the question.
CREATE TABLE semantic_cache (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX semantic_cache_scope ON semantic_cache (scope, created_at);
