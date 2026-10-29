-- M3: durable project memory driven by the model (facts) and the links between them (relations).
CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('convention', 'decision', 'fact', 'entity')),
    subject TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE relations (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (subject, relation, object)
);
