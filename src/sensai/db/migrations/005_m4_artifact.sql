-- M4: the state document of a session, one row per section, and the history of its patches.
CREATE TABLE artifact_sections (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    section TEXT NOT NULL,
    content TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (session_id, section)
);

CREATE TABLE artifact_patches (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    run_id TEXT,
    section TEXT NOT NULL,
    op TEXT NOT NULL CHECK (op IN ('replace', 'append')),
    before TEXT NOT NULL,
    after TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX artifact_patches_session ON artifact_patches (session_id, created_at);
