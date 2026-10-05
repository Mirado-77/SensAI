CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    profile_id TEXT,
    head_message_id TEXT,
    title TEXT
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    parent_id TEXT REFERENCES messages(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'complete',
    tool_calls_json TEXT,
    tool_name TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_messages_parent ON messages(parent_id);

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    model TEXT,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE tool_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    result_json TEXT,
    risk TEXT,
    approved INTEGER,
    status TEXT NOT NULL,
    duration_ms REAL
);
CREATE INDEX idx_tool_calls_run ON tool_calls(run_id);

CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX idx_events_run ON events(run_id);

CREATE TABLE profiles (
    id TEXT PRIMARY KEY,
    data_json TEXT NOT NULL
);
