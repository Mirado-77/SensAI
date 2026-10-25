-- X5: commands run at a fixed interval, and the commands the user approved for them.
CREATE TABLE scheduled_tasks (
    id TEXT PRIMARY KEY,
    argv_json TEXT NOT NULL,
    interval_minutes INTEGER NOT NULL CHECK (interval_minutes > 0),
    status TEXT NOT NULL CHECK (status IN ('active', 'cancelled')),
    created_at REAL NOT NULL,
    last_run_at REAL,
    last_status TEXT
);

CREATE TABLE allowed_commands (
    argv_json TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);
