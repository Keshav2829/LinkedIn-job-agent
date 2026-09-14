"""SQLite state store for the LinkedIn job agent.

Everything the agent learns lives here: your profile, the answer bank used to
fill application forms, preferences, every company/job seen, every application
made, every contact touched, every message queued or sent, and every reply.

Design goals
------------
1. Idempotence. Re-running a search must never re-apply to a job or re-invite a
   person. Natural keys (`jobs.job_id`, `contacts.profile_url`) are UNIQUE.
2. Token thrift. The agent asks this store narrow questions ("what should I do
   next?") instead of re-reading conversation history or re-scraping LinkedIn.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 4


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_time() -> str:
    """Local time-of-day, paired with today() for folder+file names that
    need finer resolution than the day — one file per session instead of
    one ever-growing (or silently overwritten) per-day file."""
    return datetime.now().strftime("%H-%M-%S")


def default_db_path() -> Path:
    env = os.environ.get("LINKEDIN_AGENT_DB")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / "state.db"


def traces_dir(db_path: Path | str | None = None) -> Path:
    """Where step-by-step run traces (for future local-model training) live.

    Sits next to state.db so it inherits the same gitignore and per-machine
    scoping. Plain JSONL + PNG files, not SQL — that's the format training
    tooling actually wants, and it keeps the schema above untouched.
    """
    p = Path(db_path).expanduser() if db_path else default_db_path()
    return p.parent / "traces"


MIGRATIONS: list[str] = [
    # --- 1: core -----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS kv (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- The answer bank. Every question a LinkedIn/Workday form has ever asked,
    -- plus the canonical answer, so the agent never asks you twice.
    CREATE TABLE IF NOT EXISTS answers (
        key        TEXT PRIMARY KEY,        -- snake_case canonical key
        question   TEXT NOT NULL,           -- human phrasing
        value      TEXT NOT NULL,
        kind       TEXT NOT NULL DEFAULT 'text',  -- text|number|bool|choice|date
        category   TEXT NOT NULL DEFAULT 'misc',
        aliases    TEXT NOT NULL DEFAULT '[]',    -- JSON list of alt phrasings
        confidence REAL NOT NULL DEFAULT 1.0,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS companies (
        id           INTEGER PRIMARY KEY,
        name         TEXT NOT NULL,
        name_key     TEXT NOT NULL UNIQUE,   -- lowercased, punctuation-stripped
        linkedin_url TEXT,
        linkedin_id  TEXT,
        industry     TEXT,
        size         TEXT,                   -- e.g. '1001-5000'
        headcount    INTEGER,
        kind         TEXT,                   -- product|service|both|unknown
        is_mnc       INTEGER,
        hq           TEXT,
        status       TEXT NOT NULL DEFAULT 'seen',  -- seen|targeted|applied|blacklisted
        notes        TEXT,
        first_seen   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS jobs (
        id            INTEGER PRIMARY KEY,
        job_id        TEXT NOT NULL UNIQUE,  -- LinkedIn currentJobId
        title         TEXT NOT NULL,
        company_id    INTEGER REFERENCES companies(id),
        company_name  TEXT,
        location      TEXT,
        workplace     TEXT,                  -- onsite|hybrid|remote
        url           TEXT,
        posted        TEXT,
        easy_apply    INTEGER NOT NULL DEFAULT 0,
        seniority     TEXT,
        description   TEXT,
        match_score   INTEGER,
        match_reasons TEXT NOT NULL DEFAULT '[]',
        status        TEXT NOT NULL DEFAULT 'new',
        -- new|scored|queued|applied|skipped|failed|external|closed
        skip_reason   TEXT,
        discovered_at TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_id);

    CREATE TABLE IF NOT EXISTS applications (
        id           INTEGER PRIMARY KEY,
        job_id       INTEGER NOT NULL REFERENCES jobs(id),
        applied_at   TEXT NOT NULL,
        method       TEXT NOT NULL DEFAULT 'easy_apply',
        resume_used  TEXT,
        answers_used TEXT NOT NULL DEFAULT '{}',
        screening    TEXT NOT NULL DEFAULT '[]',  -- questions we could not answer
        outcome      TEXT NOT NULL DEFAULT 'submitted',
        outcome_at   TEXT,
        notes        TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_app_job ON applications(job_id);

    CREATE TABLE IF NOT EXISTS contacts (
        id            INTEGER PRIMARY KEY,
        profile_url   TEXT NOT NULL UNIQUE,
        name          TEXT NOT NULL,
        headline      TEXT,
        company_id    INTEGER REFERENCES companies(id),
        company_name  TEXT,
        title         TEXT,
        location      TEXT,
        role_type     TEXT NOT NULL DEFAULT 'employee',  -- hr|hiring_manager|employee|alumni|peer
        degree        TEXT,                  -- 1st|2nd|3rd
        mutual_count  INTEGER NOT NULL DEFAULT 0,
        is_alumni     INTEGER NOT NULL DEFAULT 0,
        alumni_school TEXT,
        relation      TEXT NOT NULL DEFAULT 'none',
        -- none|invited|connected|declined|withdrawn
        invited_at    TEXT,
        connected_at  TEXT,
        last_touch_at TEXT,
        touch_count   INTEGER NOT NULL DEFAULT 0,
        do_not_contact INTEGER NOT NULL DEFAULT 0,
        notes         TEXT,
        first_seen    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_contacts_relation ON contacts(relation);
    CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company_id);

    -- Every outbound touch. Nothing is sent that does not first exist here.
    CREATE TABLE IF NOT EXISTS outreach (
        id          INTEGER PRIMARY KEY,
        contact_id  INTEGER NOT NULL REFERENCES contacts(id),
        job_id      INTEGER REFERENCES jobs(id),
        kind        TEXT NOT NULL,
        -- invite|hr_pitch|referral_ask|intro|followup|thanks
        body        TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'queued',
        -- queued|approved|sent|skipped|failed|rejected
        queued_at   TEXT NOT NULL,
        decided_at  TEXT,
        sent_at     TEXT,
        error       TEXT,
        attachment  TEXT
        -- body_original added in migration 4
    );
    CREATE INDEX IF NOT EXISTS idx_outreach_status ON outreach(status);

    CREATE TABLE IF NOT EXISTS replies (
        id          INTEGER PRIMARY KEY,
        contact_id  INTEGER REFERENCES contacts(id),
        outreach_id INTEGER REFERENCES outreach(id),
        thread_url  TEXT,
        snippet     TEXT NOT NULL,
        received_at TEXT NOT NULL,
        sentiment   TEXT,          -- positive|neutral|negative|auto
        notified    INTEGER NOT NULL DEFAULT 0,
        handled     INTEGER NOT NULL DEFAULT 0,
        UNIQUE(contact_id, snippet)
    );

    CREATE TABLE IF NOT EXISTS runs (
        id          INTEGER PRIMARY KEY,
        day         TEXT NOT NULL,
        started_at  TEXT NOT NULL,
        ended_at    TEXT,
        applied     INTEGER NOT NULL DEFAULT 0,
        invites     INTEGER NOT NULL DEFAULT 0,
        messages    INTEGER NOT NULL DEFAULT 0,
        searches    INTEGER NOT NULL DEFAULT 0,
        notes       TEXT
    );

    CREATE TABLE IF NOT EXISTS searches (
        id           INTEGER PRIMARY KEY,
        query        TEXT NOT NULL,
        filters      TEXT NOT NULL DEFAULT '{}',
        url          TEXT,
        results      INTEGER NOT NULL DEFAULT 0,
        new_jobs     INTEGER NOT NULL DEFAULT 0,
        ran_at       TEXT NOT NULL
    );
    """,
    # --- 2: event log ------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS events (
        id      INTEGER PRIMARY KEY,
        at      TEXT NOT NULL,
        kind    TEXT NOT NULL,
        ref     TEXT,
        detail  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
    """,
    # --- 3: follow-up scheduling ------------------------------------------
    """
    ALTER TABLE outreach ADD COLUMN followup_due TEXT;
    ALTER TABLE contacts ADD COLUMN followup_due TEXT;
    """,
    # --- 4: preserve the drafted message separately from an edited send ---
    # `outreach approve --body "<edited>"` used to overwrite `body` in place,
    # which meant the model's original draft was gone the moment a human
    # edited it — losing exactly the (draft, edit) pairs a preference-tuned
    # writer model would need. `body` stays "what is actually sent";
    # `body_original` is set once, at draft time, and never touched again.
    """
    ALTER TABLE outreach ADD COLUMN body_original TEXT;
    """,
]


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path) if path else default_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    migrate(con)
    return con


def migrate(con: sqlite3.Connection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS _schema (version INTEGER NOT NULL)"
    )
    row = con.execute("SELECT version FROM _schema").fetchone()
    current = row["version"] if row else 0
    if not row:
        con.execute("INSERT INTO _schema (version) VALUES (0)")
    for i, sql in enumerate(MIGRATIONS, start=1):
        if i <= current:
            continue
        for stmt in sql.split(";"):
            if stmt.strip():
                try:
                    con.execute(stmt)
                except sqlite3.OperationalError as e:
                    # ALTER TABLE ... ADD COLUMN on an existing column
                    if "duplicate column" not in str(e).lower():
                        raise
        con.execute("UPDATE _schema SET version = ?", (i,))
    con.commit()


# --- kv helpers -----------------------------------------------------------

def kv_get(con: sqlite3.Connection, key: str, default=None):
    row = con.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return row["value"]


def kv_set(con: sqlite3.Connection, key: str, value) -> None:
    con.execute(
        "INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, json.dumps(value), utcnow()),
    )
    con.commit()


def log_event(con: sqlite3.Connection, kind: str, ref: str = "", detail=None) -> None:
    con.execute(
        "INSERT INTO events (at, kind, ref, detail) VALUES (?,?,?,?)",
        (utcnow(), kind, ref, json.dumps(detail) if detail is not None else None),
    )
    con.commit()
