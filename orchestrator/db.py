"""SQLite state store for the orchestrator agent.

This is deliberately a *second*, independent database from
`jobagent/data/state.db` — the orchestrator's job is to drive that other
project from the outside (reading its skills, shelling out to its CLI), not
to share its memory. Mixing the two would mean a bad orchestrator run could
corrupt the job agent's own state; keeping them apart means the worst a bad
run can do is write nonsense into its own tables.

Three tables:

- `sessions`   — one row per `orchestrator run`, holding the task, the model
                 and endpoint used, and how it ended.
- `messages`   — the full transcript of a session (system/user/assistant/tool),
                 in call order. This is both what the UI renders and the raw
                 material `finetune.py` turns into training examples.
- `memory`     — small persistent key/value facts the agent itself chooses to
                 remember across sessions (via the `memory` tool), e.g. a
                 standing instruction or a fact learned last run. Separate
                 from `jobagent`'s answer bank on purpose — this is the
                 orchestrator's own memory about *how to operate*, not the
                 user's job-search profile.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    return uuid.uuid4().hex


def default_db_path() -> Path:
    env = os.environ.get("ORCHESTRATOR_DB")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent / "data" / "orchestrator.db"


MIGRATIONS: list[str] = [
    # --- 1: core -------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id             TEXT PRIMARY KEY,
        task           TEXT NOT NULL,
        status         TEXT NOT NULL DEFAULT 'running',
        -- running|waiting_for_user|completed|failed|aborted|max_steps_reached|stopped
        model          TEXT NOT NULL,
        base_url       TEXT NOT NULL,
        started_at     TEXT NOT NULL,
        ended_at       TEXT,
        step_count     INTEGER NOT NULL DEFAULT 0,
        summary        TEXT,
        error          TEXT,
        finetune       INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY,
        session_id  TEXT NOT NULL REFERENCES sessions(id),
        seq         INTEGER NOT NULL,
        role        TEXT NOT NULL,   -- system|user|assistant|tool
        content     TEXT NOT NULL,
        tool_name   TEXT,
        tool_input  TEXT,
        created_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);

    CREATE TABLE IF NOT EXISTS memory (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );
    """,
    # --- 2: screenshots ------------------------------------------------
    # An absolute path to a PNG already saved on disk by `jobagent browse
    # shot` — not the image bytes themselves. Keeps the sqlite file small
    # and lets a screenshot be inspected with any image viewer, matching
    # how `jobagent` already stores its own trace screenshots as files.
    # Only ever set on a `tool` row for the `screenshot` tool.
    """
    ALTER TABLE messages ADD COLUMN image_path TEXT;
    """,
    # --- 3: interactive Q&A ---------------------------------------------
    # `vision` and `interactive` are pinned to the session at creation time
    # (from the config in effect when it was started) rather than re-read
    # from whatever config a later `drive_session` call happens to be
    # passed — see agent.drive_session. Without that, resuming a paused
    # session from a different call site (the UI's answer endpoint vs. the
    # CLI vs. a cron-started run) could silently change what the model is
    # and isn't allowed to do mid-conversation.
    """
    ALTER TABLE sessions ADD COLUMN vision INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE sessions ADD COLUMN interactive INTEGER NOT NULL DEFAULT 0;
    """,
    # --- 4: browser-driving permission -----------------------------------
    # Same pinning rationale as vision/interactive above. `run_shell` has
    # always been technically able to reach `jobagent browse ...` — its
    # allowlist is a prefix match on "python -m jobagent", not a per-
    # subcommand one — so this flag doesn't grant a new capability, it
    # only changes what the system prompt's ground rules tell the model
    # it's allowed to do (see agent.build_system_prompt). The real safety
    # boundary is jobagent's own job/outreach approval gate, not this.
    """
    ALTER TABLE sessions ADD COLUMN browse INTEGER NOT NULL DEFAULT 0;
    """,
    # --- 5: human review (quality gate for fine-tuning) -------------------
    # A completed session passing `validate_trajectory` only proves it's
    # *structurally* sound (ends on a real final answer, no unknown tools,
    # ...) -- it says nothing about whether it was actually a *good*
    # demonstration. This is that judgment call, made by a person, at the
    # same granularity *training* happens at -- the whole session, since
    # that's finetune.py's actual export unit (see its module docstring).
    # `reward` mirrors annotation_suite's own -2..2 scale (unusable..
    # exemplary) rather than inventing a second convention. `reviewed` is
    # tracked separately from `reward` being non-NULL for the same reason
    # finetune/prepare_data.py treats them separately: a reviewed-and-
    # rejected session has a `reward` too (e.g. -1), and must not be
    # mistaken for "never looked at" by a naive `reward >= min_reward`
    # filter. See migration 6 for *per-step* review, on individual
    # messages -- a finer-grained, purely diagnostic layer on top of this;
    # export still gates on the session-level verdict here, not on it.
    """
    ALTER TABLE sessions ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE sessions ADD COLUMN reward INTEGER;
    ALTER TABLE sessions ADD COLUMN review_note TEXT;
    ALTER TABLE sessions ADD COLUMN reviewed_at TEXT;
    """,
    # --- 6: per-step review -----------------------------------------------
    # A "step" here is one assistant turn -- the actual decision point
    # (what to call, what to answer) -- so review lives on `messages`
    # rows, keyed by (session_id, seq) which is already unique. Only an
    # `assistant` row can be scored (see set_message_review) -- a
    # system/user/tool row is context the model was given, not something
    # it decided, so there's nothing to grade. This is deliberately *not*
    # what export filters on: training happens on
    # whole trajectories (see migration 5's comment and finetune.py), so
    # a per-step reward doesn't include/exclude anything on its own --
    # it's for catching a bad turn inside an otherwise-fine session before
    # you set that session's own overall reward, the same way you'd read
    # line-by-line before signing off on the whole page. Exported
    # alongside the session as a separate `step_reviews` list (see
    # finetune._session_record), never merged into the `messages` array
    # itself -- that stays exactly the OpenAI chat shape a trainer expects.
    """
    ALTER TABLE messages ADD COLUMN reward INTEGER;
    ALTER TABLE messages ADD COLUMN review_note TEXT;
    """,
    # --- 7: explicit reasoning effort (claude-cli provider) --------------
    # `ClaudeCLIClient` didn't pass `--effort` at all until now, so every
    # session silently inherited whatever `claude`'s own ambient config
    # said (~/.claude/settings.json's modelSettings.<model>.effortLevel,
    # or its own built-in default) -- fine for interactive use, a real gap
    # for a data-generation pipeline: the same "session" could have been
    # generated at a different effort level on a different machine, or
    # after a config change, with no record of which. Pinned like
    # vision/interactive/browse (same rationale: a resumed session must
    # not silently change behavior because a different call site's cfg
    # differs) -- unlike send_images, this changes what the model actually
    # does, not just what's sent to it, so it's not safe to vary per call.
    # NULL means "not set" -- ClaudeCLIClient then omits --effort entirely,
    # identical to pre-migration behavior (inherit ambient config), rather
    # than silently defaulting to some specific level on this table's behalf.
    """
    ALTER TABLE sessions ADD COLUMN effort TEXT;
    """,
    # --- 8: pin provider too (model/base_url already existed, but were
    # never actually read back on resume either) --------------------------
    # The real bug this fixes: `model`/`base_url` have been columns on
    # `sessions` since migration 1, but purely as a historical record --
    # `agent.drive_session` never read them back into the resuming `cfg`,
    # unlike vision/interactive/browse/effort. So a session started with
    # e.g. --provider claude-cli, paused on `ask`, and resumed via
    # `sessions answer <id> "..."` (or the UI's answer box) with no flags
    # re-specified would silently fall back to whatever `orchestrator run`
    # 's *default* config says -- a different provider, a different model,
    # potentially a server that doesn't even support what's already in the
    # session's history (the concrete symptom: an HTTP 400 "messages
    # contain images, but <default model> does not support image inputs"
    # from a completely different endpoint than the one that started the
    # session). `provider` didn't even have a column before this. `api_key`
    # is deliberately *not* added here or pinned -- it's a secret, and this
    # sqlite file is exactly the kind of thing that ends up copied around
    # (exported, imported on another machine); resuming still falls back to
    # $ANTHROPIC_API_KEY / --api-key on whatever cfg drives the resume call,
    # same as it always has.
    """
    ALTER TABLE sessions ADD COLUMN provider TEXT;
    """,
    # --- 9: cooperative stop ----------------------------------------------
    # A flag a *different* connection/thread/process sets to ask a running
    # session's own `drive_session` loop to stop itself -- see agent.py's
    # `request_stop`/`drive_session` and ui.py's `/api/sessions/<id>/stop`.
    # Deliberately cooperative, not a forced kill: there's no clean way to
    # abort a blocking `urllib.request.urlopen` call or a tool already
    # mid-flight without a much bigger rewrite, so the loop only checks this
    # once per step, the same boundary `max_steps`/consecutive-error limits
    # already treat as safe to stop and resume at.
    """
    ALTER TABLE sessions ADD COLUMN stop_requested INTEGER NOT NULL DEFAULT 0;
    """,
    # --- 10: liveness heartbeat --------------------------------------------
    # Cooperative stop (migration 9) only works if something is actually
    # still looping and checking the flag. A session's `drive_session` can
    # be driven by the UI's background thread *or* a separate `orchestrator
    # run`/`sessions answer` CLI process -- and either one can simply stop
    # existing (the UI server gets restarted to pick up new code, a
    # terminal gets closed, a process gets killed) without ever getting a
    # chance to write a terminal status. The row is then stuck showing
    # `running` forever, and a stop request against it sets a flag nobody
    # will ever read. `heartbeat_at` is touched once per step (see
    # `touch_heartbeat`/`heartbeat_is_stale`) so `/api/sessions/<id>/stop`
    # can tell "a live loop just hasn't gotten to the flag yet" apart from
    # "nothing is driving this any more" and force-end the latter itself.
    """
    ALTER TABLE sessions ADD COLUMN heartbeat_at TEXT;
    """,
]

REWARD_LABELS = {-2: "unusable", -1: "bad", 0: "neutral", 1: "good", 2: "exemplary"}

# The claude CLI's own --effort choices (see `claude --help`). Shared here
# rather than duplicated in llm.py/cli.py so there's one source of truth.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path) if path else default_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    migrate(con)
    return con


def migrate(con: sqlite3.Connection) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS _schema (version INTEGER NOT NULL)")
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
                    if "duplicate column" not in str(e).lower():
                        raise
        con.execute("UPDATE _schema SET version = ?", (i,))
    con.commit()


# --- sessions ---------------------------------------------------------

def create_session(
    con: sqlite3.Connection,
    task: str,
    model: str,
    base_url: str,
    finetune: bool = False,
    vision: bool = False,
    interactive: bool = False,
    browse: bool = False,
    effort: str | None = None,
    provider: str = "openai",
    session_id: str | None = None,
) -> str:
    if effort is not None and effort not in EFFORT_LEVELS:
        raise ValueError(f"effort must be one of {EFFORT_LEVELS} or None, got {effort!r}")
    sid = session_id or new_session_id()
    con.execute(
        "INSERT INTO sessions (id, task, status, model, base_url, started_at, finetune, vision, "
        "interactive, browse, effort, provider) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, task, "running", model, base_url, utcnow(), 1 if finetune else 0,
         1 if vision else 0, 1 if interactive else 0, 1 if browse else 0, effort, provider),
    )
    con.commit()
    return sid


def restore_session(
    con: sqlite3.Connection,
    session_id: str,
    task: str,
    model: str,
    status: str,
    started_at: str,
    messages: list[dict[str, Any]],
    base_url: str = "imported",
    ended_at: str | None = None,
    step_count: int = 0,
    summary: str | None = None,
    finetune: bool = True,
    effort: str | None = None,
    provider: str | None = None,
) -> None:
    """Write a full session (row + messages) in one go, for importing an
    externally-produced or previously-exported record — the counterpart to
    reading one out with `get_session`/`get_messages`. Unlike
    `create_session` (which starts a *live* session at step 0, status
    'running', to be driven forward by agent.py), this writes a session
    already in its final state; there's no live loop involved.

    Caller must ensure `session_id` doesn't already exist (or has been
    removed via `delete_session` first) — this does a plain INSERT, not
    an upsert, so a collision raises sqlite3.IntegrityError rather than
    silently overwriting something.
    """
    con.execute(
        "INSERT INTO sessions (id, task, status, model, base_url, started_at, ended_at, "
        "step_count, summary, finetune, effort, provider) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (session_id, task, status, model, base_url, started_at, ended_at,
         step_count, summary, 1 if finetune else 0, effort, provider),
    )
    for seq, m in enumerate(messages, start=1):
        con.execute(
            "INSERT INTO messages (session_id, seq, role, content, tool_name, tool_input, "
            "image_path, created_at, reward, review_note) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (session_id, seq, m["role"], m["content"], m.get("tool_name"), m.get("tool_input"),
             m.get("image_path"), utcnow(), m.get("reward"), m.get("review_note")),
        )
    con.commit()


def end_session(
    con: sqlite3.Connection,
    session_id: str,
    status: str,
    summary: str | None = None,
    error: str | None = None,
) -> None:
    # stop_requested=0 -- a session ending for any reason (including
    # `stopped` itself) should never carry a stale flag into whatever
    # happens to it next (e.g. a retry).
    con.execute(
        "UPDATE sessions SET status=?, ended_at=?, summary=?, error=?, stop_requested=0 WHERE id=?",
        (status, utcnow(), summary, error, session_id),
    )
    con.commit()


def set_status(con: sqlite3.Connection, session_id: str, status: str, clear_error: bool = False) -> None:
    """Change status without touching `ended_at`/`summary` — for a session
    that's merely pausing (`waiting_for_user`) or resuming (`running`), not
    actually finishing. Use `end_session` for a terminal state.

    `clear_error` defaults off (the `waiting_for_user` <-> `running` pause/
    resume never has one to clear) but `retry_session` passes it explicitly:
    without it, a `failed`/`aborted`/`stopped` session's `error` column --
    both display surfaces (the UI footer, `sessions show`) show it
    unconditionally whenever it's set, regardless of current status -- would
    keep showing the *previous* failure's message for the whole duration of
    the retry attempt, making a session that's genuinely running again look
    like it's stuck erroring with no retry button (the button correctly
    disappears once status flips to `running`, but the stale text stays).
    Also clears `stop_requested`, for the same reason `end_session` always
    does: a fresh `running` attempt must not inherit a stop flag left over
    from whatever the session was doing before."""
    if clear_error:
        con.execute(
            "UPDATE sessions SET status=?, error=NULL, stop_requested=0 WHERE id=?", (status, session_id)
        )
    else:
        con.execute("UPDATE sessions SET status=? WHERE id=?", (status, session_id))
    con.commit()


def request_stop(con: sqlite3.Connection, session_id: str) -> None:
    """Ask a running session's `drive_session` loop (running in another
    thread, or another process entirely -- this only touches the shared
    sqlite row, so it works across both) to stop at its next step boundary.
    See migration 9's comment for why this is cooperative, not a hard kill."""
    con.execute("UPDATE sessions SET stop_requested=1 WHERE id=?", (session_id,))
    con.commit()


def is_stop_requested(con: sqlite3.Connection, session_id: str) -> bool:
    row = con.execute("SELECT stop_requested FROM sessions WHERE id=?", (session_id,)).fetchone()
    return bool(row and row["stop_requested"])


def touch_heartbeat(con: sqlite3.Connection, session_id: str) -> None:
    """Called once per step, right before the (possibly slow) model call --
    see migration 10's comment. Records "a loop was alive and about to work
    on this session as of this timestamp", not "this step finished"."""
    con.execute("UPDATE sessions SET heartbeat_at=? WHERE id=?", (utcnow(), session_id))
    con.commit()


def heartbeat_is_stale(con: sqlite3.Connection, session_id: str, max_age_seconds: float) -> bool:
    """True if no loop has touched this session's heartbeat recently enough
    to still plausibly be alive -- falls back to `started_at` for a session
    whose driving loop died before ever reaching its first step (or a
    pre-migration-10 row that's never had a heartbeat at all)."""
    row = con.execute("SELECT heartbeat_at, started_at FROM sessions WHERE id=?", (session_id,)).fetchone()
    if row is None:
        return True
    ts = row["heartbeat_at"] or row["started_at"]
    if not ts:
        return True
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
    return age > max_age_seconds


def bump_step(con: sqlite3.Connection, session_id: str) -> int:
    con.execute("UPDATE sessions SET step_count = step_count + 1 WHERE id=?", (session_id,))
    con.commit()
    row = con.execute("SELECT step_count FROM sessions WHERE id=?", (session_id,)).fetchone()
    return row["step_count"] if row else 0


def get_session(con: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = con.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(con: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = con.execute(
        "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# --- messages -----------------------------------------------------------

def add_message(
    con: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    tool_name: str | None = None,
    tool_input: str | None = None,
    image_path: str | None = None,
) -> int:
    row = con.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM messages WHERE session_id=?",
        (session_id,),
    ).fetchone()
    seq = row["n"]
    con.execute(
        "INSERT INTO messages (session_id, seq, role, content, tool_name, tool_input, image_path, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (session_id, seq, role, content, tool_name, tool_input, image_path, utcnow()),
    )
    con.commit()
    return seq


def get_messages(con: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    rows = con.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY seq ASC", (session_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_message_image_path(con: sqlite3.Connection, session_id: str, seq: int) -> str | None:
    row = con.execute(
        "SELECT image_path FROM messages WHERE session_id=? AND seq=?", (session_id, seq)
    ).fetchone()
    return row["image_path"] if row else None


def delete_session(con: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    """Permanently delete one session and all its messages. Returns
    `{"session_id", "message_count", "image_paths"}`, or None if the
    session doesn't exist.

    Deliberately does *not* touch any screenshot files on disk, even
    though `image_paths` lists every one referenced by the deleted
    messages: those PNGs live in `jobagent`'s own shared trace store
    (`data/traces/screenshots/`, see jobagent/browser.py's `shot()`), not
    anywhere this module owns, and the exact same file could be cited by
    `jobagent`'s own harvest/training corpus independently of this
    session. Deleting them here would risk silently breaking a different
    pipeline's data for a win that's purely disk space. The caller
    decides, explicitly, whether to also remove them (see cli.py's
    `sessions delete --delete-screenshots` / ui.py's DELETE endpoint).
    """
    if get_session(con, session_id) is None:
        return None
    message_count = con.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE session_id=?", (session_id,)
    ).fetchone()["n"]
    image_paths = [
        r["image_path"] for r in con.execute(
            "SELECT image_path FROM messages WHERE session_id=? AND image_path IS NOT NULL",
            (session_id,),
        ).fetchall()
    ]
    con.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    con.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    con.commit()
    return {"session_id": session_id, "message_count": message_count, "image_paths": image_paths}


def set_review(con: sqlite3.Connection, session_id: str, reward: int | None, note: str | None = None) -> bool:
    """Record a human review verdict for one session. `reward` follows
    `annotation_suite`'s own -2..2 scale (see `REWARD_LABELS`) for
    consistency across the project's two fine-tuning pipelines, though
    this one reviews whole sessions, not individual steps. Returns False
    if the session doesn't exist.
    """
    if reward is not None and reward not in REWARD_LABELS:
        raise ValueError(f"reward must be one of {sorted(REWARD_LABELS)} or None, got {reward!r}")
    if get_session(con, session_id) is None:
        return False
    con.execute(
        "UPDATE sessions SET reviewed=1, reward=?, review_note=?, reviewed_at=? WHERE id=?",
        (reward, note, utcnow(), session_id),
    )
    con.commit()
    return True


def clear_review(con: sqlite3.Connection, session_id: str) -> bool:
    """Reset a session back to unreviewed. Returns False if it doesn't exist."""
    if get_session(con, session_id) is None:
        return False
    con.execute(
        "UPDATE sessions SET reviewed=0, reward=NULL, review_note=NULL, reviewed_at=NULL WHERE id=?",
        (session_id,),
    )
    con.commit()
    return True


def set_message_review(
    con: sqlite3.Connection, session_id: str, seq: int, reward: int | None, note: str | None = None,
) -> bool:
    """Record a reward for one step (one message row, identified by its
    (session_id, seq) -- see migration 6). Only an `assistant` row is a
    real decision point (a system/user/tool row is context the model was
    given, not something it decided); scoring one of those is rejected
    with ValueError, the same way an out-of-range reward is. Returns
    False if no such row exists at all.
    """
    if reward is not None and reward not in REWARD_LABELS:
        raise ValueError(f"reward must be one of {sorted(REWARD_LABELS)} or None, got {reward!r}")
    row = con.execute(
        "SELECT role FROM messages WHERE session_id=? AND seq=?", (session_id, seq)
    ).fetchone()
    if row is None:
        return False
    if row["role"] != "assistant":
        raise ValueError(f"only assistant messages can be scored (row {seq} is {row['role']!r})")
    con.execute(
        "UPDATE messages SET reward=?, review_note=? WHERE session_id=? AND seq=?",
        (reward, note, session_id, seq),
    )
    con.commit()
    return True


def clear_message_review(con: sqlite3.Connection, session_id: str, seq: int) -> bool:
    cur = con.execute(
        "UPDATE messages SET reward=NULL, review_note=NULL WHERE session_id=? AND seq=?",
        (session_id, seq),
    )
    con.commit()
    return cur.rowcount > 0


def _image_data_uri(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def to_api_messages(rows: list[dict[str, Any]], include_images: bool = True) -> list[dict[str, Any]]:
    """Map our stored transcript onto plain OpenAI chat roles.

    `tool` observations are sent back as `user` turns rather than the
    OpenAI `tool` role — that role is only valid when paired with a
    `tool_calls` id from native function-calling, which we deliberately
    don't rely on (see agent.py). A prefixed `user` turn works against any
    OpenAI-compatible endpoint, including a bare-bones vLLM server.

    A `tool` row with an `image_path` (only ever set by the `screenshot`
    tool) becomes a multimodal turn — `content` is a list of an OpenAI-style
    `image_url` block alongside the text — instead of a plain string. Every
    other row is untouched. `include_images=False` degrades those rows back
    to text-only (naming the file instead of attaching it), for a config or
    a replay where the model being called doesn't take image input.
    """
    out = []
    for m in rows:
        if m["role"] == "tool":
            label = m["tool_name"] or "tool"
            text = f"Observation ({label}):\n{m['content']}"
            raw_image_path = m.get("image_path")
            image_path = raw_image_path if include_images else None
            data_uri = _image_data_uri(image_path) if image_path else None
            if data_uri:
                content: Any = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
            else:
                if raw_image_path and not include_images:
                    text += "\n[screenshot omitted: replaying without images this turn]"
                elif image_path and not data_uri:
                    text += "\n[image file is missing on disk; it could not be attached]"
                content = text
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


# --- memory ---------------------------------------------------------------

def memory_set(con: sqlite3.Connection, key: str, value: Any) -> None:
    con.execute(
        "INSERT INTO memory (key, value, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, json.dumps(value), utcnow()),
    )
    con.commit()


def memory_get(con: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = con.execute("SELECT value FROM memory WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return row["value"]


def memory_list(con: sqlite3.Connection) -> dict[str, Any]:
    rows = con.execute("SELECT key, value FROM memory ORDER BY key ASC").fetchall()
    out = {}
    for r in rows:
        try:
            out[r["key"]] = json.loads(r["value"])
        except json.JSONDecodeError:
            out[r["key"]] = r["value"]
    return out
