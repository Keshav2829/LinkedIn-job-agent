"""Turn session transcripts into fine-tuning data.

Nothing fancy: a session is already a sequence of
system/user/assistant/tool turns that reproduces exactly the input the model
saw and the ```action block it was supposed to produce. That *is* the
training example for teaching a smaller local model to follow this
project's response protocol reliably. Structural soundness needs no human
adjudication the way `annotation_suite/` needs for the browser-trace corpus
— either the model's action block was valid and got acted on, or it wasn't,
and `validate_trajectory` checks that mechanically. Whether a structurally
sound session was actually a *good* demonstration is still a human call —
see `db.set_review`/`export_all`'s `require_review`/`min_reward`.

Three ways this data gets written:

- Per-session, automatically, right after a session that was started with
  `--finetune` completes (see agent.drive_session).
- On demand, combining many sessions: `python -m orchestrator finetune export`.
- Imported from an external or previously-exported file:
  `python -m orchestrator sessions import` — see `import_file` below.

Format: one JSON object per line, `{"schema": "orchestrator-sft/1", ...}`,
with a `messages` list in plain OpenAI chat format — directly usable by most
SFT trainers (TRL, axolotl, etc.) after stripping the metadata keys.

## Dataset-quality practices this module follows

Adapted from a general agent-fine-tuning-dataset guideline, but kept on this
project's own wire protocol: a text ```action block on the assistant turn,
tool results folded into `role: "user"` (see db.to_api_messages and
agent.py's own docstring for why) — not the native OpenAI `tool_calls` /
`role: "tool"` shape that guideline describes. Changing that would mean
changing what actually gets served, not just how training data is packaged,
so the wire format stays as-is; only how this module curates and reports on
that data changed.

- **Complete trajectories, not steps.** Unchanged from before: a session
  (task -> every tool call -> every observation -> final answer) is one
  training example already.
- **Only verified-successful trajectories, by default.** `export_all` only
  includes `status == "completed"` sessions unless told otherwise via
  `statuses=` / the CLI's `--status` / `--any-status`. A `failed` /
  `aborted` / `max_steps_reached` session never reached a real final
  answer, so training on it as-is would teach the model to stop, not to
  finish — pass explicit statuses in if you deliberately want a
  negative-example slice.
- **Structural validation before export.** `validate_trajectory` checks
  role ordering, that every assistant turn parses as a valid action block
  (or is a recoverable protocol error mid-trajectory — see below), that
  every `tool`-role row names a real tool, and that the trajectory actually
  ends on a `final` action. A session that fails this is skipped and
  counted, never silently included. `validate_export_file` does the
  lighter, file-level version of the same checks for a JSONL that's already
  been exported (e.g. before handing it to a trainer).
- **Trajectory-type coverage, reported not assumed.** `classify_trajectory`
  buckets each kept session — single tool call, multi-step, no tool needed,
  clarification (via `ask`), error recovery, multi-turn — and `export_all`
  reports the mix, in the same spirit as a recommended-coverage table: a
  lopsided export (all single-tool, no error recovery) is visible instead
  of silently shipped. There is deliberately no "parallel calls" category —
  agent.py's protocol allows exactly one action per reply, so that shape
  can't occur here.
- **Split by task, not by session.** `val_frac > 0` groups sessions by
  their literal task string before splitting, so two retries of the same
  task can't land on opposite sides of the split (leakage).
- **Assistant-only loss, and why it's automatic here.** `to_api_messages`
  already turns every tool observation into a `role: "user"` turn — there
  is no `role: "tool"` in this schema at all. So the standard trainer
  setting ("mask everything but assistant turns" — TRL's
  `assistant_only_loss=True`, axolotl's `train_on_inputs: false`) is
  exactly correct out of the box: every `user` row (task, observations,
  protocol errors, resumed answers) gets masked; every `assistant` row
  (reasoning + the action block) is trained on. Skipping that trainer flag
  is the single most common way this kind of data goes wrong — the model
  starts hallucinating tool output instead of waiting for the real thing.
"""

from __future__ import annotations

import base64
import json
import random
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

from . import db, tools

SCHEMA = "orchestrator-sft/1"

# Every tool name that could legitimately appear on a `tool`-role row across
# any config (vision on or off — `screenshot` is only ever registered when
# `cfg.vision` is set, see tools.available_tools), plus the two synthetic
# names agent.py uses for protocol-error and ask-pause observations. Not a
# real "tool" call in either case, so they're allowed without a lookup.
KNOWN_TOOL_NAMES = set(tools.TOOL_REGISTRY) | {"screenshot"}
SYNTHETIC_TOOL_NAMES = {"_protocol", "_ask_user"}

DEFAULT_STATUSES: tuple[str, ...] = ("completed",)


def _session_record(con, session_id: str) -> dict[str, Any]:
    session = db.get_session(con, session_id)
    if session is None:
        raise ValueError(f"no such session: {session_id}")
    rows = db.get_messages(con, session_id)
    messages = db.to_api_messages(rows)
    record = {
        "schema": SCHEMA,
        "session_id": session_id,
        "task": session["task"],
        "provider": session["provider"] or "openai",
        "model": session["model"],
        "status": session["status"],
        "step_count": session["step_count"],
        "started_at": session["started_at"],
        "ended_at": session["ended_at"],
        "messages": messages,
    }
    # Traceability for generation params that affect *quality*, not just
    # routing -- worth knowing which effort level a Claude-driven session
    # was actually generated at (see db.py migration 7). None for anything
    # not driven by --provider claude-cli, or a claude-cli session that
    # didn't pin one -- omitted entirely rather than a misleading key.
    if session.get("effort"):
        record["effort"] = session["effort"]
    # Per-step (per-message) review, kept as a side list keyed by `seq`
    # rather than merged into `messages` -- that array has to stay exactly
    # the OpenAI chat shape a trainer expects (see the assistant-only-loss
    # note in this module's docstring); an extra key on a message dict
    # would break a strict schema validator for no training-time benefit,
    # since export still gates on the session-level reward, not this.
    step_reviews = [
        {"seq": r["seq"], "role": r["role"], "reward": r["reward"], "note": r["review_note"]}
        for r in rows if r["reward"] is not None
    ]
    if step_reviews:
        record["step_reviews"] = step_reviews
    return record


def export_session(con, session_id: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = _session_record(con, session_id)
    out_path = out_dir / f"{session_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


# --- validation -----------------------------------------------------------

def validate_trajectory(rows: list[dict[str, Any]]) -> list[str]:
    """Structural checks for one session's raw message rows (as returned by
    `db.get_messages`), adapted from a general agent-dataset-validation
    checklist to this project's own action-block protocol. Returns a list
    of problems; empty means the trajectory is safe to export.
    """
    errors: list[str] = []
    if not rows:
        return ["no messages"]

    roles = [r["role"] for r in rows]
    unknown_roles = sorted({r for r in roles if r not in ("system", "user", "assistant", "tool")})
    if unknown_roles:
        errors.append(f"unknown role(s): {unknown_roles}")
    if "system" in roles[1:]:
        errors.append("a 'system' row appears after position 0")
    first_non_system = next((r for r in rows if r["role"] != "system"), None)
    if first_non_system is None or first_non_system["role"] != "user":
        errors.append("first non-system row must be 'user'")

    last_index = len(rows) - 1
    for i, row in enumerate(rows):
        if row["role"] != "assistant":
            continue
        action = tools.extract_action(row["content"])
        if action is None:
            # A protocol error is legitimate mid-trajectory — agent.py logs
            # it as a 'tool' row (tool_name='_protocol') and gives the model
            # a chance to recover. Only the *last* assistant row failing to
            # parse is a real problem, since a genuinely completed
            # trajectory must end on a real action.
            if i == last_index:
                errors.append(f"row {i}: last assistant row has no valid action block")
            continue
        if i == last_index and "final" not in action:
            kind = next(iter(action), "?")
            errors.append(f"row {i}: last assistant row's action is {kind!r}, not 'final'")

    for i, row in enumerate(rows):
        if row["role"] != "tool":
            continue
        name = row.get("tool_name")
        if name and name not in KNOWN_TOOL_NAMES and name not in SYNTHETIC_TOOL_NAMES:
            errors.append(f"row {i}: unknown tool_name {name!r}")
        raw_input = row.get("tool_input")
        if raw_input:
            try:
                json.loads(raw_input)
            except json.JSONDecodeError:
                errors.append(f"row {i}: tool_input for {name!r} is not valid JSON")

    return errors


def validate_export_file(path: Path) -> dict[str, Any]:
    """Lighter, file-level validator for an already-exported
    `orchestrator-sft/1` JSONL — run this on a file before handing it to a
    trainer. Works on the exported `messages` shape (tool observations
    already folded into `role: "user"`), so unlike `validate_trajectory` it
    can't re-check tool names/inputs against the live tool registry — that
    check already happened at export time, in `export_all`. This catches
    corruption or hand-editing after the fact instead: bad JSON, bad role
    ordering, a trajectory that doesn't end on a real final answer.
    """
    path = Path(path)
    total = 0
    problems: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append({"line": line_no, "errors": [f"invalid JSON: {e}"]})
            continue
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or not msgs:
            problems.append({"line": line_no, "errors": ["missing or empty 'messages'"]})
            continue

        errs: list[str] = []
        roles = [m.get("role") for m in msgs]
        bad_roles = sorted(set(roles) - {"system", "user", "assistant"})
        if bad_roles:
            errs.append(f"unexpected role(s): {bad_roles}")
        if "system" in roles[1:]:
            errs.append("'system' row not first")

        last = msgs[-1]
        if last.get("role") != "assistant":
            errs.append("last row is not 'assistant'")
        else:
            content = last.get("content")
            if isinstance(content, str):
                text = content
            else:
                text = next(
                    (b.get("text", "") for b in (content or []) if isinstance(b, dict) and b.get("type") == "text"),
                    "",
                )
            action = tools.extract_action(text)
            if action is None or "final" not in action:
                errs.append("last assistant row does not end on a 'final' action")

        if errs:
            problems.append({"line": line_no, "session_id": rec.get("session_id"), "errors": errs})

    return {
        "path": str(path),
        "total": total,
        "valid": total - len(problems),
        "invalid": len(problems),
        "problems": problems,
    }


# --- trajectory-type coverage ----------------------------------------------

def classify_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bucket one session's raw rows into the coverage categories a healthy
    agent fine-tuning set needs a mix of. Multiple flags can be true at
    once — e.g. a session can be both 'multi_step' and 'error_recovery'.
    Assumes `rows` already passed `validate_trajectory`.
    """
    tool_calls = 0
    used_ask = False
    had_error = False
    user_rows = 0

    for row in rows:
        if row["role"] == "user":
            user_rows += 1
        elif row["role"] == "assistant":
            action = tools.extract_action(row["content"])
            if action and "tool" in action:
                tool_calls += 1
        elif row["role"] == "tool":
            if row.get("tool_name") == "_ask_user" and not row["content"].startswith("ERROR"):
                used_ask = True
            elif row["content"].startswith("ERROR"):
                had_error = True

    return {
        "tool_calls": tool_calls,
        "single_tool": tool_calls == 1,
        "multi_step": tool_calls >= 2,
        "no_tool": tool_calls == 0,
        "clarification": used_ask,
        # Exported sessions are `completed` by default (see DEFAULT_STATUSES),
        # so any ERROR observation seen along the way was, by definition,
        # recovered from — the session still reached a real final answer.
        "error_recovery": had_error,
        # A resumed clarification also adds a 'user' row (the answer) — only
        # count this when the extra user turn *isn't* just that, so it
        # tracks genuinely separate follow-up requests in one trajectory.
        "multi_turn": user_rows > 1 and not used_ask,
    }


# --- export -----------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict[str, Any]], strip_metadata: bool) -> Path | None:
    """Write `records` to `path`. With `strip_metadata`, each line is
    reduced to exactly `{"messages": [...]}` — the shape a hosted
    fine-tuning API (which typically rejects unknown top-level keys) or a
    strict trainer expects — and everything else (`session_id`, `task`,
    `model`, `status`, ...) is written instead to a same-order sidecar,
    `<path>.meta.jsonl`, per the guideline this module follows: strip before
    uploading, keep the metadata in a sidecar rather than lose it. Returns
    the sidecar path, or None when `strip_metadata` is False.
    """
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            row = {"messages": record["messages"]} if strip_metadata else record
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not strip_metadata:
        return None
    meta_path = path.with_name(path.name + ".meta.jsonl")
    with meta_path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            meta = {"line": i, **{k: v for k, v in record.items() if k != "messages"}}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return meta_path


def export_all(
    con,
    out_path: Path,
    include_all: bool = False,
    statuses: tuple[str, ...] | None = DEFAULT_STATUSES,
    val_frac: float = 0.0,
    seed: int = 3407,
    strip_metadata: bool = False,
    require_review: bool = False,
    min_reward: int | None = None,
) -> dict[str, Any]:
    """Combine sessions into fine-tuning JSONL file(s).

    - `include_all=False` (default): only sessions started with
      `--finetune` (or flagged in the UI). `include_all=True` ignores that
      flag and considers every session.
    - `statuses`: only sessions in one of these statuses are eligible
      (default: `("completed",)` — see the module docstring). Pass `None`
      to include every status regardless.
    - `require_review`: only include sessions a human has reviewed via
      `db.set_review` (see the UI's review panel / `sessions review`).
      Structural validation (`validate_trajectory`) proves a session is
      well-formed; it says nothing about whether it was actually a *good*
      demonstration, which is what this is for.
    - `min_reward`: only include sessions with `reward >= min_reward`.
      Deliberately a *separate* check from `require_review` rather than
      folded into one filter — a session's reward is `NULL` until
      reviewed, never `0`, so `reward >= min_reward` alone would silently
      let every never-reviewed session through a naive filter (the same
      bug `finetune/prepare_data.py`'s own comment calls out for the
      harvested corpus). Pass both to get "reviewed and good."
    - `val_frac > 0`: also writes a validation split, grouped by the
      session's literal task string so retries of the same task can't
      leak across train/val. Train goes to `out_path`; val goes to
      `out_path` with `.val` inserted before the suffix (e.g.
      `export.jsonl` -> `export.jsonl` + `export.val.jsonl`).
    - `strip_metadata`: write each line as bare `{"messages": [...]}` and
      move `schema`/`session_id`/`task`/`model`/`status`/`step_count`/
      `started_at`/`ended_at` to a `<file>.meta.jsonl` sidecar instead (see
      `_write_jsonl`). Off by default — this project's own trainer (the
      Colab notebook) only ever reads `record["messages"]`, so the extra
      keys are harmless there and useful for debugging; turn this on only
      when exporting for something that validates unknown keys strictly.

    Returns a report: `{"count", "train_count", "val_count", "val_path",
    "train_meta_path", "val_meta_path", "skipped_invalid",
    "skipped_unreviewed", "by_category"}`. A session that fails
    `validate_trajectory` is skipped — never written, never counted in
    `count` — and listed under `skipped_invalid` with its reasons. One
    that's merely unreviewed (or below `min_reward`) is listed separately
    under `skipped_unreviewed`, since that's not a defect in the session.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if include_all:
        rows = con.execute("SELECT id, task FROM sessions ORDER BY started_at ASC").fetchall()
    else:
        rows = con.execute(
            "SELECT id, task FROM sessions WHERE finetune = 1 ORDER BY started_at ASC"
        ).fetchall()

    kept: list[tuple[str, dict]] = []  # (task, record)
    skipped_invalid: list[dict[str, Any]] = []
    skipped_unreviewed: list[str] = []
    by_category: dict[str, int] = {}

    for row in rows:
        session = db.get_session(con, row["id"])
        if session is None:
            continue
        if statuses is not None and session["status"] not in statuses:
            continue
        if require_review and not session["reviewed"]:
            skipped_unreviewed.append(row["id"])
            continue
        if min_reward is not None:
            reward = session["reward"]
            if reward is None or reward < min_reward:
                skipped_unreviewed.append(row["id"])
                continue
        raw_rows = db.get_messages(con, row["id"])
        errors = validate_trajectory(raw_rows)
        if errors:
            skipped_invalid.append({"session_id": row["id"], "errors": errors})
            continue
        for k, v in classify_trajectory(raw_rows).items():
            if v is True:
                by_category[k] = by_category.get(k, 0) + 1
        kept.append((row["task"], _session_record(con, row["id"])))

    if val_frac <= 0:
        records = [record for _, record in kept]
        train_meta_path = _write_jsonl(out_path, records, strip_metadata)
        return {
            "count": len(kept), "train_count": len(kept), "val_count": 0,
            "val_path": None, "train_meta_path": str(train_meta_path) if train_meta_path else None,
            "val_meta_path": None, "skipped_invalid": skipped_invalid,
            "skipped_unreviewed": skipped_unreviewed, "by_category": by_category,
        }

    by_task: dict[str, list[dict]] = {}
    for task, record in kept:
        by_task.setdefault(task.strip(), []).append(record)
    task_keys = sorted(by_task)
    random.Random(seed).shuffle(task_keys)
    n_val_tasks = max(1, int(len(task_keys) * val_frac)) if len(task_keys) > 1 else 0
    val_keys = set(task_keys[:n_val_tasks])

    train_records = [r for k in task_keys if k not in val_keys for r in by_task[k]]
    val_records = [r for k in task_keys if k in val_keys for r in by_task[k]]

    val_path = out_path.with_name(out_path.stem + ".val" + out_path.suffix)
    train_meta_path = _write_jsonl(out_path, train_records, strip_metadata)
    val_meta_path = _write_jsonl(val_path, val_records, strip_metadata)

    return {
        "count": len(kept), "train_count": len(train_records), "val_count": len(val_records),
        "val_path": str(val_path),
        "train_meta_path": str(train_meta_path) if train_meta_path else None,
        "val_meta_path": str(val_meta_path) if val_meta_path else None,
        "skipped_invalid": skipped_invalid, "skipped_unreviewed": skipped_unreviewed,
        "by_category": by_category,
    }


# --- import ------------------------------------------------------------
#
# The counterpart to export_all: bring an `orchestrator-sft/1` record (or
# a bare `{"messages": [...]}`, e.g. --strip-metadata output) back into
# the sessions/messages tables, so it can be reviewed and re-exported
# alongside locally-generated sessions. Not a lossless round-trip -- see
# _reconstruct_message -- but validated the same way a live session is
# (validate_trajectory), so an imported session is never a second-class,
# unchecked citizen in the corpus.

_OBSERVATION_PREFIX_RE = re.compile(r"^Observation \(([^)]*)\):\n")


def _content_text_and_image(content: Any, session_id: str, seq: int, image_dir: Path) -> tuple[str, str | None]:
    """A message's `content` (string, or an OpenAI-style block list) ->
    (text, image_path). A base64 `image_url` block is decoded back to a
    real PNG/JPEG file under `image_dir`; there is nowhere else it could
    live, since the original file (in jobagent's own trace store) is very
    likely on a different machine from whoever is importing this.
    """
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        return str(content), None
    text_parts: list[str] = []
    image_path: str | None = None
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and "," in url:
                header, _, b64data = url.partition(",")
                ext = "png"
                if header.startswith("data:image/"):
                    ext = header[len("data:image/"):].split(";")[0] or "png"
                try:
                    raw = base64.b64decode(b64data)
                except Exception:
                    continue
                image_dir.mkdir(parents=True, exist_ok=True)
                dest = image_dir / f"{session_id}_{seq:03d}.{ext}"
                dest.write_bytes(raw)
                image_path = str(dest)
    return "\n".join(text_parts), image_path


def _reconstruct_message(m: dict[str, Any], session_id: str, seq: int, image_dir: Path) -> dict[str, Any]:
    """One exported message -> a row shape `db.restore_session` accepts.

    An exported `tool` observation was already folded into `role: "user"`
    with a `"Observation (<name>):\\n"` prefix (see db.to_api_messages) --
    exactly reversing that here (role becomes 'tool' again, tool_name
    extracted) is what lets validate_trajectory's tool-name check and
    classify_trajectory's category counts work the same on an imported
    session as a native one. A `user` row that doesn't match the prefix
    (a real user turn, or external data that never went through
    to_api_messages) is kept as plain 'user' rather than guessed at.
    """
    role = m.get("role")
    text, image_path = _content_text_and_image(m.get("content"), session_id, seq, image_dir)
    tool_name = None
    if role == "user":
        match = _OBSERVATION_PREFIX_RE.match(text)
        if match:
            role = "tool"
            tool_name = match.group(1)
            text = text[match.end():]
    return {"role": role, "content": text, "tool_name": tool_name, "image_path": image_path}


def import_record(
    con,
    record: dict[str, Any],
    image_dir: Path,
    overwrite: bool = False,
    mark_finetune: bool = True,
) -> dict[str, Any]:
    """Import one exported/external record. Returns `{"session_id",
    "imported": bool, "reason": str | None}` — `imported=False` for a
    duplicate (pass `overwrite=True` to replace it), a shape that isn't
    even a messages list, or one that fails `validate_trajectory`.
    """
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return {"session_id": record.get("session_id"), "imported": False,
                 "reason": "missing or empty 'messages'"}

    session_id = record.get("session_id") or uuid.uuid4().hex
    if db.get_session(con, session_id) is not None:
        if not overwrite:
            return {"session_id": session_id, "imported": False,
                     "reason": "a session with this id already exists (pass overwrite to replace)"}
        db.delete_session(con, session_id)

    rebuilt = [_reconstruct_message(m, session_id, i, image_dir) for i, m in enumerate(messages, start=1)]
    errors = validate_trajectory(rebuilt)
    if errors:
        return {"session_id": session_id, "imported": False, "reason": "; ".join(errors)}

    # `step_reviews`'s `seq` lines up with position in `messages` (1-indexed)
    # because _session_record built it from the very rows `messages` was
    # rendered from, in the same order, with no gaps -- see its own comment.
    for sr in record.get("step_reviews") or []:
        seq = sr.get("seq")
        if isinstance(seq, int) and 1 <= seq <= len(rebuilt):
            rebuilt[seq - 1]["reward"] = sr.get("reward")
            rebuilt[seq - 1]["review_note"] = sr.get("note")

    task = record.get("task")
    if not task:
        task = next((m["content"] for m in rebuilt if m["role"] == "user" and isinstance(m["content"], str)), "")
    db.restore_session(
        con, session_id, task=task, model=record.get("model", "imported"),
        status=record.get("status", "completed"), started_at=record.get("started_at") or db.utcnow(),
        messages=rebuilt, base_url=record.get("base_url", "imported"), ended_at=record.get("ended_at"),
        step_count=record.get("step_count", 0), summary=record.get("summary"), finetune=mark_finetune,
        effort=record.get("effort"), provider=record.get("provider"),
    )
    return {"session_id": session_id, "imported": True, "reason": None}


def import_records(
    con,
    records: Iterable[dict[str, Any]],
    image_dir: Path,
    overwrite: bool = False,
    mark_finetune: bool = True,
) -> dict[str, Any]:
    """Import many records (e.g. every line of an export file). Returns
    `{"imported": [session_id, ...], "skipped": [{"session_id", "reason"}, ...]}`.
    """
    imported: list[str] = []
    skipped: list[dict[str, Any]] = []
    for record in records:
        result = import_record(con, record, image_dir, overwrite=overwrite, mark_finetune=mark_finetune)
        if result["imported"]:
            imported.append(result["session_id"])
        else:
            skipped.append({"session_id": result["session_id"], "reason": result["reason"]})
    return {"imported": imported, "skipped": skipped}


def import_lines(
    con,
    lines: Iterable[str],
    image_dir: Path,
    overwrite: bool = False,
    mark_finetune: bool = True,
) -> dict[str, Any]:
    """Like `import_records`, but from raw JSONL text lines (a file's
    contents, or a paste) instead of already-parsed dicts. A line that
    isn't valid JSON is skipped and reported like any other bad record.
    """
    records: list[dict[str, Any]] = []
    bad_json: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            bad_json.append({"session_id": None, "reason": f"invalid JSON: {e}"})
    report = import_records(con, records, image_dir, overwrite=overwrite, mark_finetune=mark_finetune)
    report["skipped"] = bad_json + report["skipped"]
    return report


def import_file(
    con,
    path: Path,
    image_dir: Path,
    overwrite: bool = False,
    mark_finetune: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    return import_lines(
        con, path.read_text(encoding="utf-8").splitlines(), image_dir,
        overwrite=overwrite, mark_finetune=mark_finetune,
    )
