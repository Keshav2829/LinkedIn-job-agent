"""The agent loop: system prompt, action parsing, tool dispatch.

## Why a text protocol instead of native OpenAI function-calling

The whole point of this project is "works with whatever local model you
happen to be serving." Native `tools=[...]` function-calling only works
when the serving stack has a tool-call parser wired up for that specific
model family (vLLM needs `--enable-auto-tool-choice --tool-call-parser ...`
per model), and plenty of local setups (llama.cpp's server, a base model,
an older vLLM build) don't have one. A plain-text protocol that any chat
model can follow — "end your reply with one fenced ```action block of
JSON" — works identically against every OpenAI-compatible endpoint,
degrades to a visible, debuggable parse error instead of a silent 400, and
is exactly the kind of thing you can fine-tune a small local model to do
reliably (see finetune.py). If your server *does* support tool-calling well
and you'd rather use it, that's a reasonable follow-up; this file is the
seam where you'd add it.

## The loop

1. Render the system prompt (identity, tool docs, protocol, ground rules).
2. Send the transcript so far; get one assistant reply.
3. Parse the trailing ```action block:
   - `{"tool": "<name>", "input": {...}}` -> run the tool, append the
     observation as the next turn, go to 2.
   - `{"final": "<message>"}` -> end the session, that message is the summary.
   - `{"ask": "<question>"}` -> only when `config.interactive` — pause the
     session (`status: waiting_for_user`) instead of calling the model
     again; some external caller (the CLI's `input()` prompt, or the UI's
     answer box) supplies the answer later via `answer_question`, and
     whoever calls `drive_session` again picks the loop back up. When
     `interactive` is off, asking is treated like any other disallowed
     action: an error observation, not a pause — a scheduled or unattended
     run can never get stuck waiting on someone who isn't there.
   - Missing/invalid -> append a protocol-error observation and retry, up to
     `max_consecutive_errors`.
4. Stop at `max_steps` regardless, so a confused model can't loop forever.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable, Optional

from . import db, finetune, tools
from .config import Config
from .llm import LLMError, build_client
from .tools import ToolContext, ToolResult

EventFn = Optional[Callable[[str, dict[str, Any]], None]]


def build_system_prompt(cfg: Config) -> str:
    if cfg.browse:
        browse_rule = (
            "- You may drive the browser directly with `jobagent browse ...` via `run_shell` "
            "(see `plugin/references/browser-playbook.md` for the verbs and "
            "`plugin/references/cli.md` for exact syntax) — check `jobagent browse status` first, "
            "and if no driver is connected, say so plainly rather than guessing at what the page "
            "shows or faking a result. `job queue`/`outreach next` will never hand you a job or "
            "message to actually submit unless it's been `approve`d — if `apply_requires_approval` "
            "or `outreach_requires_approval` leaves one empty, call `job pending`/`outreach pending`, "
            "report what's waiting on a decision, and stop there; you do not have standing "
            "authorization to approve your own submissions."
        )
    else:
        browse_rule = (
            "- If a skill's instructions call for a browser action, you cannot perform it — say so "
            "in your final answer and name which skill needs a human or a browser-capable agent "
            "(e.g. the `linkedin-agent` Claude Code subagent, or re-run this session with "
            "`--browse`) to finish it. Do not attempt to fake browser output."
        )
    vision_rule = (
        "\n- You have a `screenshot` tool. Use it when you need to see the page "
        "rather than guess from text — e.g. to confirm a click landed, or to read "
        "something `jobagent`'s text tools didn't surface. Don't call it "
        "speculatively on every step; each screenshot stays in the conversation "
        "and costs context on every later turn."
        if cfg.vision else ""
    )
    if cfg.interactive:
        talk_rule = (
            "You may ask the person who started this task a question and wait "
            "for their answer — but only when you're genuinely blocked on "
            "something only they can decide (an ambiguous choice, a missing "
            "credential, confirmation before something you can't undo). Don't "
            "ask about anything `jobagent`'s own tools can already answer, and "
            "don't ask the same thing twice."
        )
        ask_block = """
To ask a question and wait for the answer instead of continuing:
```action
{"ask": "<your question>"}
```
"""
    else:
        talk_rule = (
            "You do not talk to the user mid-task — if you are blocked, say so "
            "in your final answer instead of guessing or waiting."
        )
        ask_block = ""
    return f"""You are a local orchestrator agent for the LinkedIn Job Agent project,
checked out at: {cfg.repo_root}

Your only job is to carry out the task you are given by reading that
project's skills (procedures written for a human or another AI to follow)
and running its command-line tool, `jobagent`, the same way a person would
from a terminal. You do not have any other capability. {talk_rule}

## Response protocol

Every reply you send must end with exactly one fenced block tagged
`action` containing a single JSON object, and nothing after it. You may
write reasoning before the block; only the block is acted on.

To call a tool:
```action
{{"tool": "<tool_name>", "input": {{...}}}}
```

To end the task and report back:
```action
{{"final": "<your report to whoever asked for this task>"}}
```
{ask_block}
Never emit more than one ```action block per reply. Never invent a tool
result — wait for the real observation before continuing.

## Tools

{tools.tool_help_text(cfg)}

## Ground rules (carried over from the LinkedIn agent's own rules)

- Never guess a form answer or a fact you don't have; if a `jobagent`
  command reports it needs a human, record that in your final answer
  instead of working around it.
- Never approve or force-send outreach messages, and never approve your own
  job applications — `outreach approve`/`job approve` exist for a human
  decision; only bypass either if the task explicitly says the user already
  disabled approval for this run.
- Prefer the narrow `jobagent report ...` / `jobagent run ...` commands over
  re-deriving state yourself; the database is the source of truth.
{browse_rule}
- Keep your final answer short and specific: what you ran, what it
  returned, and what (if anything) still needs a human.{vision_rule}
"""


def start_session(cfg: Config, task: str, finetune_flag: bool = False) -> str:
    """Create the session row and seed messages. Fast, synchronous, DB-only —
    split out from drive_session so a caller (e.g. the UI) can hand back a
    session id immediately and run the actual loop in a background thread."""
    con = db.connect(cfg.db_path)
    try:
        session_id = db.create_session(
            con, task, cfg.model, cfg.base_url, finetune_flag,
            vision=cfg.vision, interactive=cfg.interactive, browse=cfg.browse, effort=cfg.effort,
            provider=cfg.provider,
        )
        db.add_message(con, session_id, "system", build_system_prompt(cfg))
        db.add_message(con, session_id, "user", task)
        return session_id
    finally:
        con.close()


def drive_session(cfg: Config, session_id: str, on_event: EventFn = None) -> None:
    """Run the loop for an already-started session to completion."""
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, session_id)
        if session is None:
            raise ValueError(f"no such session: {session_id}")
        # `vision`/`interactive`/`browse`/`effort`/`provider`/`model`/
        # `base_url` are all pinned to the session at creation time (see
        # start_session) rather than taken from whatever `cfg` this
        # particular call was handed — a paused session resumed from a
        # different call site (UI answer endpoint, CLI, a future cron
        # resume) must not have its capabilities, its generation quality/
        # cost, or -- the sharpest version of this bug -- *which model
        # answers it at all* silently change mid-conversation just because
        # the caller's config differs. `model`/`base_url` were columns on
        # `sessions` from the start but were never actually read back here
        # before this fix, so e.g. `sessions answer <id> "..."` (or the
        # UI's answer box) with no flags re-specified would silently fall
        # back to `orchestrator run`'s *default* config -- a different
        # provider, a different model, sometimes one that doesn't even
        # support what's already in the session's history (the concrete
        # symptom that surfaced this: an HTTP 400 "messages contain images,
        # but <default model> does not support image inputs", from a
        # completely different endpoint than the one that actually started
        # the session). A pre-migration-8 session has no stored `provider`
        # (NULL) -- back then "openai" was the only option, so that's the
        # fallback, not a guess. Plain `dataclasses.replace`, not
        # `with_overrides`, deliberately: that helper drops `None` values
        # (right for turning CLI args into overrides, where None means
        # "flag wasn't passed, don't touch this") -- but a pinned field
        # being None/unset (effort, or a legacy session's provider) is
        # itself a real, meaningful value that must still win over whatever
        # the resuming cfg happens to carry, not be silently skipped the
        # way with_overrides would skip it. `api_key` is deliberately not
        # pinned here -- it's a secret, never stored on the session row in
        # the first place (see db.py migration 8's comment) -- so it still
        # comes from whatever the resuming cfg provides, same as always.
        cfg = replace(
            cfg, vision=bool(session["vision"]), interactive=bool(session["interactive"]),
            browse=bool(session["browse"]), effort=session["effort"],
            provider=session["provider"] or "openai", model=session["model"],
            base_url=session["base_url"],
        )
        client = build_client(cfg)
        ctx = ToolContext(repo_root=cfg.repo_root, con=con, session_id=session_id, config=cfg)
        registry = tools.available_tools(cfg)

        def emit(kind: str, payload: dict[str, Any]) -> None:
            if on_event:
                on_event(kind, payload)

        if session["step_count"] == 0:
            emit("session_started", {"session_id": session_id, "task": session["task"]})
        else:
            emit("session_resumed", {"session_id": session_id, "task": session["task"]})
        consecutive_errors = 0

        # A resumed session (after `waiting_for_user`) re-enters this loop
        # from `range(0)`, not `range(max_steps)` — the budget is total
        # steps for the session, not per `drive_session` call.
        remaining_steps = max(0, cfg.max_steps - session["step_count"])
        for _ in range(remaining_steps):
            # Checked once per step, not mid-call -- see migration 9's
            # comment for why a stop request can't preempt a model call or
            # tool already in flight. This still stops *before* spending
            # the next (potentially slow/costly) model call, same boundary
            # max_steps and the consecutive-error limits already use.
            if db.is_stop_requested(con, session_id):
                db.end_session(con, session_id, "stopped")
                emit("session_stopped", {"session_id": session_id})
                return
            # Proof of life for this step, *before* the model call that
            # might hang or take a while -- see migration 10's comment.
            # Lets `/api/sessions/<id>/stop` tell a merely-slow step apart
            # from a session whose driving thread/process no longer exists
            # (e.g. the UI server was restarted mid-session) and force-end
            # the latter instead of setting a flag nothing will ever read.
            db.touch_heartbeat(con, session_id)

            # Unlike vision/interactive/browse, `send_images` is *not*
            # pinned to the session — it only ever removes what gets sent,
            # never grants a new capability, so it's safe (and useful) to
            # vary per resume call. This is what lets a session that took a
            # screenshot earlier (vision was on then) still be resumed
            # against a model/endpoint that turns out not to actually
            # accept image input -- pass --no-send-images (CLI) or
            # send_images: false (UI answer override) on the resuming
            # call, without needing the session to have been started that
            # way. See db.to_api_messages's include_images param.
            api_messages = db.to_api_messages(db.get_messages(con, session_id), include_images=cfg.send_images)
            try:
                reply = client.chat(api_messages)
            except LLMError as e:
                db.end_session(con, session_id, "failed", error=str(e))
                emit("session_failed", {"session_id": session_id, "error": str(e)})
                return

            db.add_message(con, session_id, "assistant", reply)
            db.bump_step(con, session_id)
            emit("assistant_message", {"session_id": session_id, "content": reply})

            action = tools.extract_action(reply)

            if action is None:
                consecutive_errors += 1
                obs = (
                    "ERROR: your reply did not end with exactly one valid "
                    "```action fenced JSON block (either {\"tool\":...,\"input\":...} "
                    "or {\"final\": \"...\"}). Emit one now."
                )
                db.add_message(con, session_id, "tool", obs, tool_name="_protocol")
                emit("tool_result", {"session_id": session_id, "tool": "_protocol", "output": obs})
                if consecutive_errors >= cfg.max_consecutive_errors:
                    db.end_session(con, session_id, "aborted", error="repeated protocol violations")
                    emit("session_aborted", {"session_id": session_id, "error": "repeated protocol violations"})
                    return
                continue

            if "final" in action:
                summary = str(action["final"])
                db.end_session(con, session_id, "completed", summary=summary)
                emit("session_completed", {"session_id": session_id, "summary": summary})
                if session["finetune"]:
                    finetune.export_session(con, session_id, cfg.finetune_dir)
                return

            if "ask" in action:
                question = str(action["ask"])
                if not cfg.interactive:
                    consecutive_errors += 1
                    obs = (
                        "ERROR: asking questions is disabled for this session. "
                        "Record the gap in your final answer instead of waiting for a person."
                    )
                    db.add_message(con, session_id, "tool", obs, tool_name="_ask_user")
                    emit("tool_result", {"session_id": session_id, "tool": "_ask_user", "output": obs, "image_path": None})
                    if consecutive_errors >= cfg.max_consecutive_errors:
                        db.end_session(con, session_id, "aborted", error="repeated disallowed asks")
                        emit("session_aborted", {"session_id": session_id, "error": "repeated disallowed asks"})
                        return
                    continue
                db.add_message(con, session_id, "tool", question, tool_name="_ask_user")
                db.set_status(con, session_id, "waiting_for_user")
                emit("waiting_for_user", {"session_id": session_id, "question": question})
                return

            tool_name = str(action.get("tool", ""))
            tool_input = action.get("input") or {}
            if not isinstance(tool_input, dict):
                tool_input = {}
            fn = registry.get(tool_name)
            image_path: str | None = None
            if fn is None:
                consecutive_errors += 1
                obs = f"ERROR: unknown tool '{tool_name}'. Valid tools: {', '.join(registry)}."
            else:
                try:
                    result = fn(ctx, tool_input)
                except Exception as e:  # tool bugs should not crash the whole session
                    consecutive_errors += 1
                    obs = f"ERROR running {tool_name}: {e}"
                else:
                    if isinstance(result, ToolResult):
                        obs = result.text
                        image_path = str(result.image_path) if result.image_path else None
                    else:
                        obs = result
                    consecutive_errors = 0 if not obs.startswith("ERROR") else consecutive_errors + 1

            db.add_message(
                con, session_id, "tool", obs,
                tool_name=tool_name, tool_input=json.dumps(tool_input), image_path=image_path,
            )
            emit("tool_result", {"session_id": session_id, "tool": tool_name, "output": obs, "image_path": image_path})

            if consecutive_errors >= cfg.max_consecutive_errors:
                db.end_session(con, session_id, "aborted", error="repeated tool errors")
                emit("session_aborted", {"session_id": session_id, "error": "repeated tool errors"})
                return

        db.end_session(con, session_id, "max_steps_reached")
        emit("session_max_steps", {"session_id": session_id})
    finally:
        con.close()


def pending_question(session: dict[str, Any], messages: list[dict[str, Any]]) -> Optional[str]:
    """The question a `waiting_for_user` session is paused on, or None."""
    if session["status"] != "waiting_for_user":
        return None
    for m in reversed(messages):
        if m["tool_name"] == "_ask_user":
            return m["content"]
    return None


def answer_question(cfg: Config, session_id: str, answer: str) -> None:
    """Resolve a `waiting_for_user` pause: append the answer as a `user`
    turn (the same role the original task was given in) and flip the
    session back to `running`. Doesn't itself resume the loop — call
    `drive_session` (or `drive_interactively`) again after this."""
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, session_id)
        if session is None:
            raise ValueError(f"no such session: {session_id}")
        if session["status"] != "waiting_for_user":
            raise ValueError(
                f"session {session_id} is not waiting for an answer (status: {session['status']})"
            )
        db.add_message(con, session_id, "user", answer)
        db.set_status(con, session_id, "running")
    finally:
        con.close()


def retry_session(cfg: Config, session_id: str) -> None:
    """Resume a session that ended in `failed`, `aborted`, or `stopped` by
    flipping it back to `running`, without touching its message history.
    Doesn't itself resume the loop — call `drive_session` (or
    `drive_interactively`) again after this, same as `answer_question`.

    This is safe to just replay because of exactly where these statuses get
    set in `drive_session`:

    - `failed` comes from `client.chat()` raising `LLMError` (a network
      timeout, a bad/missing api_key, ...) -- that happens *before*
      `db.add_message` appends anything for the attempt, so the stored
      transcript is unchanged and the very next `client.chat()` call sends
      the exact same messages that failed last time. A fixed api_key or a
      working network is genuinely all that's needed.
    - `aborted` comes from too many *consecutive* protocol/tool errors --
      `consecutive_errors` is a local variable inside `drive_session`, reset
      to 0 on every fresh call, so retrying hands the model a brand new
      error budget against a transcript that already shows it what went
      wrong, rather than a blank slate.
    - `stopped` comes from `request_stop`, checked only at the top of a
      step (before the next model call) -- so, same as `failed`, nothing
      partial was ever appended for the step that got skipped.

    `completed`/`max_steps_reached`/`waiting_for_user` are not retryable
    here: the first two are a real finish, not an interruption, and the
    third already has its own resume path (`answer_question`)."""
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, session_id)
        if session is None:
            raise ValueError(f"no such session: {session_id}")
        if session["status"] not in ("failed", "aborted", "stopped"):
            raise ValueError(
                f"session {session_id} is not in a retryable state (status: {session['status']})"
            )
        db.set_status(con, session_id, "running", clear_error=True)
    finally:
        con.close()


def request_stop(cfg: Config, session_id: str) -> None:
    """Ask a `running` session to stop at its next step boundary (see
    migration 9's comment on `stop_requested` for why this can't preempt a
    model call or tool already in flight). Usually just sets the flag --
    whatever thread/process is actually running `drive_session` for this
    session notices it and ends the session itself with status `stopped`,
    resumable later via `retry_session`.

    But that only works if something is actually still looping. A session
    can be orphaned -- its driving UI thread died with a server restart, or
    its driving CLI process was killed -- and left stuck showing `running`
    forever with no one left to read the flag. `heartbeat_at` (touched once
    per step, see migration 10) is how this tells "a live loop just hasn't
    gotten to the flag yet" apart from "nothing is driving this any more";
    for the latter, this ends the session itself right here instead of
    leaving a flag nothing will ever see."""
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, session_id)
        if session is None:
            raise ValueError(f"no such session: {session_id}")
        if session["status"] != "running":
            raise ValueError(
                f"session {session_id} is not running (status: {session['status']})"
            )
        db.request_stop(con, session_id)
        stale_after = max(600, cfg.request_timeout * 3)
        if db.heartbeat_is_stale(con, session_id, stale_after):
            db.end_session(con, session_id, "stopped", error=(
                f"stopped: no heartbeat in over {stale_after}s -- nothing appears to "
                "still be driving this session (most likely an `orchestrator ui` "
                "restart, or a CLI process ending, mid-session). Resume it with retry."
            ))
    finally:
        con.close()


def drive_interactively(
    cfg: Config, session_id: str, on_event: EventFn = None,
    on_ask: Optional[Callable[[str], str]] = None,
) -> str:
    """Drive an existing session to a terminal state, resolving any
    `waiting_for_user` pauses along the way via `on_ask(question) -> answer`.

    Without `on_ask` this is just `drive_session` with a return value: it
    runs once and returns whatever state the session is in, including
    `waiting_for_user` — the right behavior for the UI, where a pause has
    to surface to a person through the browser rather than block a request
    thread. With `on_ask` (the CLI's `input()` prompt) it loops, answering
    each pause itself, so `run` reads as one uninterrupted session from the
    terminal even though it's several `drive_session` calls underneath.
    """
    while True:
        drive_session(cfg, session_id, on_event=on_event)
        con = db.connect(cfg.db_path)
        try:
            session = db.get_session(con, session_id)
            messages = db.get_messages(con, session_id)
        finally:
            con.close()
        if session["status"] != "waiting_for_user" or on_ask is None:
            return session_id
        question = pending_question(session, messages) or ""
        answer = on_ask(question)
        answer_question(cfg, session_id, answer)


def run_session(
    cfg: Config, task: str, finetune_flag: bool = False, on_event: EventFn = None,
    on_ask: Optional[Callable[[str], str]] = None,
) -> str:
    """Convenience wrapper for the CLI: start + drive (+ answer, if asked) in
    one call, synchronously."""
    session_id = start_session(cfg, task, finetune_flag)
    return drive_interactively(cfg, session_id, on_event=on_event, on_ask=on_ask)
