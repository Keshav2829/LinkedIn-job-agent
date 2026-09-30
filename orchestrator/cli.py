"""`python -m orchestrator <command>` — see orchestrator/README.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import agent, db, finetune
from .config import Config, load_config


def _cfg_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {
        "provider": getattr(args, "provider", None),
        "base_url": getattr(args, "base_url", None),
        "api_key": getattr(args, "api_key", None),
        "model": getattr(args, "model", None),
        "temperature": getattr(args, "temperature", None),
        "max_tokens": getattr(args, "max_tokens", None),
        "max_steps": getattr(args, "max_steps", None),
        "repo_root": getattr(args, "repo", None),
        "db_path": getattr(args, "db", None),
        "vision": getattr(args, "vision", None),
        "interactive": getattr(args, "interactive", None),
        "browse": getattr(args, "browse", None),
        "send_images": getattr(args, "send_images", None),
        "effort": getattr(args, "effort", None),
    }
    return load_config(overrides)


def _safe_print(text: str, file=None) -> None:
    """print(), but never crash the run over a character the console's
    codepage can't display. A local model's output, or a job/company name
    surfaced through `run_shell`, can contain anything — an em dash, a
    curly quote, non-Latin text — and plenty of Windows consoles (cmd,
    PowerShell, git-bash) still default to a legacy codepage that can't
    encode most of that. Losing the final status line of a run because of
    a glyph is worse than showing a `?` in its place."""
    stream = file or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"), file=stream)


def _print_event(kind: str, payload: dict[str, Any]) -> None:
    if kind == "session_started":
        _safe_print(f"[session {payload['session_id']}] task: {payload['task']}")
    elif kind == "session_resumed":
        _safe_print(f"\n--- resuming session {payload['session_id']} ---")
    elif kind == "assistant_message":
        _safe_print("\n--- model ---")
        _safe_print(payload["content"])
    elif kind == "tool_result":
        _safe_print(f"\n--- observation ({payload['tool']}) ---")
        _safe_print(payload["output"])
    elif kind == "session_completed":
        _safe_print(f"\n=== completed ===\n{payload['summary']}")
    elif kind == "session_failed":
        _safe_print(f"\n=== failed ===\n{payload['error']}", file=sys.stderr)
    elif kind == "session_aborted":
        _safe_print(f"\n=== aborted ===\n{payload['error']}", file=sys.stderr)
    elif kind == "session_stopped":
        _safe_print(f"\n=== stopped (by request) — retry with:\n"
                     f"python -m orchestrator sessions retry {payload['session_id']} ===")
    elif kind == "session_max_steps":
        _safe_print("\n=== stopped: max_steps reached ===")
    elif kind == "waiting_for_user":
        _safe_print("\n--- agent asks ---")
        _safe_print(payload["question"])


def _stdin_asker():
    """Builds the CLI's `on_ask` callback: the question was already printed
    by `_print_event`'s `waiting_for_user` handler (it fires before this is
    called), so this just prompts for and returns the answer."""
    def on_ask(question: str) -> str:
        try:
            return input("> ")
        except EOFError:
            return ""
    return on_ask


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    if args.task == ["-"]:
        task = sys.stdin.read().strip()
    else:
        task = " ".join(args.task)
    if not task:
        print("no task given (empty string or empty stdin)", file=sys.stderr)
        return 1
    on_ask = _stdin_asker() if cfg.interactive else None
    session_id = agent.run_session(cfg, task, finetune_flag=args.finetune, on_event=_print_event, on_ask=on_ask)
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, session_id)
    finally:
        con.close()
    print(f"\nsession_id: {session_id}  status: {session['status']}")
    return 0 if session["status"] == "completed" else 1


def cmd_sessions_list(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    con = db.connect(cfg.db_path)
    try:
        rows = db.list_sessions(con, limit=args.limit)
    finally:
        con.close()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("(no sessions yet)")
        return 0
    for r in rows:
        task = r["task"] if len(r["task"]) <= 60 else r["task"][:57] + "..."
        review = f"reward={r['reward']}" if r["reviewed"] else "unreviewed"
        _safe_print(f"{r['id']}  {r['started_at']}  {r['status']:<18} steps={r['step_count']:<3} "
                    f"{review:<12} {task}")
    return 0


def cmd_sessions_show(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, args.session_id)
        if session is None:
            print(f"no such session: {args.session_id}", file=sys.stderr)
            return 1
        messages = db.get_messages(con, args.session_id)
    finally:
        con.close()
    if args.json:
        print(json.dumps({"session": session, "messages": messages}, indent=2))
        return 0
    effort_suffix = f"  effort={session['effort']}" if session["effort"] else ""
    print(f"session {session['id']}  status={session['status']}  model={session['model']}{effort_suffix}")
    _safe_print(f"task: {session['task']}")
    print()
    for m in messages:
        header = f"[{m['seq']:02d}] {m['role']}"
        if m["tool_name"]:
            header += f" ({m['tool_name']})"
        if m["reward"] is not None:
            header += f"  [step reward={m['reward']} ({db.REWARD_LABELS.get(m['reward'], m['reward'])})]"
        print(header)
        _safe_print(m["content"])
        print()
    if session["summary"]:
        _safe_print(f"summary: {session['summary']}")
    if session["error"]:
        _safe_print(f"error: {session['error']}")
    if session["reviewed"]:
        label = db.REWARD_LABELS.get(session["reward"], session["reward"])
        _safe_print(f"review: reward={session['reward']} ({label}) at {session['reviewed_at']}"
                    + (f" -- {session['review_note']}" if session["review_note"] else ""))
    else:
        print("review: unreviewed -- "
              f"python -m orchestrator sessions review {session['id']} --reward N")
    if session["status"] == "waiting_for_user":
        print(f"\nwaiting for an answer — python -m orchestrator sessions answer {session['id']} \"...\"")
    elif session["status"] in ("failed", "aborted", "stopped"):
        print(f"\n{session['status']} — python -m orchestrator sessions retry {session['id']}")
    elif session["status"] == "running":
        print(f"\nrunning — python -m orchestrator sessions stop {session['id']} to interrupt it")
    return 0


def cmd_sessions_delete(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    con = db.connect(cfg.db_path)
    try:
        result = db.delete_session(con, args.session_id)
    finally:
        con.close()
    if result is None:
        print(f"no such session: {args.session_id}", file=sys.stderr)
        return 1

    export_path = cfg.finetune_dir / f"{args.session_id}.jsonl"
    removed_export = export_path.is_file()
    if removed_export:
        export_path.unlink()

    removed_images = 0
    if args.delete_screenshots:
        for p in result["image_paths"]:
            fp = Path(p)
            if fp.is_file():
                fp.unlink()
                removed_images += 1

    print(f"deleted session {args.session_id}: {result['message_count']} message(s)")
    if removed_export:
        print(f"  removed fine-tune export: {export_path}")
    if result["image_paths"]:
        if args.delete_screenshots:
            print(f"  removed {removed_images}/{len(result['image_paths'])} screenshot file(s)")
        else:
            print(f"  left {len(result['image_paths'])} screenshot file(s) in place "
                  "(shared with jobagent's own trace store — pass --delete-screenshots to also remove them)")
    return 0


def cmd_sessions_review(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    con = db.connect(cfg.db_path)
    try:
        if args.clear:
            ok = db.clear_review(con, args.session_id)
        else:
            if args.reward is None:
                print("--reward N (-2..2) or --clear is required", file=sys.stderr)
                return 1
            try:
                ok = db.set_review(con, args.session_id, args.reward, args.note)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 1
    finally:
        con.close()
    if not ok:
        print(f"no such session: {args.session_id}", file=sys.stderr)
        return 1
    if args.clear:
        print(f"cleared review for session {args.session_id}")
    else:
        label = db.REWARD_LABELS[args.reward]
        print(f"reviewed session {args.session_id}: reward={args.reward} ({label})")
    return 0


def cmd_sessions_step_review(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    con = db.connect(cfg.db_path)
    try:
        if args.clear:
            ok = db.clear_message_review(con, args.session_id, args.seq)
        else:
            if args.reward is None:
                print("--reward N (-2..2) or --clear is required", file=sys.stderr)
                return 1
            try:
                ok = db.set_message_review(con, args.session_id, args.seq, args.reward, args.note)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 1
    finally:
        con.close()
    if not ok:
        print(f"no such step: session {args.session_id}, seq {args.seq}", file=sys.stderr)
        return 1
    if args.clear:
        print(f"cleared step review: session {args.session_id}, seq {args.seq}")
    else:
        label = db.REWARD_LABELS[args.reward]
        print(f"reviewed step: session {args.session_id}, seq {args.seq}: reward={args.reward} ({label})")
    return 0


def cmd_sessions_import(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    image_dir = cfg.finetune_dir / "imported_images"
    con = db.connect(cfg.db_path)
    try:
        report = finetune.import_file(
            con, Path(args.file), image_dir, overwrite=args.overwrite, mark_finetune=not args.no_finetune_flag,
        )
    finally:
        con.close()
    print(f"imported {len(report['imported'])} session(s) from {args.file}")
    for sid in report["imported"]:
        print(f"  {sid}")
    if report["skipped"]:
        print(f"skipped {len(report['skipped'])}:")
        for s in report["skipped"]:
            _safe_print(f"  {s['session_id'] or '(no id)'}: {s['reason']}")
    return 0 if report["imported"] else 1


def cmd_sessions_answer(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, args.session_id)
    finally:
        con.close()
    if session is None:
        print(f"no such session: {args.session_id}", file=sys.stderr)
        return 1
    if session["status"] != "waiting_for_user":
        print(f"session is not waiting for an answer (status: {session['status']})", file=sys.stderr)
        return 1

    agent.answer_question(cfg, args.session_id, args.answer)
    on_ask = _stdin_asker()  # further pauses in the same `sessions answer` call stay interactive
    session_id = agent.drive_interactively(cfg, args.session_id, on_event=_print_event, on_ask=on_ask)

    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, session_id)
    finally:
        con.close()
    print(f"\nsession_id: {session_id}  status: {session['status']}")
    return 0 if session["status"] == "completed" else 1


def cmd_sessions_retry(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, args.session_id)
    finally:
        con.close()
    if session is None:
        print(f"no such session: {args.session_id}", file=sys.stderr)
        return 1
    if session["status"] not in ("failed", "aborted"):
        print(f"session is not in a retryable state (status: {session['status']})", file=sys.stderr)
        return 1

    agent.retry_session(cfg, args.session_id)
    on_ask = _stdin_asker()  # a pause after the retry stays interactive, same as `run`/`sessions answer`
    session_id = agent.drive_interactively(cfg, args.session_id, on_event=_print_event, on_ask=on_ask)

    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, session_id)
    finally:
        con.close()
    print(f"\nsession_id: {session_id}  status: {session['status']}")
    return 0 if session["status"] == "completed" else 1


def cmd_sessions_stop(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    con = db.connect(cfg.db_path)
    try:
        session = db.get_session(con, args.session_id)
    finally:
        con.close()
    if session is None:
        print(f"no such session: {args.session_id}", file=sys.stderr)
        return 1
    if session["status"] != "running":
        print(f"session is not running (status: {session['status']})", file=sys.stderr)
        return 1

    agent.request_stop(cfg, args.session_id)
    print(f"stop requested for session {args.session_id} -- it will stop at its next step "
          "boundary (won't interrupt a model call or tool already in flight). "
          f"python -m orchestrator sessions retry {args.session_id} resumes it.")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from . import ui

    cfg = _cfg_from_args(args)
    ui.serve(cfg, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_finetune_export(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    statuses = None if args.any_status else tuple(args.status or ["completed"])
    con = db.connect(cfg.db_path)
    try:
        report = finetune.export_all(
            con, Path(args.out), include_all=args.all,
            statuses=statuses, val_frac=args.val_frac,
            strip_metadata=args.strip_metadata,
            require_review=args.require_review, min_reward=args.min_reward,
        )
    finally:
        con.close()
    print(f"wrote {report['train_count']} session(s) to {args.out}")
    if report["train_meta_path"]:
        print(f"  (metadata stripped -> {report['train_meta_path']})")
    if report["val_path"]:
        print(f"wrote {report['val_count']} session(s) to {report['val_path']}")
        if report["val_meta_path"]:
            print(f"  (metadata stripped -> {report['val_meta_path']})")
    if report["skipped_invalid"]:
        print(f"skipped {len(report['skipped_invalid'])} invalid session(s):")
        for s in report["skipped_invalid"]:
            _safe_print(f"  {s['session_id']}: {'; '.join(s['errors'])}")
    if report["skipped_unreviewed"]:
        print(f"skipped {len(report['skipped_unreviewed'])} unreviewed/below-min-reward session(s)")
    if report["by_category"]:
        print("trajectory mix (a session can count in more than one row):")
        for k in sorted(report["by_category"]):
            print(f"  {k}: {report['by_category'][k]}")
    return 0


def cmd_finetune_validate(args: argparse.Namespace) -> int:
    report = finetune.validate_export_file(Path(args.file))
    print(f"{report['path']}: {report['valid']}/{report['total']} valid")
    for p in report["problems"]:
        _safe_print(f"  line {p['line']}: {'; '.join(p.get('errors') or [])}")
    return 0 if report["invalid"] == 0 else 1


def cmd_config_show(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    print(json.dumps(cfg.masked(), indent=2))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    if args.memory_action in ("get", "set") and not args.key:
        print(f"'{args.memory_action}' requires a key", file=sys.stderr)
        return 1
    if args.memory_action == "set" and args.value is None:
        print("'set' requires a value", file=sys.stderr)
        return 1
    con = db.connect(cfg.db_path)
    try:
        if args.memory_action == "get":
            val = db.memory_get(con, args.key)
            print(json.dumps(val) if val is not None else "null")
        elif args.memory_action == "set":
            try:
                value: Any = json.loads(args.value)
            except json.JSONDecodeError:
                value = args.value
            db.memory_set(con, args.key, value)
            _safe_print(f"OK: remembered '{args.key}'.")
        elif args.memory_action == "list":
            print(json.dumps(db.memory_list(con), indent=2))
    finally:
        con.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orchestrator", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provider", choices=["openai", "anthropic", "claude-cli"],
        help="which model to drive this session with (default: openai, any OpenAI-compatible "
             "endpoint). 'anthropic' calls Claude's native Messages API (needs --api-key or "
             "$ANTHROPIC_API_KEY). 'claude-cli' shells out to the `claude` CLI instead -- no API "
             "key, just Claude Code installed and logged in; --model should be a CLI model alias "
             "like 'sonnet'/'opus'/'haiku', not an API model id",
    )
    common.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1 "
                                            "(ignored for --provider anthropic unless you're proxying it)")
    common.add_argument("--api-key", help="API key for the endpoint (vLLM default: EMPTY; for "
                                           "--provider anthropic, falls back to $ANTHROPIC_API_KEY)")
    common.add_argument("--model", help="model name as the server expects it")
    common.add_argument("--repo", help="path to the linkedinAgent project root (default: parent of this folder)")
    common.add_argument("--db", help="path to the orchestrator's own sqlite db")
    common.add_argument(
        "--vision", action="store_true", default=None,
        help="enable the screenshot tool (needs a vision-capable local model and a connected browser driver)",
    )
    common.add_argument(
        "--interactive", action="store_true", default=None,
        help="let the agent pause and ask a question instead of guessing (run: prompts on stdin)",
    )
    common.add_argument(
        "--browse", action="store_true", default=None,
        help="let the agent drive `jobagent browse ...` directly via run_shell, instead of "
             "declining and delegating browser actions. Submission still requires an explicit "
             "`job approve`/`outreach approve` when apply_requires_approval/outreach_requires_"
             "approval is on -- see orchestrator/README.md",
    )
    common.add_argument(
        "--no-send-images", dest="send_images", action="store_false", default=None,
        help="replay any earlier screenshot as text only, without the actual image -- unlike "
             "--vision (which only gates whether the screenshot tool exists), this can be set on "
             "a *resume* call even for a session that took a screenshot earlier, to recover a "
             "session stuck 400ing against a model/endpoint that doesn't actually accept image "
             "input (e.g. 'messages contain images, but <model> does not support image inputs')",
    )
    common.add_argument(
        "--effort", choices=db.EFFORT_LEVELS,
        help="reasoning effort for --provider claude-cli (low/medium/high/xhigh/max; see `claude "
             "--help`). Unset by default -- the claude CLI then falls back to your ambient "
             "~/.claude/settings.json (or its own built-in default), which is fine interactively "
             "but means two 'identical' sessions on different machines (or after a settings "
             "change) could silently be generated at different effort levels. Set this to make it "
             "explicit and reproducible; ignored by --provider openai/anthropic",
    )

    sub = p.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", parents=[common], help="run one session against a task")
    p_run.add_argument("task", nargs="+", help="the task to give the agent, or '-' to read from stdin")
    p_run.add_argument("--finetune", action="store_true", help="also export this session as fine-tuning data on completion")
    p_run.add_argument("--temperature", type=float)
    p_run.add_argument("--max-tokens", type=int)
    p_run.add_argument("--max-steps", type=int)
    p_run.set_defaults(func=cmd_run)

    p_sessions = sub.add_parser("sessions", help="inspect past sessions")
    sessions_sub = p_sessions.add_subparsers(dest="sessions_command", required=True)

    p_list = sessions_sub.add_parser("list", parents=[common], help="list recent sessions")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_sessions_list)

    p_show = sessions_sub.add_parser("show", parents=[common], help="show one session's transcript")
    p_show.add_argument("session_id")
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=cmd_sessions_show)

    p_answer = sessions_sub.add_parser("answer", parents=[common], help="answer a session paused on a question")
    p_answer.add_argument("session_id")
    p_answer.add_argument("answer")
    p_answer.set_defaults(func=cmd_sessions_answer)

    p_retry = sessions_sub.add_parser(
        "retry", parents=[common], help="resume a session that ended `failed`, `aborted`, or `stopped`",
    )
    p_retry.add_argument("session_id")
    p_retry.set_defaults(func=cmd_sessions_retry)

    p_stop = sessions_sub.add_parser(
        "stop", parents=[common], help="ask a running session to stop at its next step boundary",
    )
    p_stop.add_argument("session_id")
    p_stop.set_defaults(func=cmd_sessions_stop)

    p_delete = sessions_sub.add_parser(
        "delete", parents=[common], help="permanently delete a session, its messages, and its fine-tune export",
    )
    p_delete.add_argument("session_id")
    p_delete.add_argument(
        "--delete-screenshots", action="store_true",
        help="also delete this session's screenshot files from jobagent's shared trace store "
             "(off by default -- the same file may be referenced by jobagent's own harvest corpus)",
    )
    p_delete.set_defaults(func=cmd_sessions_delete)

    p_review = sessions_sub.add_parser(
        "review", parents=[common],
        help="record a human quality verdict for a session, for finetune export's --require-review/--min-reward",
    )
    p_review.add_argument("session_id")
    p_review.add_argument(
        "--reward", type=int, choices=sorted(db.REWARD_LABELS),
        help="-2 unusable, -1 bad, 0 neutral, 1 good, 2 exemplary (same scale annotation_suite uses)",
    )
    p_review.add_argument("--note", help="optional free-text note (why, what to fix, ...)")
    p_review.add_argument("--clear", action="store_true", help="reset back to unreviewed instead of setting one")
    p_review.set_defaults(func=cmd_sessions_review)

    p_step_review = sessions_sub.add_parser(
        "step-review", parents=[common],
        help="record a per-step (per-message) quality verdict -- a finer-grained, purely "
             "diagnostic layer; finetune export still gates on `sessions review`, not this",
    )
    p_step_review.add_argument("session_id")
    p_step_review.add_argument("seq", type=int, help="the message's seq number, from `sessions show`")
    p_step_review.add_argument("--reward", type=int, choices=sorted(db.REWARD_LABELS))
    p_step_review.add_argument("--note", help="optional free-text note")
    p_step_review.add_argument("--clear", action="store_true")
    p_step_review.set_defaults(func=cmd_sessions_step_review)

    p_import = sessions_sub.add_parser(
        "import", parents=[common],
        help="load external session data -- an orchestrator-sft/1 export (this project's own, or "
             "someone else's) -- back into sessions/messages so it can be reviewed and re-exported",
    )
    p_import.add_argument("file", help="a JSONL file: one orchestrator-sft/1 record (or bare "
                                        "{'messages': [...]}) per line")
    p_import.add_argument(
        "--overwrite", action="store_true",
        help="replace a session that already exists under the same session_id (default: skip it)",
    )
    p_import.add_argument(
        "--no-finetune-flag", action="store_true",
        help="don't flag imported sessions for finetune export's default (--finetune-flagged-only) "
             "view -- they'll still show up with --all",
    )
    p_import.set_defaults(func=cmd_sessions_import)

    p_ui = sub.add_parser("ui", parents=[common], help="serve the local web UI")
    p_ui.add_argument("--port", type=int, default=8787)
    p_ui.add_argument("--no-browser", action="store_true")
    p_ui.set_defaults(func=cmd_ui)

    p_finetune = sub.add_parser("finetune", help="fine-tuning data export")
    finetune_sub = p_finetune.add_subparsers(dest="finetune_command", required=True)
    p_ft_export = finetune_sub.add_parser("export", parents=[common], help="combine sessions into one JSONL file")
    p_ft_export.add_argument("--out", default="orchestrator/data/finetune/export.jsonl")
    p_ft_export.add_argument("--all", action="store_true", help="include every session, not just --finetune-flagged ones")
    p_ft_export.add_argument(
        "--status", action="append",
        help="only sessions in this status (repeatable; default: completed only)",
    )
    p_ft_export.add_argument(
        "--any-status", action="store_true",
        help="include sessions in any status, not just completed (overrides --status)",
    )
    p_ft_export.add_argument(
        "--val-frac", type=float, default=0.0,
        help="also write a validation split grouped by task (default: 0, single file)",
    )
    p_ft_export.add_argument(
        "--strip-metadata", action="store_true",
        help="write bare {'messages': [...]} lines (for a hosted API/strict trainer that rejects "
             "unknown keys); the stripped fields go to a same-order <file>.meta.jsonl sidecar",
    )
    p_ft_export.add_argument(
        "--require-review", action="store_true",
        help="only include sessions reviewed via `sessions review` (see the UI's review panel too)",
    )
    p_ft_export.add_argument(
        "--min-reward", type=int, choices=sorted(db.REWARD_LABELS),
        help="only include sessions with a review reward >= this (implies reviewed -- an "
             "unreviewed session's reward is NULL, never 0, so it never silently passes this)",
    )
    p_ft_export.set_defaults(func=cmd_finetune_export)

    p_ft_validate = finetune_sub.add_parser("validate", help="validate an already-exported JSONL file")
    p_ft_validate.add_argument("file")
    p_ft_validate.set_defaults(func=cmd_finetune_validate)

    p_config = sub.add_parser("config", help="configuration")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    p_config_show = config_sub.add_parser("show", parents=[common], help="print the effective config (api key masked)")
    p_config_show.set_defaults(func=cmd_config_show)

    p_memory = sub.add_parser("memory", parents=[common], help="the agent's own persistent memory")
    p_memory.add_argument("memory_action", choices=["get", "set", "list"])
    p_memory.add_argument("key", nargs="?")
    p_memory.add_argument("value", nargs="?")
    p_memory.set_defaults(func=cmd_memory)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
