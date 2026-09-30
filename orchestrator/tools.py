"""The tool surface the local model is allowed to use.

Kept deliberately small and read-mostly. The orchestrator's job is to *read*
the LinkedIn agent's skills (plain files under `plugin/skills/`) and drive
its CLI the way a human would from a terminal — it never imports `jobagent`
or touches `data/state.db` directly. `run_shell` is the only tool that can
change anything, and it is allowlisted to `python -m jobagent ...` /
`python -m orchestrator ...` plus a handful of read-only commands, so a
confused or adversarial local model cannot turn "run today's job search"
into "delete the repo."
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union

from . import db
from .config import Config

MAX_OUTPUT_CHARS = 6000
MAX_FILE_CHARS = 20000

SAFE_SHELL_PREFIXES = (
    "python -m jobagent",
    "python3 -m jobagent",
    "py -m jobagent",
    "python -m orchestrator",
    "python3 -m orchestrator",
    "py -m orchestrator",
    "git status",
    "git log",
    "git diff",
    "git branch",
)
SAFE_SHELL_EXACT = {"dir", "ls", "ls -la", "ls -l", "pwd", "cd"}

# --- response-protocol helpers ------------------------------------------
#
# The one place that knows what a valid ```action block looks like. Lives
# here (not in agent.py, where it's used to drive the live loop) so
# finetune.py can import it too, to validate/classify exported trajectories
# against the exact same parser the live loop used to produce them --
# without a circular import (agent.py already imports finetune.py).

ACTION_BLOCK_RE = re.compile(r"```(?:action)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_action(reply: str) -> dict[str, Any] | None:
    """Pull the one trailing ```action block out of a model reply, or None."""
    matches = ACTION_BLOCK_RE.findall(reply)
    if not matches:
        return None
    for candidate in reversed(matches):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and ("tool" in parsed or "final" in parsed or "ask" in parsed):
            return parsed
    return None


def is_shell_allowed(command: str) -> bool:
    cmd = command.strip()
    if not cmd:
        return False
    if cmd in SAFE_SHELL_EXACT:
        return True
    return any(cmd.startswith(prefix) for prefix in SAFE_SHELL_PREFIXES)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"


@dataclass
class ToolContext:
    repo_root: Path
    con: Any  # sqlite3.Connection, orchestrator's own db
    session_id: str
    config: Config


@dataclass
class ToolResult:
    """A tool result that also carries an image for the model to see.

    Every other tool just returns a `str`. Only `screenshot` (gated behind
    `config.vision`) returns one of these, because it's the one case where
    text alone can't carry what the tool produced — see agent.py for how
    the image gets attached to the next turn as an OpenAI-style
    `image_url` content block.
    """
    text: str
    image_path: Path | None = None


def _parse_skill_frontmatter(text: str) -> dict[str, str]:
    """Pull `name:` / `description:` out of a SKILL.md's `---` frontmatter.

    Intentionally not a YAML parser — these files only ever use flat
    `key: value` lines here, and pulling in PyYAML for two fields would be
    the first third-party dependency in a project that has none.
    """
    out: dict[str, str] = {}
    if not text.startswith("---"):
        return out
    end = text.find("\n---", 3)
    if end == -1:
        return out
    block = text[3:end]
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def tool_list_skills(ctx: ToolContext, input: dict[str, Any]) -> str:
    skills_dir = ctx.repo_root / "plugin" / "skills"
    if not skills_dir.is_dir():
        return f"No skills directory found at {skills_dir}"
    lines = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        meta = _parse_skill_frontmatter(skill_md.read_text(encoding="utf-8", errors="replace"))
        name = meta.get("name", skill_md.parent.name)
        desc = meta.get("description", "(no description)")
        lines.append(f"- {name}: {desc}")
    if not lines:
        return f"No SKILL.md files found under {skills_dir}"
    return "\n".join(lines)


_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def tool_read_skill(ctx: ToolContext, input: dict[str, Any]) -> str:
    name = str(input.get("name", "")).strip()
    if not name or not _NAME_RE.match(name):
        return "ERROR: 'name' must be a bare skill folder name, e.g. 'linkedin-daily-run'."
    path = ctx.repo_root / "plugin" / "skills" / name / "SKILL.md"
    if not path.is_file():
        return f"ERROR: no SKILL.md at {path}. Call list_skills to see valid names."
    return _truncate(path.read_text(encoding="utf-8", errors="replace"), MAX_FILE_CHARS)


def tool_read_file(ctx: ToolContext, input: dict[str, Any]) -> str:
    rel = str(input.get("path", "")).strip()
    if not rel:
        return "ERROR: 'path' is required, relative to the project root."
    candidate = (ctx.repo_root / rel).resolve()
    try:
        candidate.relative_to(ctx.repo_root.resolve())
    except ValueError:
        return f"ERROR: '{rel}' resolves outside the project root; refusing to read it."
    if not candidate.is_file():
        return f"ERROR: no such file: {rel}"
    try:
        text = candidate.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"ERROR reading {rel}: {e}"
    return _truncate(text, MAX_FILE_CHARS)


def tool_run_shell(ctx: ToolContext, input: dict[str, Any]) -> str:
    command = str(input.get("command", "")).strip()
    if not command:
        return "ERROR: 'command' is required."
    if not is_shell_allowed(command):
        return (
            f"ERROR: command rejected by the allowlist: {command!r}. "
            "Only 'python -m jobagent ...', 'python -m orchestrator ...', and "
            "read-only 'git status/log/diff/branch' commands are permitted. "
            "Rewrite the command to use the jobagent CLI (see plugin/references/cli.md)."
        )
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.repo_root),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=ctx.config.shell_timeout,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {ctx.config.shell_timeout}s: {command}"
    parts = [f"$ {command}", f"exit_code: {proc.returncode}"]
    if proc.stdout:
        parts.append("stdout:\n" + _truncate(proc.stdout))
    if proc.stderr:
        parts.append("stderr:\n" + _truncate(proc.stderr))
    return "\n".join(parts)


_LABEL_RE = re.compile(r"[^A-Za-z0-9_-]")


def tool_screenshot(ctx: ToolContext, input: dict[str, Any]) -> Union[str, ToolResult]:
    """Capture a screenshot via the job agent's own CDP browser driver and
    show it to the model. Gated behind `config.vision` — see agent.py's
    `available_tools`, which only registers this tool at all when vision is
    on. The check here is defense-in-depth for callers that reach this
    function directly (tests, a future non-agent caller), not the primary
    gate.

    Deliberately shells out to `jobagent browse shot` rather than driving a
    browser itself — `jobagent/browser.py` already owns viewport-vs-full-page
    capture, the CDP connection, and where screenshots get saved; duplicating
    that here would be a second, divergent implementation of the same thing.
    """
    if not ctx.config.vision:
        return (
            "ERROR: screenshots are disabled. Set \"vision\": true in "
            "orchestrator/config.json (and use a vision-capable local model) "
            "to enable this tool."
        )
    raw_label = str(input.get("label", "")).strip() or "orchestrator"
    label = _LABEL_RE.sub("_", raw_label)[:60]
    full = bool(input.get("full", False))
    command = f"python -m jobagent browse shot --label {label}"
    if full:
        command += " --full"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.repo_root),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=ctx.config.shell_timeout,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: screenshot command timed out after {ctx.config.shell_timeout}s"

    payload: dict[str, Any] = {}
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if last_line:
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            payload = {}

    if not payload.get("ok"):
        detail = payload.get("error") or proc.stderr.strip() or proc.stdout.strip() or f"exit_code {proc.returncode}"
        return (
            f"ERROR: could not capture a screenshot: {detail}. "
            "Is a browser driver connected? Check `python -m jobagent browse status` first."
        )

    image_path = Path(payload["path"])
    if not image_path.is_file():
        return f"ERROR: jobagent reported a screenshot at {image_path} but that file doesn't exist."

    text = (
        f"Captured a screenshot of {payload.get('url', 'the current page')} "
        f"(viewport {payload.get('viewport')}, {'full page' if full else 'visible area only'})."
    )
    return ToolResult(text=text, image_path=image_path)


def tool_memory(ctx: ToolContext, input: dict[str, Any]) -> str:
    action = str(input.get("action", "")).strip().lower()
    if action == "set":
        key = str(input.get("key", "")).strip()
        if not key:
            return "ERROR: 'key' is required for action=set."
        db.memory_set(ctx.con, key, input.get("value"))
        return f"OK: remembered '{key}'."
    if action == "get":
        key = str(input.get("key", "")).strip()
        if not key:
            return "ERROR: 'key' is required for action=get."
        val = db.memory_get(ctx.con, key)
        if val is None:
            return f"(no memory stored for '{key}')"
        return str(val)
    if action == "list":
        mem = db.memory_list(ctx.con)
        if not mem:
            return "(memory is empty)"
        return "\n".join(f"{k}: {v}" for k, v in mem.items())
    return "ERROR: 'action' must be one of: get, set, list."


ToolFn = Callable[[ToolContext, dict[str, Any]], Union[str, ToolResult]]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "list_skills": tool_list_skills,
    "read_skill": tool_read_skill,
    "read_file": tool_read_file,
    "run_shell": tool_run_shell,
    "memory": tool_memory,
}

TOOL_HELP = """\
- list_skills: input {} — lists every LinkedIn-agent skill (name + one-line description).
- read_skill: input {"name": "<skill-folder-name>"} — returns the full SKILL.md for one skill.
- read_file: input {"path": "<repo-relative path>"} — reads a text file from the project (e.g. a reference doc).
- run_shell: input {"command": "<shell command>"} — runs a command in the project root. Only `python -m jobagent ...`, `python -m orchestrator ...`, and read-only `git status/log/diff/branch` are allowed; anything else is rejected.
- memory: input {"action": "get"|"set"|"list", "key": "...", "value": ...} — your own persistent memory across sessions, separate from the job agent's answer bank.\
"""

SCREENSHOT_HELP = """\
- screenshot: input {"label": "optional-short-name", "full": false} — captures a screenshot of the current page through the job agent's own browser driver (`jobagent browse shot`) and shows it to you directly, as an image. Needs a connected browser driver — check `run_shell` with `python -m jobagent browse status` first if it fails. `full: true` captures the whole page instead of just the viewport.\
"""


def available_tools(cfg: Config) -> dict[str, ToolFn]:
    """The tool registry for one session, gated by config.

    A plain `dict` module constant would freeze in `screenshot` (or leave it
    out) at import time. Building it fresh per-session means a config's
    `vision` flag is the only thing that decides whether the model ever
    hears about the tool — and, since this looks up `tool_screenshot` by
    name in this module's globals at call time rather than closing over it,
    tests can substitute a fake via `unittest.mock.patch` without needing a
    real browser.
    """
    registry = dict(TOOL_REGISTRY)
    if cfg.vision:
        registry["screenshot"] = tool_screenshot
    return registry


def tool_help_text(cfg: Config) -> str:
    return TOOL_HELP + ("\n" + SCREENSHOT_HELP if cfg.vision else "")
