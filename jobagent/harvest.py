"""jobagent.harvest — turn Claude Code session transcripts into training data.

Why this exists
---------------
`trace log` is a promise the agent has to remember to keep, and it doesn't.
Measured on two real runs: 123 browser actions produced 11 trace records.
The other 112 happened and left nothing behind.

But Claude Code already writes every one of them down. Each session is a
JSONL file under `~/.claude/projects/<slug>/`, with the sub-agent that did
the actual work in `<session-id>/subagents/*.jsonl`. Those files hold the
verbatim tool call, the verbatim result, the assistant's own words, and a
timestamp — the whole (observation → action → result) chain, recorded by the
runtime rather than by the agent describing itself afterwards.

This module reads those files and emits one fixed-shape record per step.

What it can and cannot recover
------------------------------
Thinking blocks are present in the transcript but their text is stripped on
write — `{"type":"thinking","thinking":"","signature":"…"}`. The signature is
opaque. So the model's actual reasoning is **not** recoverable from disk, and
no amount of parsing changes that.

Every record therefore carries an explicit `thinking` block saying which case
it is:

  * `stripped`  — a thinking block existed and was emptied. The model did
                  reason here; the words are gone.
  * `absent`    — no thinking block at all.
  * `present`   — text survived (rare; kept verbatim).

`thinking.text` is null in the first two cases and `thinking.placeholder`
holds a token for a later pass to fill. That pass sets `thinking.text` and
flips `thinking.generated` to true — so synthesised reasoning is never
confusable with the real thing.

On top of the transcript, `data/traces/*.jsonl` contributes the labels the
transcript has no way to know: which skill, which step in that skill's
vocabulary, which job the episode belongs to. Records are joined by time.

Sources, in one picture
-----------------------
    ~/.claude/projects/<slug>/<session>.jsonl      the parent session
    ~/.claude/projects/<slug>/<session>/
        subagents/agent-*.jsonl                    where the work happened
        tool-results/*.txt                         results too big to inline
    data/traces/*.jsonl                            skill/step/task_id labels
    data/traces/screenshots/*.png                  what the page looked like
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import struct
from datetime import datetime, timezone
from pathlib import Path

from . import db as D

SCHEMA = "jobagent-sft/1"

# A tool result can be 110k characters. Records stay readable and trainable at
# a fraction of that; the full text is kept on disk and pointed at, never
# thrown away.
DEFAULT_MAX_CHARS = 8000

THINKING_STRIPPED = "<THINKING_STRIPPED>"
THINKING_ABSENT = "<THINKING_ABSENT>"


# --------------------------------------------------------------------------
# discovery — find the transcripts for a project without guessing its slug
# --------------------------------------------------------------------------

def claude_home(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".claude"


def discover_sessions(claude_dir: Path, project_root: Path | None,
                      session: str | None = None) -> list[dict]:
    """Find every session transcript belonging to `project_root`.

    Claude Code names project folders by slugifying the working directory,
    and the exact rule has changed between versions. Rather than reproduce
    it and be wrong on some machine, we read each transcript's own `cwd`
    field — the file says which directory it belongs to, so ask it.
    """
    roots = [claude_dir / "projects"]
    if not roots[0].exists():
        # Also accept being pointed straight at a slug folder or a copy of one.
        roots = [claude_dir]

    found: list[dict] = []
    for base in roots:
        if not base.exists():
            continue
        for jf in sorted(base.glob("*/*.jsonl")) + sorted(base.glob("*.jsonl")):
            if jf.parent.name == "subagents":
                continue
            cwd = _peek_cwd(jf)
            if project_root and cwd and Path(cwd) != project_root:
                continue
            if session and session not in jf.stem:
                continue
            sdir = jf.with_suffix("")
            found.append({
                "parent": jf,
                "session_id": jf.stem,
                "cwd": cwd,
                "subagents": sorted((sdir / "subagents").glob("*.jsonl"))
                             if (sdir / "subagents").exists() else [],
                "tool_results": (sdir / "tool-results")
                                if (sdir / "tool-results").exists() else None,
            })
    # Same file can be reached by two globs; keep one of each.
    seen, out = set(), []
    for f in found:
        if f["parent"] in seen:
            continue
        seen.add(f["parent"])
        out.append(f)
    return out


def _peek_cwd(path: Path, lines: int = 40) -> str | None:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= lines:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("cwd"):
                    return r["cwd"]
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------
# reading one transcript
# --------------------------------------------------------------------------

def _read(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


_BROWSE = re.compile(r"jobagent\s+browse\s+([a-z_]+)")
_CLI = re.compile(r"jobagent\s+([a-z_]+)(?:\s+([a-z_]+))?")


def classify(tool: str, tool_input: dict) -> tuple[str, dict]:
    """Give every action a stable `kind` and, for our own CLI, parsed args.

    A flat `kind` is what makes the corpus filterable later — "give me every
    browse.click" has to be one field lookup, not a regex over shell strings.
    """
    if tool != "Bash":
        return f"tool.{tool}", dict(tool_input or {})

    cmd = (tool_input or {}).get("command", "") or ""
    m = _BROWSE.search(cmd)
    if m:
        return f"browse.{m.group(1)}", _browse_args(cmd, m.group(1))
    m = _CLI.search(cmd)
    if m and "jobagent" in cmd:
        grp = m.group(1)
        sub = m.group(2) or ""
        return f"cli.{grp}" + (f".{sub}" if sub else ""), {}
    return "bash", {}


def _browse_args(cmd: str, verb: str) -> dict:
    """Pull --flags off the browse call so they are queryable as fields."""
    args: dict = {"verb": verb}
    try:
        tokens = shlex.split(cmd, posix=False)
    except ValueError:
        tokens = cmd.split()
    i = 0
    while i < len(tokens):
        t = tokens[i].strip("'\"")
        if t.startswith("--") or t in ("-q",):
            key = t.lstrip("-") or t
            nxt = tokens[i + 1].strip("'\"") if i + 1 < len(tokens) else ""
            if nxt and not nxt.startswith("-"):
                args[key.replace("-", "_")] = nxt
                i += 2
                continue
            args[key.replace("-", "_")] = True
        i += 1
    return args


def _text_of(content) -> str:
    return _content_of(content)[0]


def _content_of(content) -> tuple[str, int]:
    """Text of a tool result, plus how many image blocks came with it.

    The image count matters: the agent reads its own screenshots back with
    `Read`, and those results are image blocks with no text at all. Counting
    only text made 21 of those look like empty observations, when in fact
    they are the most informative records in the corpus — the ones where the
    model looked at a picture before deciding.
    """
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        parts, images = [], 0
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text") or "")
                elif b.get("type") == "image":
                    images += 1
        return "\n".join(parts), images
    return (json.dumps(content, default=str) if content is not None else ""), 0


_SPILL = re.compile(r"saved to ([^\s\"]+\.txt)")


def _resolve_spill(text: str, tool_results: Path | None) -> tuple[str, bool]:
    """A result too large to inline names a file. Read it back in."""
    if not tool_results or "exceeds maximum" not in text:
        return text, False
    m = _SPILL.search(text)
    if not m:
        return text, False
    cand = tool_results / Path(m.group(1)).name
    if cand.exists():
        try:
            return cand.read_text(encoding="utf-8", errors="replace"), True
        except Exception:
            return text, False
    return text, False


def _png_dims(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as fh:
            head = fh.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        w, h = struct.unpack(">II", head[16:24])
        return int(w), int(h)
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# --------------------------------------------------------------------------
# redaction
#
# AGENTS.md forbids logging a salary figure, phone number or other PII into
# the corpus. The transcript contains all of it verbatim, so redaction is on
# by default here — the training value is in the *shape* of the answer, not
# the digits.
# --------------------------------------------------------------------------

_PHONE = re.compile(r"\b[6-9]\d{9}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_MONEY = re.compile(r"\b\d{1,3}(?:[.,]\d{1,2})?\s*(?:LPA|lpa|lakhs?|Lakhs?)\b")
_LONGNUM = re.compile(r"\b\d{6,}\b")


def redact(text: str) -> str:
    if not text:
        return text
    text = _PHONE.sub("<PHONE>", text)
    text = _EMAIL.sub("<EMAIL>", text)
    text = _MONEY.sub("<SALARY>", text)
    # Job ids are 10 digits and genuinely useful; keep those, mask the rest.
    text = re.sub(r"\b\d{11,}\b", "<NUM>", text)
    return text


# --------------------------------------------------------------------------
# the trace side — labels the transcript cannot know
# --------------------------------------------------------------------------

def _ts(v) -> datetime | None:
    if not v:
        return None
    try:
        s = str(v).replace("Z", "+00:00")
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_traces(traces_dir: Path) -> dict:
    """Return {episodes: [...], steps: [...]} from data/traces/**/*.jsonl —
    both the legacy flat `<day>.jsonl` files and the current `<day>/
    <session>.jsonl` layout (one folder per day, one file per run)."""
    steps = []
    files = sorted(traces_dir.glob("*.jsonl"))
    for day_dir in sorted(p for p in traces_dir.glob("*")
                          if p.is_dir() and p.name not in ("screenshots", "pages")):
        files.extend(sorted(day_dir.glob("*.jsonl")))
    for jf in files:
        for r in _read(jf):
            t = _ts(r.get("ts"))
            if t:
                r["_t"] = t
                steps.append(r)
    steps.sort(key=lambda r: r["_t"])

    # `_task_start` .. `_task_end` bracket an episode. Anything the agent did
    # between them belongs to that job, whether or not it was logged.
    episodes, open_ep = [], {}
    for r in steps:
        tid = r.get("task_id")
        if not tid:
            continue
        if r.get("step") == "_task_start":
            open_ep[tid] = {"task_id": tid, "skill": r.get("skill"),
                            "instruction": r.get("instruction"),
                            "goal": r.get("goal"), "run_id": r.get("run_id"),
                            "start": r["_t"], "end": None}
        elif r.get("step") == "_task_end" and tid in open_ep:
            ep = open_ep.pop(tid)
            ep["end"] = r["_t"]
            ep["outcome"] = r.get("outcome")
            episodes.append(ep)
    for ep in open_ep.values():
        episodes.append(ep)
    return {"episodes": episodes, "steps": steps}


def _episode_at(episodes, t, slack=90):
    if not t:
        return None
    best = None
    for ep in episodes:
        start, end = ep["start"], ep.get("end")
        if t < start:
            continue
        if end and (t - end).total_seconds() > slack:
            continue
        if best is None or ep["start"] > best["start"]:
            best = ep
    return best


def _trace_at(steps, t, window=15):
    """A hand-written record within a few seconds is describing this action."""
    if not t:
        return None
    best, gap = None, window + 1
    for r in steps:
        if r.get("step", "").startswith("_task_"):
            continue
        d = abs((r["_t"] - t).total_seconds())
        if d < gap:
            best, gap = r, d
    return best if gap <= window else None


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

def harvest_transcript(path: Path, *, tool_results: Path | None = None,
                       is_subagent: bool = False, max_chars: int = DEFAULT_MAX_CHARS,
                       do_redact: bool = True) -> list[dict]:
    """One record per tool call, carrying what the model saw and what it did."""
    recs = _read(path)

    # tool_use_id -> the result that came back
    results: dict[str, dict] = {}
    for r in recs:
        m = r.get("message")
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_result":
                txt, images = _content_of(b.get("content"))
                txt, spilled = _resolve_spill(txt, tool_results)
                results[b.get("tool_use_id")] = {
                    "text": txt,
                    "images": images,
                    "is_error": bool(b.get("is_error")),
                    "spilled": spilled,
                    "meta": r.get("toolUseResult"),
                }

    steps: list[dict] = []
    pending_text: str | None = None
    pending_thinking: str | None = None
    saw_thinking = False
    last_result: dict | None = None      # the observation the model acted on
    last_result_kind: str | None = None
    last_shot: dict | None = None
    shot_age = 0
    task_prompt = None
    index = 0

    for r in recs:
        rtype = r.get("type")
        m = r.get("message")

        if rtype == "user":
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str) and content.strip() and task_prompt is None:
                task_prompt = content.strip()
            # A turn boundary: whatever the model said belongs to the call it
            # made, not to the next one.
            pending_text, pending_thinking, saw_thinking = None, None, False
            continue

        if rtype != "assistant" or not isinstance(m, dict):
            continue

        for b in (m.get("content") or []):
            kind = b.get("type")

            if kind == "text":
                t = (b.get("text") or "").strip()
                if t:
                    pending_text = t
                continue

            if kind == "thinking":
                saw_thinking = True
                t = b.get("thinking") or ""
                if t.strip():
                    pending_thinking = t
                continue

            if kind != "tool_use":
                continue

            tool = b.get("name") or "?"
            tin = b.get("input") or {}
            act_kind, args = classify(tool, tin)
            res = results.get(b.get("id"), {})
            rtext = res.get("text", "") or ""

            # A single Bash block routinely chains several `jobagent` calls
            # (log the trace, fill a field, take a shot), each printing its
            # own JSON line via cli.py's out() — so rtext is N JSON objects,
            # newline-separated, not one. json.loads(rtext) as a whole always
            # threw on those and silently dropped every command but a lone
            # one, which is how a `browse shot` chained after a `browse fill`
            # went unrecorded and the *next* step kept citing the screenshot
            # from steps earlier. Parse line by line instead.
            #
            # And don't gate the shot capture on this block's classified
            # act_kind: classify() tags the whole block with the *first*
            # `jobagent browse <verb>` it matches in the command text, so a
            # `browse shot` running after a `browse fill` in the same block
            # left act_kind == "browse.fill" and the shot's own result was
            # ignored even once parsed. A shot's result shape — `path` plus
            # `viewport` and `dpr` together — is unique among this CLI's
            # commands (dump() has `path` with no `dpr`; fill()/click() have
            # neither), so key on the shape instead of the block's act_kind.
            ok, err = (not res.get("is_error"), None)
            for line in rtext.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                if parsed.get("ok") is False:
                    ok = False
                    err = err or parsed.get("error")
                if parsed.get("path") and "viewport" in parsed and "dpr" in parsed:
                    last_shot = {"path": parsed["path"],
                                 "viewport": parsed.get("viewport"),
                                 "scroll": parsed.get("scroll"),
                                 "dpr": parsed.get("dpr")}
                    shot_age = 0

            prev = last_result or {}
            obs_text = prev.get("text", "") or ""
            page = _page_state(obs_text)
            # The screenshot fallback grounds "what did the page look like
            # when this observation was formed" — only meaningful when the
            # observation actually came from a browser action's result.
            # Without this, `last_shot` kept getting attached to every kind
            # of step unconditionally (a `cli.job.applied` bookkeeping call,
            # a bare `bash`, a `Grep`), each one just inheriting whatever
            # screenshot happened to exist, growing staler with no relevance
            # to what the step is actually about.
            page_observed = (last_result_kind or "").startswith("result_of:browse.")

            # An observation that was a picture. The path comes from the
            # `Read` that produced it, so the record points at the actual
            # file rather than carrying a megabyte of base64.
            obs_images = []
            if prev.get("images"):
                src = (prev.get("input") or {}).get("file_path")
                if src:
                    obs_images.append(src)
                if not obs_text.strip():
                    obs_text = (f"[image: {Path(str(src).replace(chr(92), '/')).name}]"
                                if src else f"[{prev['images']} image(s)]")

            rec = {
                "schema": SCHEMA,
                "episode_id": path.stem,
                "step_index": index,
                "ts": r.get("timestamp"),
                "source": {
                    "transcript": str(path),
                    "session_id": r.get("sessionId"),
                    "is_subagent": is_subagent,
                    "cwd": r.get("cwd"),
                    "model": m.get("model"),
                    "claude_version": r.get("version"),
                    "uuid": r.get("uuid"),
                },
                "task": {
                    "prompt": _clip(task_prompt, 4000, do_redact),
                    # filled by the trace join
                    "skill": None, "step": None, "task_id": None,
                    "run_id": None, "plan_step": None, "goal": None,
                },
                "observation": {
                    "kind": last_result_kind or "start_of_episode",
                    "text": _clip(obs_text, max_chars, do_redact),
                    "chars": len(obs_text),
                    "truncated": len(obs_text) > max_chars,
                    **page,
                    "images": obs_images or None,
                    # A screenshot the model actually looked at beats the
                    # last one it happened to take: age 0, and no guessing.
                    "screenshot": (
                        {"path": obs_images[0], "age_steps": 0, "from": "read"}
                        if obs_images
                        else (dict(last_shot, age_steps=shot_age)
                              if last_shot and page_observed else None)
                    ),
                },
                "thinking": _thinking(pending_thinking, saw_thinking),
                "rationale": _clip(pending_text, 4000, do_redact),
                "action": {
                    "tool": tool,
                    "kind": act_kind,
                    "command": _clip(tin.get("command"), 2000, do_redact)
                               if tool == "Bash" else None,
                    "args": redact_obj(args, do_redact),
                    "input": _small(tin, do_redact),
                    # Where the click actually landed, lifted out of the
                    # result into fields. Buried in a result string these are
                    # unusable for click-prediction; as fields they are the
                    # label. `scroll` and `viewport` travel with them because
                    # a coordinate without its frame names a place on a
                    # screen, not on a page.
                    **redact_obj(_effect(rtext), do_redact),
                },
                "result": {
                    "ok": ok,
                    "error": err,
                    "text": _clip(rtext, max_chars, do_redact),
                    "chars": len(rtext),
                    "truncated": len(rtext) > max_chars,
                    "from_spill_file": bool(res.get("spilled")),
                },
            }
            steps.append(rec)
            index += 1

            last_result = {"text": rtext, "images": res.get("images", 0),
                           "input": tin, "kind": act_kind}
            last_result_kind = f"result_of:{act_kind}"
            pending_text, pending_thinking, saw_thinking = None, None, False
            if act_kind != "browse.shot":
                shot_age += 1

    return steps


def _thinking(text: str | None, saw_block: bool) -> dict:
    """Always say which of the three cases this is — never leave it implied.

    A later LLM pass fills `text` and sets `generated: true`. Keeping
    `state` alongside means synthesised reasoning can always be told apart
    from the real thing, including after the two are mixed in one file.
    """
    if text:
        return {"state": "present", "text": text, "placeholder": None,
                "generated": False}
    if saw_block:
        return {"state": "stripped", "text": None,
                "placeholder": THINKING_STRIPPED, "generated": False}
    return {"state": "absent", "text": None,
            "placeholder": THINKING_ABSENT, "generated": False}


_URL = re.compile(r'"url"\s*:\s*"([^"]{0,400})"')
_TITLE = re.compile(r'"title"\s*:\s*"([^"]{0,200})"')


def _page_state(text: str) -> dict:
    """Lift url/scroll/viewport out of a browse result so they are fields.

    These are what place a click on a page rather than on a screen — the
    thing the 4 Sep corpus audit found missing everywhere.
    """
    out: dict = {"url": None, "title": None, "viewport": None, "scroll": None}
    if not text or not text.lstrip().startswith("{"):
        return out
    try:
        d = json.loads(text)
    except Exception:
        m = _URL.search(text)
        if m:
            out["url"] = m.group(1)
        return out
    if not isinstance(d, dict):
        return out
    for k in ("url", "title", "viewport", "scroll"):
        if d.get(k) is not None:
            out[k] = d[k]
    return out


_EFFECT_KEYS = ("coords", "bbox", "scroll", "viewport", "navigated", "name",
                "value", "matches", "checked", "clicked", "ref")


def _effect(result_text: str) -> dict:
    """The measurable consequences of the action, as fields."""
    if not result_text or not result_text.lstrip().startswith("{"):
        return {}
    try:
        d = json.loads(result_text)
    except Exception:
        return {}
    if not isinstance(d, dict) or d.get("ok") is False:
        return {}
    out = {k: d[k] for k in _EFFECT_KEYS if d.get(k) is not None}
    return {"effect": out} if out else {}


def redact_obj(obj, do_redact: bool = True):
    """Redaction has to reach parsed fields too, not just free text.

    The first version only cleaned the text blobs, and a phone number typed
    into a form still came through intact in `action.args.value` and
    `action.effect.value` — the two places a training run is most likely to
    read. Structured fields are exactly where PII hides in plain sight.
    """
    if not do_redact:
        return obj
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v, True) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, True) for v in obj]
    return obj


def _clip(text, limit: int, do_redact: bool):
    if text is None:
        return None
    if do_redact:
        text = redact(text)
    return text if len(text) <= limit else text[:limit] + f"\n…[+{len(text)-limit} chars]"


def _small(obj, do_redact: bool, limit: int = 2000):
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        return {}
    if do_redact:
        s = redact(s)
    if len(s) > limit:
        return {"_truncated": True, "_preview": s[:limit]}
    try:
        return json.loads(s)
    except Exception:
        return {"_raw": s[:limit]}


# --------------------------------------------------------------------------
# joining, screenshots, output
# --------------------------------------------------------------------------

def attach_labels(steps: list[dict], traces: dict) -> dict:
    stats = {"joined_episode": 0, "joined_step": 0, "unlabelled": 0}
    for s in steps:
        t = _ts(s.get("ts"))
        ep = _episode_at(traces["episodes"], t)
        if ep:
            s["task"].update({"skill": ep.get("skill"), "task_id": ep["task_id"],
                              "run_id": ep.get("run_id"), "goal": ep.get("goal")})
            if not s["task"].get("prompt"):
                s["task"]["prompt"] = ep.get("instruction")
            stats["joined_episode"] += 1
        tr = _trace_at(traces["steps"], t)
        if tr:
            s["task"].update({"step": tr.get("step"),
                              "plan_step": tr.get("plan_step")})
            s["labels"] = {"trace_outcome": tr.get("outcome"),
                           "trace_driver": tr.get("driver"),
                           "trace_confidence": tr.get("confidence")}
            # A hand-written record may carry a screenshot ours didn't see.
            o = tr.get("observation") or {}
            if o.get("screenshot") and not s["observation"].get("screenshot"):
                s["observation"]["screenshot"] = {"path": o["screenshot"],
                                                  "age_steps": 0,
                                                  "from": "trace"}
            stats["joined_step"] += 1
        if not ep and not tr:
            stats["unlabelled"] += 1
    return stats


def resolve_screenshots(steps: list[dict], traces_dir: Path,
                        copy_to: Path | None = None) -> dict:
    """Turn a screenshot path into something a training run can actually open."""
    stats = {"with_screenshot": 0, "resolved": 0, "missing": 0, "copied": 0}
    cache: dict[str, dict] = {}
    for s in steps:
        shot = s["observation"].get("screenshot")
        if not shot or not shot.get("path"):
            continue
        stats["with_screenshot"] += 1
        raw = shot["path"]
        if raw in cache:
            shot.update(cache[raw])
            # Count per record, not per distinct file — otherwise a screenshot
            # reused across ten steps looks like one, and the totals stop
            # adding up to the record count.
            stats["resolved" if cache[raw].get("exists") else "missing"] += 1
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = traces_dir / raw
        if not p.exists():
            # Paths in a transcript are absolute and machine-specific — a
            # corpus harvested on one box, or from a copied trace folder,
            # would otherwise lose every image. The basename is unique
            # (date + hash), so falling back to it is safe.
            name = Path(raw.replace("\\", "/")).name
            for cand in (traces_dir / "screenshots" / name, traces_dir / name):
                if cand.exists():
                    p = cand
                    break
        info: dict = {"exists": p.exists()}
        if p.exists():
            dims = _png_dims(p)
            info.update({"abs_path": str(p), "bytes": p.stat().st_size,
                         "sha256": _sha256(p)})
            if dims:
                info["width"], info["height"] = dims
                # The frame check the corpus audit wanted: a coordinate is
                # only meaningful if its image is the size it claims.
                vp = s["observation"].get("viewport")
                if isinstance(vp, list) and len(vp) == 2:
                    info["frame_matches_viewport"] = (dims[0] == vp[0] and dims[1] == vp[1])
            if copy_to:
                copy_to.mkdir(parents=True, exist_ok=True)
                dest = copy_to / p.name
                if not dest.exists():
                    dest.write_bytes(p.read_bytes())
                    stats["copied"] += 1
                info["dataset_path"] = f"images/{p.name}"
            stats["resolved"] += 1
        else:
            stats["missing"] += 1
        cache[raw] = info
        shot.update(info)
    return stats


SYSTEM_PROMPT = (
    "You are a LinkedIn job-application agent driving a real browser through "
    "the `jobagent browse` CLI. You are given the result of your last command "
    "and must decide the next one. Think first, then issue exactly one command."
)


def _yield(steps: list[dict], traces: dict) -> dict:
    """How many actions happened vs how many were logged, same window."""
    ts = [_ts(s.get("ts")) for s in steps]
    ts = [t for t in ts if t]
    if not ts:
        return {"harvested": len(steps), "hand_logged_in_window": 0}
    lo, hi = min(ts), max(ts)
    logged = [r for r in traces["steps"] if lo <= r["_t"] <= hi
              and not str(r.get("step", "")).startswith("_task_")]
    browser = [s for s in steps if s["action"]["kind"].startswith("browse.")]
    return {
        "window": [lo.isoformat(), hi.isoformat()],
        "harvested": len(steps),
        "harvested_browser_actions": len(browser),
        "hand_logged_in_window": len(logged),
        "ratio": round(len(steps) / len(logged), 1) if logged else None,
    }


def to_messages(rec: dict) -> dict:
    """Chat-shaped view of one step, for supervised fine-tuning.

    The thinking placeholder is emitted verbatim rather than dropped: a
    training run can filter on it, and a generation pass can fill it in
    place without having to re-derive which records need it.
    """
    task = rec.get("task") or {}
    obs = rec.get("observation") or {}
    act = rec.get("action") or {}
    th = rec.get("thinking") or {}

    header = []
    if task.get("skill"):
        header.append(f"Skill: {task['skill']}")
    if task.get("goal"):
        header.append(f"Goal: {task['goal']}")
    if task.get("plan_step"):
        header.append(f"Current sub-goal: {task['plan_step']}")
    if obs.get("url"):
        header.append(f"URL: {obs['url']}")
    if obs.get("scroll"):
        header.append(f"Scroll: {obs['scroll']}  Viewport: {obs.get('viewport')}")
    shot = obs.get("screenshot") or {}
    if shot.get("dataset_path") or shot.get("abs_path"):
        header.append(f"Screenshot: {shot.get('dataset_path') or shot['abs_path']}")

    user = "\n".join(header)
    user += f"\n\nLast result ({obs.get('kind')}):\n{obs.get('text') or '(none)'}"

    think = th.get("text") or th.get("placeholder") or THINKING_ABSENT
    assistant = f"<thinking>\n{think}\n</thinking>\n"
    if rec.get("rationale"):
        assistant += f"\n{rec['rationale']}\n"
    assistant += f"\n<action kind=\"{act.get('kind')}\">\n{act.get('command') or json.dumps(act.get('input'))}\n</action>"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "episode_id": rec.get("episode_id"),
            "step_index": rec.get("step_index"),
            "kind": act.get("kind"),
            "skill": task.get("skill"),
            "step": task.get("step"),
            "task_id": task.get("task_id"),
            "thinking_state": th.get("state"),
            "ok": (rec.get("result") or {}).get("ok"),
            "has_screenshot": bool(shot.get("exists")),
            "session_id": (rec.get("source") or {}).get("session_id"),
        },
    }


# --------------------------------------------------------------------------
# the entry point the CLI calls
# --------------------------------------------------------------------------

def run(*, claude_dir=None, project_root=None, transcript=None, session=None,
        db_path=None, out_dir=None, fmt="steps", kinds=None, max_chars=DEFAULT_MAX_CHARS,
        do_redact=True, copy_images=False, include_parent=True) -> dict:
    project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    traces_dir = D.traces_dir(db_path)
    out_dir = Path(out_dir) if out_dir else (traces_dir.parent / "sft")
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions: list[dict] = []
    if transcript:
        p = Path(transcript).expanduser()
        if p.is_dir():
            sessions = discover_sessions(p, None, session)
        else:
            sdir = p.with_suffix("")
            sessions = [{"parent": p, "session_id": p.stem, "cwd": _peek_cwd(p),
                         "subagents": sorted((sdir / "subagents").glob("*.jsonl"))
                                      if (sdir / "subagents").exists() else [],
                         "tool_results": (sdir / "tool-results")
                                         if (sdir / "tool-results").exists() else None}]
    else:
        sessions = discover_sessions(claude_home(claude_dir), project_root, session)

    if not sessions:
        raise FileNotFoundError(
            f"no transcripts for {project_root} under {claude_home(claude_dir)}. "
            f"Pass --transcript <file|dir>, or --claude-dir if your Claude Code "
            f"config lives elsewhere (WSL keeps it in the Linux home, not C:\\Users)."
        )

    traces = load_traces(traces_dir) if traces_dir.exists() else {"episodes": [], "steps": []}

    steps: list[dict] = []
    per_file = []
    for s in sessions:
        files = list(s["subagents"])
        if include_parent:
            files.append(s["parent"])
        for f in files:
            got = harvest_transcript(
                f, tool_results=s.get("tool_results"),
                is_subagent=(f.parent.name == "subagents"),
                max_chars=max_chars, do_redact=do_redact)
            for g in got:
                g["source"]["session_id"] = g["source"].get("session_id") or s["session_id"]
            per_file.append({"file": str(f), "steps": len(got),
                             "subagent": f.parent.name == "subagents"})
            steps.extend(got)

    steps.sort(key=lambda r: (r.get("ts") or "", r.get("step_index", 0)))
    for i, s in enumerate(steps):
        s["step_index"] = i

    join_stats = attach_labels(steps, traces)

    # One folder per day, one set of harvest files per invocation inside
    # it — `stamp` (not just `day`) so running harvest twice in the same
    # day produces two sessions instead of the second silently overwriting
    # the first's .steps.jsonl/.manifest.json.
    day = D.today()
    stamp = D.now_time()
    day_dir = out_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    shot_stats = resolve_screenshots(
        steps, traces_dir, day_dir / "images" if copy_images else None)

    if kinds:
        want = set(kinds)
        steps = [s for s in steps
                 if s["action"]["kind"] in want
                 or s["action"]["kind"].split(".")[0] in want]

    out_path = day_dir / f"{stamp}-harvest.{'messages' if fmt == 'messages' else 'steps'}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for s in steps:
            row = to_messages(s) if fmt == "messages" else s
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    counts: dict = {}
    thinking: dict = {}
    for s in steps:
        counts[s["action"]["kind"]] = counts.get(s["action"]["kind"], 0) + 1
        st = s["thinking"]["state"]
        thinking[st] = thinking.get(st, 0) + 1

    manifest = {
        "schema": SCHEMA,
        "generated_at": D.utcnow(),
        "project_root": str(project_root),
        "out": str(out_path),
        "format": fmt,
        "redacted": do_redact,
        "sessions": [{"session_id": s["session_id"], "parent": str(s["parent"]),
                      "subagents": [str(x) for x in s["subagents"]]} for s in sessions],
        "files": per_file,
        "records": len(steps),
        "by_kind": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "thinking": thinking,
        "join": join_stats,
        "screenshots": shot_stats,
        "trace_records_available": len(traces["steps"]),
        "trace_episodes": len(traces["episodes"]),
        # The number this whole module exists to move. Compared inside the
        # harvested window only — counting every trace record ever written
        # against one session's actions would flatter nobody.
        "yield_vs_trace_log": _yield(steps, traces),
    }
    (day_dir / f"{stamp}-harvest.manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
