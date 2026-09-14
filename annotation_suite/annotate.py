"""annotation_suite.annotate — a review UI for turning a harvested corpus into fine-tuning data.

Standalone on purpose: this folder is not part of the LinkedIn agent. It knows
nothing about `jobagent` and imports nothing from it — the only contract it
depends on is the shape of a `jobagent-sft/1` record (`schema`, `episode_id`,
`step_index`, `task`, `observation`, `thinking`, `rationale`, `action`,
`result`), documented in `plugin/references/harvest.md` in the sibling
project. Point it at any `*.steps.jsonl` with that shape.

Why this exists: `jobagent harvest` produces a faithful record of what
happened. It is not yet something you would train on — reasoning is missing
on most steps (`thinking.state != "present"`), some steps are scaffolding
that should never become a training pair, and a claim in a record ("well
under 2000 employees") is worthless supervision unless it resolves to
something the record actually shows. Closing that gap by hand-editing JSONL
is how you introduce the exact defect that matters most: a correction made
from memory, with no record of who made it or why.

This module is the annotation pass. It serves a browser UI over one
`*.steps.jsonl` file that can:

  * walk every record, grouped by task, with the screenshot alongside it;
  * edit any field a human might need to correct (observation text, task
    labels, the rationale, the action, the result);
  * mark a step kept or dropped, and reward it (-2..2) with a reason;
  * classify it against the five example types and the provenance classes a
    supervised corpus needs (grounding/judgement/decision/field_fill/stop,
    and SELF/CITED/IN_SHOT/EP_SHOT/NO_SHOT) — the two signals raw harvesting
    has no way to produce on its own;
  * backfill a stripped or absent `thinking` block by shelling out to the
    `claude` CLI itself (`claude -p ... --output-format json --restricted`),
    setting `generated: true` so synthesised reasoning is never confused
    with the real thing;
  * export a clean corpus — deletions dropped, edits applied, rewards
    attached — ready to fine-tune on, alongside a manifest.

Nothing here rewrites the harvested file. Every edit is appended as one JSON
line to a sidecar (`<input>.annotations.jsonl`), replayed newest-line-wins on
load, so the review pass gets a free, inspectable audit trail instead of a
silently mutated dataset.

Usage:
    python -m annotation_suite serve   <file>.steps.jsonl
    python -m annotation_suite export  <file>.steps.jsonl --format messages
    python -m annotation_suite stats   <file>.steps.jsonl
"""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

SIDECAR_SUFFIX = ".annotations.jsonl"

# The two closed vocabularies a raw harvest has no way to fill in on its
# own — only a human reading the record can say which of these it is.
EXAMPLE_TYPES = ["grounding", "judgement", "decision", "field_fill", "stop",
                  "scaffolding", "unclear"]
PROVENANCE = ["SELF", "CITED", "IN_SHOT", "EP_SHOT", "NO_SHOT", "unclassified"]
REWARD_LABELS = {-2: "unusable", -1: "bad", 0: "neutral", 1: "good", 2: "exemplary"}
TAG_VOCAB = ["hallucination", "unsourced_claim", "wrong_coords", "stale_screenshot",
             "pii_leak", "good_recovery", "needs_lookup", "off_vocabulary",
             "duplicate", "great_example"]

# Fields a reviewer is allowed to overwrite. Anything else stays exactly as
# harvested — this whitelist is what keeps an editor from quietly drifting
# the record shape the harvester guarantees downstream tooling.
ALLOWED_EDIT_PATHS = {
    "task.goal", "task.skill", "task.step", "task.task_id", "task.plan_step", "task.prompt",
    "observation.text", "observation.url", "observation.title",
    "rationale",
    "action.kind", "action.command", "action.args",
    "result.text", "result.ok", "result.error",
}

ANNOTATION_KEYS = {"status", "reward", "example_type", "provenance", "tags", "note",
                   "edits", "thinking_override", "reviewed", "screenshot_removed"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# loading the corpus + the sidecar log
# --------------------------------------------------------------------------

def load_steps(path: Path) -> list[dict]:
    recs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        raise ValueError(f"{path} has no records")
    if "messages" in recs[0]:
        raise ValueError(
            f"{path} is in `messages` format — annotate needs the `steps` format (the one "
            "with observation/action/thinking blocks); re-run the harvester without "
            "--format messages, or point at the matching .steps.jsonl")
    # A corpus routinely interleaves several episodes by timestamp (concurrent
    # sessions, or several files from one `harvest` run merged together), so
    # "the previous line" is not "the previous action in this trajectory" —
    # it can belong to a different episode entirely. Steps within one episode
    # keep their relative order through that merge (harvest.py sorts by
    # (ts, step_index) and a single episode's own timestamps are already
    # monotonic), so walking back by episode_id here is the correct notion of
    # "what did this agent just do", and it's what `generate-thinking` uses
    # to ground reasoning in the step immediately before this one.
    last_idx_by_episode: dict[str, int] = {}
    for i, r in enumerate(recs):
        r["_idx"] = i
        r["_prev_idx"] = last_idx_by_episode.get(r.get("episode_id"))
        last_idx_by_episode[r.get("episode_id")] = i
    return recs


def rec_key(rec: dict) -> str:
    return f"{rec.get('episode_id')}::{rec.get('step_index')}"


def sidecar_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return input_path.with_name(name + SIDECAR_SUFFIX)


def guess_traces_dir(corpus_path: Path) -> Path:
    """Screenshots referenced by a record live under the harvest project's
    `data/traces/`. This tool doesn't share a package with that project, so
    it locates them the same way `harvest.py` locates transcripts: by
    structure, not by import. A corpus written by `jobagent harvest` sits at
    `data/sft/<day>/<file>.steps.jsonl` (or `data/sft/<file>.steps.jsonl`) —
    walk up to the `sft` folder and take its sibling `traces`. Pass
    --traces-dir explicitly if the corpus was copied somewhere else."""
    p = corpus_path.resolve()
    for parent in p.parents:
        if parent.name == "sft":
            return parent.parent / "traces"
    return p.parent.parent / "traces"


def default_annotation() -> dict:
    return {"status": "kept", "reward": None, "example_type": None, "provenance": None,
            "tags": [], "note": "", "edits": {}, "thinking_override": None,
            "reviewed": False, "annotator": None, "updated_at": None,
            "screenshot_removed": False}


def load_annotations(sidecar: Path) -> dict[str, dict]:
    """Replay the append log: last line for a key wins, so history is free."""
    out: dict[str, dict] = {}
    if not sidecar.exists():
        return out
    with sidecar.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            k = ev.get("record_key")
            if k:
                out[k] = ev
    return out


_LOCK = threading.Lock()


def append_annotation(sidecar: Path, key: str, ann: dict) -> None:
    ev = dict(ann)
    ev["record_key"] = key
    ev["ts"] = utcnow()
    ev["updated_at"] = ev["ts"]
    with _LOCK:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, default=str, ensure_ascii=False) + "\n")


def apply_patch(current: dict, patch: dict) -> dict:
    out = dict(current)
    for k, v in patch.items():
        if k == "edits" and isinstance(v, dict):
            edits = dict(out.get("edits") or {})
            for ek, ev in v.items():
                if ev is None:
                    edits.pop(ek, None)
                else:
                    edits[ek] = ev
            out["edits"] = edits
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# merging an annotation onto the harvested record for display / export
# --------------------------------------------------------------------------

def set_path(obj: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    d = obj
    for p in parts[:-1]:
        nxt = d.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            d[p] = nxt
        d = nxt
    d[parts[-1]] = value


def get_path(obj: dict, dotted: str):
    d = obj
    for p in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


# Fields the "assist" endpoint (a per-field, user-instructed claude CLI call)
# is allowed to write. Same whitelist as edits, plus `thinking.text`, which
# isn't a plain edit — it goes through `thinking_override` instead.
ASSISTABLE_FIELDS = ALLOWED_EDIT_PATHS | {"thinking.text"}


def merge_record(rec: dict, ann: dict | None) -> dict:
    r = json.loads(json.dumps(rec, default=str))
    ann = ann or default_annotation()
    for path, value in (ann.get("edits") or {}).items():
        if path in ALLOWED_EDIT_PATHS:
            set_path(r, path, value)
    ov = ann.get("thinking_override")
    if ov and ov.get("text"):
        r["thinking"] = {"state": "present", "text": ov["text"], "placeholder": None,
                          "generated": bool(ov.get("generated", True)), "author": ov.get("author")}
    r["annotation"] = {k: ann.get(k) for k in ANNOTATION_KEYS if k != "edits"}
    r["annotation"]["edited_fields"] = sorted((ann.get("edits") or {}).keys())
    r["annotation"]["annotator"] = ann.get("annotator")
    r["annotation"]["updated_at"] = ann.get("updated_at")
    r["_deleted"] = ann.get("status") == "deleted"
    return r


_SEVERE = ("missing", "different frame", "nothing to act on", "no coordinates", "no frame")


def warnings_for(r: dict) -> list[str]:
    """The same checks `tools/view_harvest.py` runs in the sibling project,
    kept as a local copy so this folder has no import dependency on it."""
    w = []
    obs, act, res = r.get("observation") or {}, r.get("action") or {}, r.get("result") or {}
    if not (obs.get("text") or "").strip():
        w.append("no observation — the model had nothing to act on")
    if not (r.get("task") or {}).get("task_id"):
        w.append("no task_id — not joined to any episode")
    if res.get("ok") is False:
        w.append("the action failed")
    shot = obs.get("screenshot") or {}
    if shot:
        if shot.get("exists") is False:
            w.append("screenshot file is missing")
        if shot.get("frame_matches_viewport") is False:
            w.append("screenshot size disagrees with the viewport — coordinates are in a different frame")
        if (shot.get("age_steps") or 0) > 3:
            w.append("screenshot is stale (>3 steps old)")
    elif str(act.get("kind", "")).startswith("browse."):
        w.append("no screenshot for a browser action")
    if act.get("kind") == "browse.click":
        eff = act.get("effect") or {}
        if not eff.get("coords"):
            w.append("click has no coordinates — unusable for click training")
        elif not eff.get("viewport") or eff.get("scroll") is None:
            w.append("click coordinates have no frame (scroll/viewport)")
    return w


def is_severe(warns: list[str]) -> bool:
    return any(s in x for x in warns for s in _SEVERE)


def task_groups(records: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for r in records:
        key = (r.get("episode_id"), (r.get("task") or {}).get("task_id"))
        if not groups or groups[-1]["key"] != key:
            goal = (r.get("task") or {}).get("goal")
            groups.append({"key": key, "episode": key[0], "task_id": key[1],
                           "goal": goal, "indices": [],
                           "session_id": (r.get("source") or {}).get("session_id"),
                           "start": r.get("ts"), "end": r.get("ts")})
        g = groups[-1]
        g["indices"].append(r["_idx"])
        if not g["goal"]:
            g["goal"] = (r.get("task") or {}).get("goal")
        ts = r.get("ts")
        if ts:
            if not g["start"] or ts < g["start"]:
                g["start"] = ts
            if not g["end"] or ts > g["end"]:
                g["end"] = ts
    return groups


def session_summary(records: list[dict]) -> list[dict]:
    """One row per session, spanning the whole corpus (not just contiguous
    runs) — the thing a reviewer actually means by "a run": everything
    `jobagent harvest` pulled in from one Claude Code session transcript.
    `harvest` re-walks the *entire* project history on every invocation
    (see plugin/references/harvest.md), so one `.steps.jsonl` routinely
    mixes many past sessions together; this is what lets the UI default to
    only the newest one instead of showing everything at once.

    A session can bundle more than one *episode*: a sub-agent spawned via
    the Agent tool gets its own transcript file (its own `episode_id`), but
    Claude Code stamps its records with the same top-level `session_id` as
    the parent conversation that spawned it — from the runtime's point of
    view a sub-agent isn't a separate session, it's a nested call inside
    one. That is a common source of "why is data from two different runs
    showing up as one session" confusion, so each session's `episodes`
    breakdown is reported explicitly here instead of silently flattened."""
    by_sess: dict[str, dict] = {}
    for r in records:
        sid = (r.get("source") or {}).get("session_id")
        if not sid:
            continue
        g = by_sess.setdefault(sid, {"session_id": sid, "start": None, "end": None,
                                     "count": 0, "task_ids": set(), "skills": set(),
                                     "goal": None, "cwd": (r.get("source") or {}).get("cwd"),
                                     "episodes": {}})
        ts = r.get("ts")
        if ts:
            if not g["start"] or ts < g["start"]:
                g["start"] = ts
            if not g["end"] or ts > g["end"]:
                g["end"] = ts
        g["count"] += 1
        task = r.get("task") or {}
        if task.get("task_id"):
            g["task_ids"].add(task["task_id"])
        if task.get("skill"):
            g["skills"].add(task["skill"])
        if not g["goal"]:
            g["goal"] = task.get("goal") or task.get("prompt")
        eid = r.get("episode_id")
        ep = g["episodes"].setdefault(eid, {"episode_id": eid,
                                            "is_subagent": bool((r.get("source") or {}).get("is_subagent")),
                                            "count": 0, "start": ts, "end": ts})
        ep["count"] += 1
        if ts:
            if not ep["start"] or ts < ep["start"]:
                ep["start"] = ts
            if not ep["end"] or ts > ep["end"]:
                ep["end"] = ts
    out = []
    for g in by_sess.values():
        g["task_count"] = len(g.pop("task_ids"))
        g["skills"] = sorted(g["skills"])
        eps = sorted(g["episodes"].values(), key=lambda e: e["start"] or "")
        g["episodes"] = eps
        g["episode_count"] = len(eps)
        out.append(g)
    out.sort(key=lambda g: g["start"] or "", reverse=True)
    return out


# --------------------------------------------------------------------------
# generating missing thinking with the claude CLI
# --------------------------------------------------------------------------

THINKING_PROMPT = """You are backfilling a stripped <thinking> block for one step of a recorded \
agent trajectory that will be used to fine-tune a small model. Write the reasoning in exactly TWO \
BEATS, in this order, first person, 4-7 sentences total, grounded strictly in the context given:

  BEAT 1 — what just happened. Name what the last action actually was and what the observation \
that followed it actually shows (a specific element, value, error, or state — not a vague "the page \
loaded"). This beat must come first and must not be skipped, even briefly; a reasoning that opens \
by talking about the next action without first grounding in what the last action produced is wrong.
  BEAT 2 — what that implies. Given what beat 1 just established, explain why the next action shown \
below is the correct thing to do now: what it follows up on, checks, recovers from, or advances \
toward the goal/sub-goal. Stop at the decision to act — do not comment on what happens after.

Do not just restate the action as the reasoning, and do not open with the next action itself — it \
must arrive as the conclusion of beat 2, earned by beat 1. Base the reasoning ONLY on the goal, the \
last action, and the observation — never on the "result of the next action" given below for \
reference. The agent could not see that result yet when it reasoned; if it leaks in, the reasoning \
reads as fabricated hindsight instead of a real, forward-looking decision. Do not mention that this \
is reconstructed or refer to yourself as an AI. If the context is too thin to reason about \
specifically, say plainly what is uncertain rather than inventing detail. Output nothing but the \
reasoning itself: no beat labels, no preamble, no headings, no markdown fences, no quotes around it.

Skill: {skill}
Goal: {goal}
Current sub-goal: {plan_step}

Last action (what the agent had just done before this observation appeared):
{prev_action}

Observation (what the agent saw right after that last action — beat 1 must be grounded in this):
{observation}

Next action to justify (this must arrive as beat 2's conclusion, not beat 1's opening line):
{action}

Result of the next action — reference only, not knowable to the agent while it was reasoning, do \
not let it leak into the reasoning:
{result}

Reasoning, beat 1 (what just happened) then beat 2 (what that implies for the next action):"""


def _action_repr(act: dict) -> str:
    return act.get("command") or json.dumps(
        {k: act.get(k) for k in ("kind", "args") if act.get(k)}, default=str) or "(none)"


def _prev_action_repr(prev_rec: dict | None) -> str:
    if not prev_rec:
        return "(none — this is the first recorded action in this episode)"
    act = prev_rec.get("action") or {}
    res = prev_rec.get("result") or {}
    outcome = "succeeded" if res.get("ok") is not False else "FAILED"
    snippet = _action_repr(act)
    return f"{snippet}\n(outcome: {outcome})"


def build_thinking_prompt(rec: dict, prev_rec: dict | None = None) -> str:
    task = rec.get("task") or {}
    obs = rec.get("observation") or {}
    act = rec.get("action") or {}
    res = rec.get("result") or {}
    return THINKING_PROMPT.format(
        skill=task.get("skill") or "(unknown)",
        goal=task.get("goal") or task.get("prompt") or "(unknown)",
        plan_step=task.get("plan_step") or "(none)",
        prev_action=_prev_action_repr(prev_rec),
        observation=(obs.get("text") or "(empty)")[:4000],
        action=_action_repr(act),
        result=(res.get("text") or "(none)")[:2000],
    )


ASSIST_PROMPT = """You are helping a human annotator write or revise one field of a recorded agent \
trajectory step that will be used to fine-tune a small model. Follow the annotator's instruction \
exactly. Output ONLY the replacement text for the field itself — no preamble, no headings, no \
markdown fences, no quotes around it, and no explanation of what you changed or why.

Field being written: {field}

Annotator's instruction:
{instruction}

Context for this step:
Skill: {skill}
Goal: {goal}
Current sub-goal: {plan_step}

Last action (what the agent had just done before this observation appeared):
{prev_action}

Observation (what the agent saw right after that last action):
{observation}

Action taken at this step:
{action}

Result of that action:
{result}

Current value of the field (revise it if the instruction asks for a revision; ignore it and write \
fresh text if it is empty or the instruction asks to replace it wholesale):
{current_value}

Output nothing but the new text for `{field}`:"""


def build_assist_prompt(rec: dict, field: str, instruction: str, prev_rec: dict | None,
                        current_value) -> str:
    task = rec.get("task") or {}
    obs = rec.get("observation") or {}
    act = rec.get("action") or {}
    res = rec.get("result") or {}
    cv = "" if current_value is None else str(current_value)
    return ASSIST_PROMPT.format(
        field=field, instruction=instruction,
        skill=task.get("skill") or "(unknown)",
        goal=task.get("goal") or task.get("prompt") or "(unknown)",
        plan_step=task.get("plan_step") or "(none)",
        prev_action=_prev_action_repr(prev_rec),
        observation=(obs.get("text") or "(empty)")[:4000],
        action=_action_repr(act),
        result=(res.get("text") or "(none)")[:2000],
        current_value=(cv[:4000] if cv.strip() else "(empty)"),
    )


def call_claude(prompt: str, model: str = "sonnet", timeout: int = 120) -> tuple[str | None, str | None]:
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model,
           "--restricted", "--strict-mcp-config", "--permission-prompts", "none"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None, "claude CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return None, f"claude CLI timed out after {timeout}s"
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:2000]
    try:
        data = json.loads(proc.stdout)
    except Exception:
        return None, "could not parse claude CLI output as JSON"
    if data.get("is_error"):
        return None, str(data.get("result"))[:2000]
    text = (data.get("result") or "").strip()
    if not text:
        return None, "empty response"
    return text, None


# --------------------------------------------------------------------------
# the `messages` (chat-shaped) view — a local copy of the sibling project's
# harvest.to_messages, so this folder has no import dependency on it either.
# --------------------------------------------------------------------------

THINKING_ABSENT = "<THINKING_ABSENT>"

SYSTEM_PROMPT = (
    "You are a LinkedIn job-application agent driving a real browser through "
    "the `jobagent browse` CLI. You are given the result of your last command "
    "and must decide the next one. Think first, then issue exactly one command."
)


def to_messages(rec: dict) -> dict:
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
# export — the point of the whole exercise
# --------------------------------------------------------------------------

def export_corpus(records: list[dict], annotations: dict[str, dict], out_path: Path, *,
                  fmt: str = "steps", min_reward: int | None = None,
                  keep_deleted: bool = False) -> dict:
    kept, dropped_deleted, dropped_reward = [], 0, 0
    counts_kind: dict[str, int] = {}
    reward_hist: dict[str, int] = {}
    example_type_hist: dict[str, int] = {}
    provenance_hist: dict[str, int] = {}
    thinking_hist: dict[str, int] = {}
    edited = 0

    for rec in records:
        key = rec_key(rec)
        ann = annotations.get(key)
        merged = merge_record(rec, ann)
        if merged["_deleted"] and not keep_deleted:
            dropped_deleted += 1
            continue
        reward = merged["annotation"].get("reward")
        if min_reward is not None and reward is not None and reward < min_reward:
            dropped_reward += 1
            continue
        merged.pop("_idx", None)
        merged.pop("_deleted", None)
        if merged["annotation"].get("screenshot_removed"):
            (merged.get("observation") or {})["screenshot"] = None
        if merged["annotation"]["edited_fields"]:
            edited += 1
        counts_kind[merged["action"]["kind"]] = counts_kind.get(merged["action"]["kind"], 0) + 1
        reward_hist[str(reward)] = reward_hist.get(str(reward), 0) + 1
        et = merged["annotation"].get("example_type")
        if et:
            example_type_hist[et] = example_type_hist.get(et, 0) + 1
        pv = merged["annotation"].get("provenance")
        if pv:
            provenance_hist[pv] = provenance_hist.get(pv, 0) + 1
        thinking_hist[merged["thinking"]["state"]] = thinking_hist.get(merged["thinking"]["state"], 0) + 1
        kept.append(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in kept:
            if fmt == "messages":
                row = to_messages(r)
                row["meta"]["reward"] = r["annotation"].get("reward")
                row["meta"]["example_type"] = r["annotation"].get("example_type")
                row["meta"]["provenance"] = r["annotation"].get("provenance")
                row["meta"]["thinking_generated"] = r["thinking"].get("generated")
            else:
                row = r
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": utcnow(),
        "source_records": len(records),
        "exported_records": len(kept),
        "dropped_deleted": dropped_deleted,
        "dropped_below_min_reward": dropped_reward,
        "edited_records": edited,
        "format": fmt,
        "min_reward": min_reward,
        "keep_deleted": keep_deleted,
        "by_kind": dict(sorted(counts_kind.items(), key=lambda kv: -kv[1])),
        "reward_histogram": reward_hist,
        "example_type_histogram": example_type_hist,
        "provenance_histogram": provenance_hist,
        "thinking_histogram": thinking_hist,
        "out": str(out_path),
    }
    out_path.with_suffix(out_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def default_export_path(input_path: Path, fmt: str) -> Path:
    name = input_path.name
    for suf in (".steps.jsonl", ".messages.jsonl", ".jsonl"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return input_path.with_name(f"{name}.annotated.{fmt}.jsonl")


# --------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------

_STATE: dict = {}
_JOBS: dict[str, dict] = {}
UI_HTML_PATH = Path(__file__).parent / "annotate_ui.html"


def _prev_record(rec: dict) -> dict | None:
    prev_idx = rec.get("_prev_idx")
    return _STATE["records"][prev_idx] if prev_idx is not None else None


def _screenshot_file(rec: dict) -> Path | None:
    shot = (rec.get("observation") or {}).get("screenshot") or {}
    raw = shot.get("abs_path") or shot.get("dataset_path") or shot.get("path")
    if not raw:
        return None
    traces_dir: Path = _STATE["traces_dir"]
    corpus_dir: Path = _STATE["path"].parent
    name = Path(str(raw).replace("\\", "/")).name
    for cand in (Path(raw), traces_dir / raw, traces_dir / "screenshots" / name,
                traces_dir / name, corpus_dir / "images" / name, corpus_dir / name):
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _corpus_summary() -> dict:
    records = _STATE["records"]
    annotations = load_annotations(_STATE["sidecar"])
    kinds, tasks_by_key, sessions, skills = set(), {}, set(), set()
    thinking_counts: dict[str, int] = {}
    reward_counts: dict[str, int] = {}
    status_counts = {"kept": 0, "deleted": 0}
    reviewed = 0
    for r in records:
        kinds.add(r["action"]["kind"])
        if r["task"].get("skill"):
            skills.add(r["task"]["skill"])
        sid = (r.get("source") or {}).get("session_id")
        if sid:
            sessions.add(sid)
        thinking_counts[r["thinking"]["state"]] = thinking_counts.get(r["thinking"]["state"], 0) + 1
        ann = annotations.get(rec_key(r), {})
        if ann.get("status") == "deleted":
            status_counts["deleted"] += 1
        else:
            status_counts["kept"] += 1
        if ann.get("reviewed"):
            reviewed += 1
        rw = ann.get("reward")
        if rw is not None:
            reward_counts[str(rw)] = reward_counts.get(str(rw), 0) + 1
    for g in task_groups(records):
        tasks_by_key[str(g["key"])] = {"task_id": g["task_id"], "episode": g["episode"],
                                       "goal": g["goal"], "count": len(g["indices"]),
                                       "session_id": g["session_id"],
                                       "start": g["start"], "end": g["end"]}
    sessions_detail = session_summary(records)
    return {
        "path": str(_STATE["path"]), "sidecar": str(_STATE["sidecar"]),
        "count": len(records), "kinds": sorted(kinds), "skills": sorted(skills),
        "sessions": sorted(sessions), "sessions_detail": sessions_detail,
        "latest_session": sessions_detail[0]["session_id"] if sessions_detail else None,
        "tasks": list(tasks_by_key.values()),
        "thinking_counts": thinking_counts, "reward_counts": reward_counts,
        "status_counts": status_counts, "reviewed": reviewed,
        "vocab": {"example_types": EXAMPLE_TYPES, "provenance": PROVENANCE,
                  "reward_labels": REWARD_LABELS, "tags": TAG_VOCAB,
                  "edit_paths": sorted(ALLOWED_EDIT_PATHS)},
    }


def _record_view(idx: int) -> dict:
    rec = _STATE["records"][idx]
    ann = load_annotations(_STATE["sidecar"]).get(rec_key(rec))
    merged = merge_record(rec, ann)
    merged["_warnings"] = warnings_for(rec)
    merged["_severe"] = is_severe(merged["_warnings"])
    merged["_idx"] = idx
    merged["_key"] = rec_key(rec)
    # `_screenshot_exists` is the raw file check; `_has_screenshot` also
    # honors this step's own `screenshot_removed` flag. Removing a shot never
    # touches the file — the same PNG routinely grounds several steps (a
    # step with no fresh shot of its own cites the last one taken, via
    # `age_steps`), so deleting it would blind every step that shares it.
    # Removal is per-step and reversible, like every other annotation.
    merged["_screenshot_exists"] = _screenshot_file(rec) is not None
    merged["_has_screenshot"] = merged["_screenshot_exists"] and not (ann or {}).get("screenshot_removed")
    return merged


def _filtered_indices(qs: dict, records: list[dict] | None = None,
                      annotations: dict[str, dict] | None = None) -> list[int]:
    records = _STATE["records"] if records is None else records
    annotations = load_annotations(_STATE["sidecar"]) if annotations is None else annotations
    kind = qs.get("kind", [None])[0]
    thinking = qs.get("thinking", [None])[0]
    task_id = qs.get("task", [None])[0]
    session = qs.get("session", [None])[0]
    status = qs.get("status", [None])[0]
    example_type = qs.get("example_type", [None])[0]
    provenance = qs.get("provenance", [None])[0]
    min_reward = qs.get("min_reward", [None])[0]
    warn_only = qs.get("warn", ["0"])[0] == "1"
    unreviewed_only = qs.get("unreviewed", ["0"])[0] == "1"
    start = qs.get("start", [None])[0] or None
    end = qs.get("end", [None])[0] or None
    q = (qs.get("q", [""])[0] or "").lower()

    out = []
    for r in records:
        # `ts` is ISO 8601 (utcnow()-shaped), so lexical comparison is a
        # correct time-window filter without parsing it back into a datetime.
        if start and (not r.get("ts") or r["ts"] < start):
            continue
        if end and (not r.get("ts") or r["ts"] > end):
            continue
        if kind and r["action"]["kind"] != kind and not r["action"]["kind"].startswith(kind + "."):
            continue
        if thinking and r["thinking"]["state"] != thinking:
            continue
        if task_id and (r.get("task") or {}).get("task_id") != task_id:
            continue
        if session and (r.get("source") or {}).get("session_id") != session:
            continue
        ann = annotations.get(rec_key(r), {})
        st = ann.get("status", "kept")
        if status and st != status:
            continue
        if example_type and ann.get("example_type") != example_type:
            continue
        if provenance and ann.get("provenance") != provenance:
            continue
        if min_reward is not None and min_reward != "":
            rw = ann.get("reward")
            if rw is None or rw < int(min_reward):
                continue
        if unreviewed_only and ann.get("reviewed"):
            continue
        if warn_only and not warnings_for(r):
            continue
        if q:
            blob = " ".join(str(x) for x in (
                (r.get("observation") or {}).get("text"), r.get("rationale"),
                (r.get("action") or {}).get("command"), (r.get("task") or {}).get("goal"),
            ) if x).lower()
            if q not in blob:
                continue
        out.append(r["_idx"])
    return out


def _summary_row(idx: int, annotations: dict) -> dict:
    r = _STATE["records"][idx]
    ann = annotations.get(rec_key(r), {})
    warns = warnings_for(r)
    obs_text = (r.get("observation") or {}).get("text") or ""
    return {
        "idx": idx, "key": rec_key(r), "kind": r["action"]["kind"],
        "skill": r["task"].get("skill"), "step": r["task"].get("step"),
        "task_id": r["task"].get("task_id"), "session": (r.get("source") or {}).get("session_id"),
        "ts": r.get("ts"), "ok": r["result"].get("ok"), "thinking_state": r["thinking"]["state"],
        "status": ann.get("status", "kept"), "reward": ann.get("reward"),
        "example_type": ann.get("example_type"), "reviewed": bool(ann.get("reviewed")),
        "warnings": len(warns), "severe": is_severe(warns),
        "has_screenshot": _screenshot_file(r) is not None and not ann.get("screenshot_removed"),
        "preview": (obs_text[:140] + ("…" if len(obs_text) > 140 else "")),
    }


def make_handler():
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "annotation-suite/1"

        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            raw = self.rfile.read(n)
            return json.loads(raw or b"{}")

        def do_GET(self):
            parts = urllib.parse.urlsplit(self.path)
            qs = urllib.parse.parse_qs(parts.query)
            path = parts.path

            if path == "/" or path == "/index.html":
                html = UI_HTML_PATH.read_text(encoding="utf-8")
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/corpus":
                return self._json(_corpus_summary())

            if path == "/api/records":
                indices = _filtered_indices(qs)
                offset = int(qs.get("offset", ["0"])[0])
                limit = int(qs.get("limit", ["200"])[0])
                annotations = load_annotations(_STATE["sidecar"])
                page = indices[offset: offset + limit]
                rows = [_summary_row(i, annotations) for i in page]
                return self._json({"total": len(indices), "offset": offset, "records": rows})

            m = re.fullmatch(r"/api/records/([^/]+)", path)
            if m:
                key = urllib.parse.unquote(m.group(1))
                idx = _STATE["by_key"].get(key)
                if idx is None:
                    return self._json({"ok": False, "error": "unknown record key"}, 404)
                return self._json(_record_view(idx))

            m = re.fullmatch(r"/api/screenshot/([^/]+)", path)
            if m:
                key = urllib.parse.unquote(m.group(1))
                idx = _STATE["by_key"].get(key)
                rec = _STATE["records"][idx] if idx is not None else None
                ann = load_annotations(_STATE["sidecar"]).get(key) if rec is not None else None
                f = _screenshot_file(rec) if rec is not None and not (ann or {}).get("screenshot_removed") else None
                if not f:
                    return self._json({"ok": False, "error": "no screenshot"}, 404)
                data = f.read_bytes()
                ctype = mimetypes.guess_type(f.name)[0] or "image/png"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
                return

            m = re.fullmatch(r"/api/jobs/([^/]+)", path)
            if m:
                job = _JOBS.get(m.group(1))
                if not job:
                    return self._json({"ok": False, "error": "unknown job"}, 404)
                return self._json(job)

            self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):
            parts = urllib.parse.urlsplit(self.path)
            path = parts.path
            try:
                body = self._body()
            except Exception:
                return self._json({"ok": False, "error": "bad JSON body"}, 400)

            m = re.fullmatch(r"/api/records/([^/]+)", path)
            if m:
                return self._handle_patch(urllib.parse.unquote(m.group(1)), body)

            m = re.fullmatch(r"/api/records/([^/]+)/generate-thinking", path)
            if m:
                return self._handle_generate(urllib.parse.unquote(m.group(1)), body)

            m = re.fullmatch(r"/api/records/([^/]+)/assist", path)
            if m:
                return self._handle_assist(urllib.parse.unquote(m.group(1)), body)

            if path == "/api/generate-thinking/bulk":
                return self._handle_bulk(body)

            if path == "/api/export":
                return self._handle_export(body)

            self._json({"ok": False, "error": "not found"}, 404)

        def _handle_patch(self, key, patch):
            idx = _STATE["by_key"].get(key)
            if idx is None:
                return self._json({"ok": False, "error": "unknown record key"}, 404)
            bad = set(patch) - ANNOTATION_KEYS
            if bad:
                return self._json({"ok": False, "error": f"unknown annotation field(s): {sorted(bad)}"}, 400)
            edits = patch.get("edits") or {}
            bad_paths = set(edits) - ALLOWED_EDIT_PATHS
            if bad_paths:
                return self._json({"ok": False, "error": f"field(s) not editable: {sorted(bad_paths)}"}, 400)
            if "reward" in patch and patch["reward"] is not None and patch["reward"] not in REWARD_LABELS:
                return self._json({"ok": False, "error": "reward must be -2..2 or null"}, 400)
            if "status" in patch and patch["status"] not in ("kept", "deleted"):
                return self._json({"ok": False, "error": "status must be kept|deleted"}, 400)

            current = load_annotations(_STATE["sidecar"]).get(key) or default_annotation()
            merged = apply_patch(current, patch)
            merged["annotator"] = _STATE.get("annotator")
            append_annotation(_STATE["sidecar"], key, merged)
            self._json({"ok": True, "record": _record_view(idx)})

        def _handle_generate(self, key, body):
            idx = _STATE["by_key"].get(key)
            if idx is None:
                return self._json({"ok": False, "error": "unknown record key"}, 404)
            rec = _STATE["records"][idx]
            force = bool(body.get("force"))
            if rec["thinking"]["state"] == "present" and not force:
                return self._json({"ok": False, "error": "already has real reasoning — pass force to overwrite"}, 409)
            model = body.get("model") or _STATE["model"]
            text, err = call_claude(build_thinking_prompt(rec, _prev_record(rec)), model=model)
            if err:
                return self._json({"ok": False, "error": err})
            current = load_annotations(_STATE["sidecar"]).get(key) or default_annotation()
            merged = apply_patch(current, {"thinking_override": {
                "text": text, "generated": True, "author": f"claude-cli:{model}"}})
            merged["annotator"] = _STATE.get("annotator")
            append_annotation(_STATE["sidecar"], key, merged)
            self._json({"ok": True, "text": text, "record": _record_view(idx)})

        def _handle_assist(self, key, body):
            idx = _STATE["by_key"].get(key)
            if idx is None:
                return self._json({"ok": False, "error": "unknown record key"}, 404)
            field = body.get("field")
            if field not in ASSISTABLE_FIELDS:
                return self._json(
                    {"ok": False, "error": f"field must be one of {sorted(ASSISTABLE_FIELDS)}"}, 400)
            instruction = (body.get("instruction") or "").strip()
            if not instruction:
                return self._json({"ok": False, "error": "instruction is required"}, 400)
            rec = _STATE["records"][idx]
            current_ann = load_annotations(_STATE["sidecar"]).get(key)
            merged_rec = merge_record(rec, current_ann)
            current_value = (merged_rec["thinking"].get("text") if field == "thinking.text"
                             else get_path(merged_rec, field))
            model = body.get("model") or _STATE["model"]
            prompt = build_assist_prompt(rec, field, instruction, _prev_record(rec), current_value)
            text, err = call_claude(prompt, model=model)
            if err:
                return self._json({"ok": False, "error": err})
            current = current_ann or default_annotation()
            if field == "thinking.text":
                patch = {"thinking_override": {"text": text, "generated": True,
                                               "author": f"claude-cli:{model} (assist)"}}
            else:
                patch = {"edits": {field: text}}
            merged_ann = apply_patch(current, patch)
            merged_ann["annotator"] = _STATE.get("annotator")
            append_annotation(_STATE["sidecar"], key, merged_ann)
            self._json({"ok": True, "text": text, "record": _record_view(idx)})

        def _handle_bulk(self, body):
            qs = {k: [str(v)] for k, v in (body.get("filter") or {}).items()}
            qs.setdefault("thinking", [None])
            indices = [i for i in _filtered_indices(qs)
                      if _STATE["records"][i]["thinking"]["state"] != "present" or body.get("force")]
            limit = body.get("limit")
            if limit:
                indices = indices[: int(limit)]
            model = body.get("model") or _STATE["model"]
            job_id = uuid.uuid4().hex[:12]
            job = {"id": job_id, "total": len(indices), "done": 0, "ok": 0, "failed": 0,
                   "current_key": None, "finished": False, "errors": []}
            _JOBS[job_id] = job

            def run():
                for i in indices:
                    rec = _STATE["records"][i]
                    key = rec_key(rec)
                    job["current_key"] = key
                    text, err = call_claude(build_thinking_prompt(rec, _prev_record(rec)), model=model)
                    if err:
                        job["failed"] += 1
                        job["errors"].append({"key": key, "error": err})
                    else:
                        current = load_annotations(_STATE["sidecar"]).get(key) or default_annotation()
                        merged = apply_patch(current, {"thinking_override": {
                            "text": text, "generated": True, "author": f"claude-cli:{model}"}})
                        merged["annotator"] = _STATE.get("annotator")
                        append_annotation(_STATE["sidecar"], key, merged)
                        job["ok"] += 1
                    job["done"] += 1
                    time.sleep(0.15)
                job["finished"] = True
                job["current_key"] = None

            threading.Thread(target=run, daemon=True).start()
            self._json({"ok": True, "job_id": job_id, "total": len(indices)})

        def _handle_export(self, body):
            fmt = body.get("format", "steps")
            out = Path(body["out"]).expanduser() if body.get("out") else default_export_path(_STATE["path"], fmt)
            records = _STATE["records"]
            # `filter` is the same shape the UI's toolbar sends to /api/records —
            # when present (even with every field empty), scope the export to
            # exactly what a matching grid/review view would show, time window
            # included, instead of always dumping the whole corpus.
            filt = body.get("filter")
            if filt:
                qs = {k: [str(v)] for k, v in filt.items() if v not in (None, "")}
                indices = _filtered_indices(qs)
                records = [_STATE["records"][i] for i in indices]
            manifest = export_corpus(
                records, load_annotations(_STATE["sidecar"]), out, fmt=fmt,
                min_reward=body.get("min_reward"), keep_deleted=bool(body.get("keep_deleted")))
            manifest["filtered"] = bool(filt)
            if filt:
                manifest["filter"] = filt
            self._json({"ok": True, "manifest": manifest})

    return Handler


def _load_state(path: str, traces_dir: str | None, model: str, annotator: str | None) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    records = load_steps(p)
    _STATE.clear()
    _STATE.update({
        "records": records, "by_key": {rec_key(r): r["_idx"] for r in records},
        "path": p, "sidecar": sidecar_path(p),
        "traces_dir": Path(traces_dir).expanduser().resolve() if traces_dir else guess_traces_dir(p),
        "model": model, "annotator": annotator,
    })
    return p


def serve(path: str, host: str = "127.0.0.1", port: int = 8765, no_browser: bool = False,
         model: str = "sonnet", traces_dir: str | None = None, annotator: str | None = None) -> None:
    p = _load_state(path, traces_dir, model, annotator)
    httpd = socketserver.ThreadingTCPServer((host, port), make_handler())
    httpd.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"annotation_suite: {len(_STATE['records'])} records from {p.name}")
    print(f"  annotations -> {_STATE['sidecar'].name}")
    print(f"  screenshots resolved under -> {_STATE['traces_dir']}")
    print(f"  serving on {url}  (Ctrl+C to stop)")
    if not no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        print("stopped.")


def export_cli(path: str, out: str | None, fmt: str = "steps", min_reward: int | None = None,
               keep_deleted: bool = False, start: str | None = None, end: str | None = None,
               kind: str | None = None, thinking: str | None = None, task_id: str | None = None,
               session_id: str | None = None, status: str | None = None,
               example_type: str | None = None, provenance: str | None = None,
               warn_only: bool = False, unreviewed_only: bool = False,
               query: str | None = None) -> dict:
    p = Path(path).expanduser().resolve()
    records = load_steps(p)
    annotations = load_annotations(sidecar_path(p))
    # Same filter shape the UI sends to /api/export — a time window (--start/
    # --end) plus every other toolbar filter, so a scripted export can match
    # exactly what a filtered grid view would show without opening the UI.
    filt = {"start": start, "end": end, "kind": kind, "thinking": thinking, "task": task_id,
            "session": session_id, "status": status, "example_type": example_type,
            "provenance": provenance, "warn": "1" if warn_only else None,
            "unreviewed": "1" if unreviewed_only else None, "q": query}
    if any(v not in (None, "") for v in filt.values()):
        qs = {k: [str(v)] for k, v in filt.items() if v not in (None, "")}
        indices = _filtered_indices(qs, records=records, annotations=annotations)
        records = [records[i] for i in indices]
    out_path = Path(out).expanduser() if out else default_export_path(p, fmt)
    return export_corpus(records, annotations, out_path, fmt=fmt, min_reward=min_reward,
                         keep_deleted=keep_deleted)


def stats(path: str, traces_dir: str | None = None) -> dict:
    _load_state(path, traces_dir, "sonnet", None)
    return _corpus_summary()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annotation_suite",
        description="Review, reward and export a jobagent-sft/1 harvested corpus.")
    sub = p.add_subparsers(dest="action", required=True)

    g = sub.add_parser("serve", help="open the review UI in a browser")
    g.add_argument("path", help="a *.steps.jsonl (not --format messages)")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--no-browser", dest="no_browser", action="store_true",
                   help="don't auto-open a browser tab")
    g.add_argument("--model", default="sonnet",
                   help="claude CLI model alias used to backfill thinking (default: sonnet)")
    g.add_argument("--traces-dir", dest="traces_dir",
                   help="where referenced screenshots live (default: guessed from the corpus "
                        "path's data/sft/<day>/ structure)")
    g.add_argument("--annotator", help="name/id recorded on every annotation this session writes")

    g = sub.add_parser("export", help="write the annotated corpus without opening the UI")
    g.add_argument("path")
    g.add_argument("--out", help="output path (default: <input>.annotated.<format>.jsonl)")
    g.add_argument("--format", default="steps", choices=["steps", "messages"])
    g.add_argument("--min-reward", dest="min_reward", type=int,
                   help="drop records with a reward set and below this")
    g.add_argument("--keep-deleted", dest="keep_deleted", action="store_true",
                   help="include steps marked deleted (default: dropped)")
    g.add_argument("--start", help="only records with ts >= this ISO timestamp "
                                    "(lexical match, e.g. 2026-09-10T00:00:00+00:00)")
    g.add_argument("--end", help="only records with ts <= this ISO timestamp")
    g.add_argument("--kind", help="only this action.kind, or its dotted prefix (browse matches browse.click)")
    g.add_argument("--thinking", choices=["present", "stripped", "absent"], help="only this thinking.state")
    g.add_argument("--task", dest="task_id", help="only this task.task_id")
    g.add_argument("--session", dest="session_id", help="only this source.session_id")
    g.add_argument("--status", choices=["kept", "deleted"], help="only this annotation status")
    g.add_argument("--example-type", dest="example_type", choices=EXAMPLE_TYPES)
    g.add_argument("--provenance", choices=PROVENANCE)
    g.add_argument("--warn-only", dest="warn_only", action="store_true",
                   help="only records with a warning (see warnings_for)")
    g.add_argument("--unreviewed-only", dest="unreviewed_only", action="store_true")
    g.add_argument("--query", dest="query", help="substring match, same as the UI search box")

    g = sub.add_parser("stats", help="print corpus + annotation counts as JSON")
    g.add_argument("path")
    g.add_argument("--traces-dir", dest="traces_dir")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "serve":
            serve(args.path, host=args.host, port=args.port, no_browser=args.no_browser,
                  model=args.model, traces_dir=args.traces_dir, annotator=args.annotator)
            return 0
        if args.action == "export":
            m = export_cli(args.path, args.out, fmt=args.format, min_reward=args.min_reward,
                           keep_deleted=args.keep_deleted, start=args.start, end=args.end,
                           kind=args.kind, thinking=args.thinking, task_id=args.task_id,
                           session_id=args.session_id, status=args.status,
                           example_type=args.example_type, provenance=args.provenance,
                           warn_only=args.warn_only, unreviewed_only=args.unreviewed_only,
                           query=args.query)
            print(json.dumps(m, default=str))
            return 0
        if args.action == "stats":
            print(json.dumps(stats(args.path, traces_dir=args.traces_dir), default=str))
            return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
