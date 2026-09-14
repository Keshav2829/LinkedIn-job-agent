#!/usr/bin/env python3
"""Render a harvested corpus as a browsable HTML page, and flag what looks wrong.

A JSONL file with 265 records is not something you can eyeball, and the parts
most likely to be broken — a screenshot that does not exist, a click whose
coordinates belong to a different frame, an observation that came through
empty — are exactly the parts that look fine in a text editor.

    python tools/view_harvest.py data/sft/2026-09-05-harvest.messages.jsonl
    python tools/view_harvest.py data/sft/2026-09-05-harvest.steps.jsonl --embed
    python tools/view_harvest.py data/sft/*.messages.jsonl --only-warnings

Writes `<input>.html` next to the input and prints a summary. Both formats
are accepted; the shape is detected from the first record.

No dependencies. The markdown renderer below is deliberately small — this is
a validation tool, not a publishing pipeline.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# a small markdown renderer
#
# Enough for what actually appears in a rationale: paragraphs, fenced and
# inline code, bold/italic, links, bullet and numbered lists, headings.
# Everything is escaped before any of it runs, so a page of LinkedIn HTML in
# an observation cannot inject anything.
# --------------------------------------------------------------------------

def md(text: str) -> str:
    if not text:
        return ""
    text = html.escape(text, quote=False)

    fences: list[str] = []

    def stash(m):
        fences.append(m.group(1))
        return f"\x00FENCE{len(fences)-1}\x00"

    text = re.sub(r"```[a-zA-Z0-9_-]*\n(.*?)```", stash, text, flags=re.S)

    out, buf, in_ul, in_ol = [], [], False, False

    def flush():
        nonlocal buf
        if buf:
            out.append("<p>" + "<br>".join(buf) + "</p>")
            buf = []

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    for line in text.split("\n"):
        s = line.rstrip()
        if not s.strip():
            flush()
            close_lists()
            continue
        h = re.match(r"^(#{1,6})\s+(.*)$", s)
        if h:
            flush()
            close_lists()
            n = len(h.group(1))
            out.append(f"<h{n+2}>{_inline(h.group(2))}</h{n+2}>")
            continue
        ul = re.match(r"^\s*[-*+]\s+(.*)$", s)
        ol = re.match(r"^\s*(\d+)[.)]\s+(.*)$", s)
        if ul:
            flush()
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline(ul.group(1))}</li>")
            continue
        if ol:
            flush()
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{_inline(ol.group(2))}</li>")
            continue
        close_lists()
        buf.append(_inline(s))

    flush()
    close_lists()
    rendered = "\n".join(out)
    for i, f in enumerate(fences):
        rendered = rendered.replace(f"\x00FENCE{i}\x00",
                                    f"<pre class='code'>{f}</pre>")
    return rendered


def _inline(s: str) -> str:
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
               r"<a href='\2' target='_blank' rel='noreferrer'>\1</a>", s)
    return s


# --------------------------------------------------------------------------
# normalising the two formats into one view model
# --------------------------------------------------------------------------

_THINK = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.S)
_ACTION = re.compile(r'<action kind="([^"]*)">\s*(.*?)\s*</action>', re.S)


def from_messages(row: dict) -> dict:
    meta = row.get("meta") or {}
    msgs = {m["role"]: m["content"] for m in row.get("messages", [])}
    a = msgs.get("assistant", "")
    think = (_THINK.search(a).group(1) if _THINK.search(a) else "")
    am = _ACTION.search(a)
    rationale = a
    for pat in (_THINK, _ACTION):
        rationale = pat.sub("", rationale)
    user = msgs.get("user", "")
    head, _, obs = user.partition("\nLast result")
    return {
        "kind": meta.get("kind"),
        "skill": meta.get("skill"),
        "step": meta.get("step"),
        "task_id": meta.get("task_id"),
        "episode": meta.get("episode_id"),
        "session": meta.get("session_id"),
        "index": meta.get("step_index"),
        "ok": meta.get("ok"),
        "thinking_state": meta.get("thinking_state"),
        "thinking": think,
        "rationale": rationale.strip(),
        "header": head.strip(),
        "observation": ("Last result" + obs).strip() if obs else "",
        "action": am.group(2) if am else "",
        "screenshot": _shot_from_header(head),
        # Not "no effect" — "this format does not carry one". `warnings_for`
        # relies on the distinction.
        "effect": None,
        "result": None,
        "ts": None,
        # True only for thinking backfilled by a generation pass, not
        # captured live -- see meta.thinking_generated / harvest.md.
        "generated": bool(meta.get("thinking_generated")),
        # This format IS a chat turn (system/user/assistant) -- the raw
        # roles are worth labelling in the viewer. Steps format isn't.
        "is_chat": True,
    }


def _shot_from_header(head: str) -> dict | None:
    m = re.search(r"^Screenshot:\s*(.+)$", head or "", re.M)
    return {"path": m.group(1).strip(), "exists": None} if m else None


def from_steps(rec: dict) -> dict:
    t = rec.get("thinking") or {}
    o = rec.get("observation") or {}
    a = rec.get("action") or {}
    r = rec.get("result") or {}
    task = rec.get("task") or {}
    head = []
    if task.get("skill"):
        head.append(f"Skill: {task['skill']}")
    if task.get("goal"):
        head.append(f"Goal: {task['goal']}")
    if task.get("plan_step"):
        head.append(f"Sub-goal: {task['plan_step']}")
    if o.get("url"):
        head.append(f"URL: {o['url']}")
    if o.get("scroll") is not None:
        head.append(f"Scroll: {o['scroll']}  Viewport: {o.get('viewport')}")
    shot = o.get("screenshot")
    return {
        "kind": a.get("kind"),
        "skill": task.get("skill"),
        "step": task.get("step"),
        "task_id": task.get("task_id"),
        "episode": rec.get("episode_id"),
        "session": (rec.get("source") or {}).get("session_id"),
        "index": rec.get("step_index"),
        "ok": r.get("ok"),
        "thinking_state": t.get("state"),
        "thinking": t.get("text") or t.get("placeholder") or "",
        "rationale": rec.get("rationale") or "",
        "header": "\n".join(head),
        "observation": o.get("text") or "",
        "action": a.get("command") or json.dumps(a.get("input"), indent=1),
        "screenshot": dict(shot) if shot else None,
        "effect": a.get("effect"),
        "result": r,
        "ts": rec.get("ts"),
        "generated": bool(t.get("generated")),
        "is_chat": False,
    }


# --------------------------------------------------------------------------
# validation — the point of looking at this at all
# --------------------------------------------------------------------------

def warnings_for(v: dict) -> list[str]:
    w = []
    if not (v.get("observation") or "").strip():
        w.append("no observation — the model had nothing to act on")
    # Missing reasoning is NOT a warning. It is the expected state of every
    # record — Claude Code strips thinking on write — so flagging it made
    # `--only-warnings` match all 123 records and show nothing. It is a
    # filter and a tag instead; the manifest counts it.
    if not v.get("task_id"):
        w.append("no task_id — not joined to any episode")
    if v.get("ok") is False:
        w.append("the action failed")

    shot = v.get("screenshot") or {}
    if shot:
        if shot.get("exists") is False:
            w.append("screenshot file is missing")
        if shot.get("frame_matches_viewport") is False:
            w.append("screenshot size disagrees with the viewport — "
                     "coordinates are in a different frame")
        if (shot.get("age_steps") or 0) > 3:
            # One bucket, not one message per distance — otherwise the
            # summary is twenty near-identical lines and the real problems
            # scroll off the top. The exact age is on the card.
            w.append("screenshot is stale (>3 steps old)")
    elif str(v.get("kind", "")).startswith("browse."):
        w.append("no screenshot for a browser action")

    # The messages format deliberately drops `effect`, so its absence there
    # says nothing about the underlying record. Checking it anyway flagged
    # all 26 clicks as coordinate-less when 22 of them had coordinates —
    # a validator that cries wolf is worse than no validator.
    if v.get("effect") is not None and str(v.get("kind")) == "browse.click":
        eff = v["effect"]
        if not eff.get("coords"):
            w.append("click has no coordinates — unusable for click training")
        elif not eff.get("viewport") or eff.get("scroll") is None:
            w.append("click coordinates have no frame (scroll/viewport)")
    return w


SEVERE = ("missing", "different frame", "nothing to act on", "no coordinates",
          "no frame")


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

CSS = """
:root{--bg:#fbfbfa;--fg:#1c1c1a;--mut:#6b6b66;--line:#e3e3df;--card:#fff;
      --warn:#8a5300;--warnbg:#fff8e8;--bad:#8c2f18;--badbg:#fdf0ec;
      --code:#f4f4f1;--accent:#2b5f8a}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--fg:#eceae4;--mut:#9c9a92;
      --line:#302e2a;--card:#1e1c19;--warn:#e0b060;--warnbg:#2a2114;
      --bad:#e08a70;--badbg:#2b1a15;--code:#26241f;--accent:#7fb0d8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);
       border-bottom:1px solid var(--line);padding:14px 20px}
h1{font-size:17px;margin:0 0 8px}
.sum{color:var(--mut);font-size:13px;margin-bottom:10px}
.sum b{color:var(--fg)}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
select,input,button{font:inherit;padding:5px 8px;border:1px solid var(--line);
     border-radius:6px;background:var(--card);color:var(--fg)}
input[type=search]{min-width:220px}
label.chk{display:flex;gap:5px;align-items:center;color:var(--mut);font-size:13px}
main{padding:16px 20px;max-width:1100px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      margin-bottom:14px;overflow:hidden}
.card.warn{border-color:#d8b878}
.card.bad{border-color:#c98a76}
.hd{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;
    padding:9px 14px;border-bottom:1px solid var(--line);background:var(--code)}
.kind{font-weight:650}
.tag{font-size:12px;color:var(--mut);border:1px solid var(--line);
     border-radius:20px;padding:1px 8px;background:var(--card)}
.tag.fail{color:var(--bad);border-color:var(--bad)}
.tag.gen{color:var(--accent);border-color:var(--accent)}
.role{font-size:11px;letter-spacing:.08em;text-transform:uppercase;font-weight:700;color:var(--mut);margin:0 0 6px;padding-top:12px;border-top:1px dashed var(--line)}
.role.first{padding-top:0;border-top:none}
.role .who{color:var(--accent)}
details.sysbox{margin-top:8px;color:var(--mut);font-size:12.5px}
details.sysbox summary{cursor:pointer}
details.sysbox pre{margin-top:6px}
.body{display:grid;grid-template-columns:1fr;gap:0}
@media(min-width:860px){.body.has-img{grid-template-columns:1fr 320px}}
.panes{padding:12px 14px}
.sec{margin-bottom:12px}
.sec:last-child{margin-bottom:0}
.lbl{font-size:11px;letter-spacing:.06em;text-transform:uppercase;
     color:var(--mut);margin-bottom:4px}
pre,.code{background:var(--code);border-radius:7px;padding:9px 11px;margin:0;
     overflow-x:auto;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
     white-space:pre-wrap;word-break:break-word}
pre.obs{max-height:260px;overflow-y:auto}
code{background:var(--code);padding:1px 4px;border-radius:4px;
     font:12.5px ui-monospace,Menlo,monospace}
.md p{margin:0 0 8px}.md p:last-child{margin:0}
.md ul,.md ol{margin:0 0 8px 20px;padding:0}
.md h3,.md h4{margin:10px 0 5px;font-size:14px}
.think{border-left:3px solid var(--accent);padding-left:10px;color:var(--mut)}
.ph{font:12.5px ui-monospace,Menlo,monospace;color:var(--warn)}
.act{background:#1c3a52;color:#e8f2fa;border-radius:7px;padding:9px 11px;
     font:12.5px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap;
     word-break:break-word}
@media(prefers-color-scheme:dark){.act{background:#12303f}}
.shot{padding:12px 14px 12px 0}
.shot img{width:100%;border:1px solid var(--line);border-radius:7px;display:block}
.shot .meta{font-size:11.5px;color:var(--mut);margin-top:5px;word-break:break-all}
.warns{background:var(--warnbg);border-top:1px solid var(--line);
       padding:8px 14px;font-size:12.5px;color:var(--warn)}
.warns.bad{background:var(--badbg);color:var(--bad)}
.warns li{margin-left:16px}
.none{color:var(--mut);font-style:italic}
.hidden{display:none}
.task{border:1px solid var(--line);border-radius:12px;margin-bottom:22px;padding:14px;background:color-mix(in srgb, var(--card) 60%, transparent)}
.taskhd{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline;padding-bottom:10px;margin-bottom:14px;border-bottom:2px solid var(--accent)}
.taskhd .tnum{font-weight:700;color:var(--accent)}
.taskhd .tid{font-weight:600}
.taskhd .tmeta{color:var(--mut);font-size:12.5px}
.taskhd .tgoal{flex-basis:100%;color:var(--mut);font-size:12.5px;font-style:italic}
.taskend{text-align:center;color:var(--mut);font-size:11px;letter-spacing:.08em;text-transform:uppercase;margin-top:6px;padding-top:8px;border-top:1px dashed var(--line)}
"""

JS = """
const cards=[...document.querySelectorAll('.card')];
const sections=[...document.querySelectorAll('.task')];
const q=document.getElementById('q'), k=document.getElementById('k'),
      t=document.getElementById('t'), w=document.getElementById('w'),
      g=document.getElementById('g'), se=document.getElementById('se'),
      n=document.getElementById('n');
function apply(){
  const qq=(q.value||'').toLowerCase(), kk=k.value, tt=t.value,
        ww=w.checked, gg=g.value, ss=se.value;
  let shown=0;
  for(const c of cards){
    let ok=true;
    if(kk && c.dataset.kind!==kk) ok=false;
    if(tt && c.dataset.think!==tt) ok=false;
    if(ww && c.dataset.warn==='0') ok=false;
    if(gg && c.dataset.task!==gg) ok=false;
    if(ss && c.dataset.session!==ss) ok=false;
    if(ok && qq && !c.textContent.toLowerCase().includes(qq)) ok=false;
    c.classList.toggle('hidden',!ok);
    if(ok) shown++;
  }
  for(const s of sections){
    const any=[...s.querySelectorAll('.card')].some(c=>!c.classList.contains('hidden'));
    s.classList.toggle('hidden',!any);
  }
  n.textContent=shown+' of '+cards.length+' shown';
}
[q,k,t,w,g,se].forEach(e=>e.addEventListener('input',apply));
apply();
"""


_GOAL_RE = re.compile(r"^Goal:\s*(.*)$", re.M)


def _task_groups(views: list[dict]) -> list[dict]:
    """Split views into consecutive runs that share (episode, task_id).

    That pair is the closest thing this data has to a task boundary: a new
    subagent episode, or a new job/profile/search within one episode, both
    mean the model started fresh on something else. Records already come in
    original order, so this is a single pass, not a sort/regroup.
    """
    groups: list[dict] = []
    for v in views:
        key = (v.get("episode"), v.get("task_id"))
        if not groups or groups[-1]["key"] != key:
            groups.append({"key": key, "views": []})
        groups[-1]["views"].append(v)
    for g in groups:
        goal = None
        for v in g["views"]:
            m = _GOAL_RE.search(v.get("header") or "")
            if m:
                goal = m.group(1)
                break
        g["goal"] = goal
        g["episode"], g["task_id"] = g["key"]
    return groups


def render(views: list[dict], title: str, embed: bool, base: Path,
           system_prompt: str | None = None) -> str:
    kinds = sorted({v["kind"] for v in views if v.get("kind")})
    session_counts: dict = {}
    for v in views:
        sid = v.get("session")
        if sid:
            session_counts[sid] = session_counts.get(sid, 0) + 1
    sessions = sorted(session_counts)
    n_warn = sum(1 for v in views if v["_w"])
    n_bad = sum(1 for v in views if any(s in x for x in v["_w"] for s in SEVERE))
    counts: dict = {}
    for v in views:
        counts[v.get("thinking_state") or "?"] = counts.get(v.get("thinking_state") or "?", 0) + 1
    groups = _task_groups(views)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
        "<header>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class='sum'><b>{len(views)}</b> records across <b>{len(groups)}</b> tasks"
        + (f" from <b>{len(sessions)}</b> sessions" if len(sessions) > 1 else "")
        + f" &middot; <b>{n_warn}</b> with warnings, <b>{n_bad}</b> serious &middot; "
        f"thinking: {', '.join(f'{v} {kk}' for kk, v in sorted(counts.items()))}</div>",
        "<div class='controls'>",
        "<input type='search' id='q' placeholder='search text…'>",
        "<select id='k'><option value=''>all kinds</option>"
        + "".join(f"<option>{html.escape(x)}</option>" for x in kinds) + "</select>",
        "<select id='t'><option value=''>any reasoning</option>"
        "<option value='present'>present</option>"
        "<option value='stripped'>stripped</option>"
        "<option value='absent'>absent</option></select>",
        "<select id='g'><option value=''>all tasks</option>"
        + "".join(
            "<option value='%d'>Task %d: %s (%d steps)</option>" % (
                i, i + 1, html.escape(str(g["task_id"] or "(none)")), len(g["views"]))
            for i, g in enumerate(groups))
        + "</select>",
        "<select id='se'><option value=''>all sessions</option>"
        + "".join(
            "<option value='%s'>%s… (%d steps)</option>" % (
                html.escape(sid), html.escape(sid[:8]), session_counts[sid])
            for sid in sessions)
        + "</select>",
        "<label class='chk'><input type='checkbox' id='w'> warnings only</label>",
        "<span class='sum' id='n'></span>",
        "</div>",
    ]
    if system_prompt:
        parts.append(
            "<details class='sysbox'><summary>system message "
            "(identical for every record in this file)</summary>"
            f"<pre>{html.escape(system_prompt)}</pre></details>")
    parts.append("</header><main>")

    i = 0  # running position across the whole file, independent of grouping
    for gi, g in enumerate(groups):
        n_gwarn = sum(1 for v in g["views"] if v["_w"])
        parts.append(f"<section class='task' data-task='{gi}'>")
        parts.append(
            "<div class='taskhd'><span class='tnum'>Task %d/%d</span>"
            "<span class='tid'>%s</span>"
            "<span class='tmeta'>%d steps &middot; %d warnings</span>%s</div>" % (
                gi + 1, len(groups),
                html.escape(str(g["task_id"] or "(none)")),
                len(g["views"]), n_gwarn,
                (f"<div class='tgoal'>Goal: {html.escape(g['goal'])}</div>" if g["goal"] else ""),
            ))

        for v in g["views"]:
            w = v["_w"]
            bad = any(s in x for x in w for s in SEVERE)
            cls = "card bad" if bad else ("card warn" if w else "card")
            shot = v.get("screenshot") or {}
            img = _img(shot, embed, base)

            parts.append(
                f"<div class='{cls}' data-kind='{html.escape(str(v.get('kind') or ''))}' "
                f"data-think='{html.escape(str(v.get('thinking_state') or ''))}' "
                f"data-task='{gi}' "
                f"data-session='{html.escape(str(v.get('session') or ''))}' "
                f"data-warn='{1 if w else 0}'>")
            tags = []
            if v.get("skill"):
                tags.append(v["skill"])
            if v.get("step"):
                tags.append(v["step"])
            if v.get("task_id"):
                tags.append(v["task_id"])
            if v.get("ts"):
                tags.append(str(v["ts"])[11:19])
            parts.append(
                "<div class='hd'><span class='tag'>row %d/%d</span>"
                "<span class='tag'>original #%s</span>"
                "<span class='kind'>%s</span>%s%s%s</div>" % (
                    i + 1, len(views),
                    html.escape(str(v.get("index"))),
                    html.escape(str(v.get("kind") or "?")),
                    "".join(f"<span class='tag'>{html.escape(str(x))}</span>" for x in tags),
                    "" if v.get("ok") is not False else "<span class='tag fail'>failed</span>",
                    "<span class='tag gen'>generated thinking</span>" if v.get("generated") else "",
                ))

            parts.append(f"<div class='body{' has-img' if img else ''}'><div class='panes'>")
            chat = v.get("is_chat")
            if chat:
                parts.append("<div class='role first'><span class='who'>user</span> message</div>")
            if v.get("header"):
                parts.append(_sec("context", f"<pre>{html.escape(v['header'])}</pre>"))
            parts.append(_sec("observation — what the model saw",
                              f"<pre class='obs'>{html.escape(_pretty(v.get('observation')))}</pre>"
                              if v.get("observation") else "<div class='none'>none</div>"))
            if chat:
                parts.append("<div class='role'><span class='who'>assistant</span> message</div>")

            th = v.get("thinking") or ""
            if th.startswith("<THINKING"):
                body = f"<div class='ph'>{html.escape(th)} — to be filled by a later pass</div>"
            elif th:
                body = f"<div class='md think'>{md(th)}</div>"
            else:
                body = "<div class='none'>none</div>"
            parts.append(_sec("thinking", body))

            parts.append(_sec("rationale — what it said at the time",
                              f"<div class='md'>{md(v['rationale'])}</div>"
                              if v.get("rationale") else "<div class='none'>none</div>"))
            parts.append(_sec("action",
                              f"<div class='act'>{html.escape(v.get('action') or '')}</div>"))
            if v.get("effect"):
                parts.append(_sec("effect",
                                  f"<pre>{html.escape(json.dumps(v['effect']))}</pre>"))
            if v.get("result") and v["result"].get("text"):
                parts.append(_sec("result",
                                  f"<pre class='obs'>{html.escape(_pretty(v['result']['text']))}</pre>"))
            parts.append("</div>")
            if img:
                parts.append("<div class='shot'>" + img + "</div>")
            parts.append("</div>")

            if w:
                parts.append("<ul class='warns%s'>%s</ul>" % (
                    " bad" if bad else "",
                    "".join(f"<li>{html.escape(x)}</li>" for x in w)))
            parts.append("</div>")
            i += 1

        parts.append(
            "<div class='taskend'>end of task %d &middot; %s</div>" % (
                gi + 1, html.escape(str(g["task_id"] or "(none)"))))
        parts.append("</section>")

    parts.append(f"</main><script>{JS}</script></body></html>")
    return "\n".join(parts)


def _sec(label: str, body: str) -> str:
    return f"<div class='sec'><div class='lbl'>{html.escape(label)}</div>{body}</div>"


def _pretty(text) -> str:
    """Pretty-print a JSON observation; leave anything else alone."""
    if not isinstance(text, str):
        return str(text)
    s = text.strip()
    if not s.startswith(("{", "[")):
        return text
    try:
        return json.dumps(json.loads(s), indent=1, ensure_ascii=False)
    except Exception:
        return text


def _img(shot: dict, embed: bool, base: Path) -> str:
    raw = shot.get("path")
    if not raw:
        return ""
    p = Path(shot.get("abs_path") or raw)
    if not p.exists():
        name = Path(str(raw).replace("\\", "/")).name
        for cand in (base / "images" / name,
                     base.parent / "traces" / "screenshots" / name):
            if cand.exists():
                p = cand
                break
    meta = []
    if shot.get("width"):
        meta.append(f"{shot['width']}x{shot['height']}")
    if shot.get("age_steps") is not None:
        meta.append(f"age {shot['age_steps']} steps")
    if shot.get("frame_matches_viewport") is False:
        meta.append("FRAME MISMATCH")
    if not p.exists():
        return ("<div class='meta'>screenshot missing:<br>"
                f"{html.escape(str(raw))}</div>")
    if embed:
        b64 = base64.b64encode(p.read_bytes()).decode()
        src = f"data:image/png;base64,{b64}"
    else:
        try:
            src = str(p.resolve().as_uri())
        except Exception:
            src = str(p)
    return (f"<img loading='lazy' src='{src}' alt=''>"
            f"<div class='meta'>{html.escape(' · '.join(meta))}<br>"
            f"{html.escape(p.name)}</div>")


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a .messages.jsonl or .steps.jsonl from `jobagent harvest`")
    ap.add_argument("-o", "--out", help="output .html (default: alongside the input)")
    ap.add_argument("--limit", type=int, help="only the first N records")
    ap.add_argument("--kind", action="append",
                    help="only these action kinds — `browse` for all browser "
                         "actions, or an exact kind. Repeatable")
    ap.add_argument("--only-warnings", action="store_true",
                    help="only records that failed a check")
    ap.add_argument("--only-serious", action="store_true",
                    help="only the checks that make a record untrainable — "
                         "a missing screenshot, a frame mismatch, an empty "
                         "observation, a click with no coordinates")
    ap.add_argument("--embed", action="store_true",
                    help="inline the screenshots as data URIs so the HTML is "
                         "one portable file (much larger)")
    args = ap.parse_args(argv)

    src = Path(args.input).expanduser()
    if not src.exists():
        print(f"no such file: {src}", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    if not rows:
        print("file is empty", file=sys.stderr)
        return 1

    to_view = from_messages if "messages" in rows[0] else from_steps
    views = [to_view(r) for r in rows]

    system_prompt = None
    if to_view is from_messages:
        sys_msgs = {m["role"]: m["content"] for m in rows[0].get("messages", [])}
        system_prompt = sys_msgs.get("system")

    if args.kind:
        want = set(args.kind)
        views = [v for v in views
                 if v.get("kind") in want
                 or str(v.get("kind", "")).split(".")[0] in want]
    for v in views:
        v["_w"] = warnings_for(v)
    if args.only_serious:
        views = [v for v in views
                 if any(sv in x for x in v["_w"] for sv in SEVERE)]
    elif args.only_warnings:
        views = [v for v in views if v["_w"]]
    if args.limit:
        views = views[:args.limit]

    out = Path(args.out) if args.out else src.with_suffix(".html")
    out.write_text(render(views, src.name, args.embed, src.parent, system_prompt),
                   encoding="utf-8")

    n_warn = sum(1 for v in views if v["_w"])
    n_bad = sum(1 for v in views if any(s in x for x in v["_w"] for s in SEVERE))
    print(f"{len(views)} records -> {out}")
    print(f"  {n_warn} with warnings, {n_bad} serious")
    tally: dict = {}
    for v in views:
        for x in v["_w"]:
            tally[x] = tally.get(x, 0) + 1
    for msg, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5}  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
