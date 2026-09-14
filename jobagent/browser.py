"""jobagent.browser — drive the user's own Chrome directly, over CDP.

Why this module exists
----------------------
Asking an MCP browser server to describe a LinkedIn page returns the whole
accessibility tree — 100k+ characters on a job page, a profile, search
results or the feed. That is bigger than the per-tool-result cap every host
enforces, so the call fails outright (see
`plugin/references/browser-adapter.md`, "`snapshot` output size").

The fix is not a bigger cap. It is to stop shipping the page across the wire.
Here the searching happens inside this process, against a live DOM, and only
the answer — a handful of elements, a form's fields — is printed. A `find`
that used to cost 100,000 characters costs about 600.

Everything in this file talks to a Chrome that is **already running and
already logged in**. We attach to it; we never launch a fresh browser and we
never touch cookies or credentials. Start Chrome once with the two flags
below (see `config/cdp.setup.md` for the Windows/macOS/Linux launch commands
in full):

    --remote-debugging-port=9222 --user-data-dir=<a profile dir just for this>

Recent Chrome refuses `--remote-debugging-port` on the default profile
directory, which is why the dedicated `--user-data-dir` is not optional. Log
into LinkedIn once in that profile and it stays logged in.

Contract
--------
Every public function returns a plain dict, small enough to print. Callers
(`cli.cmd_browse`) do not format; they print what they get.

Element identity across processes
---------------------------------
Each CLI invocation is a new process, so a Playwright element handle cannot
survive from `find` to `click`. Two mechanisms carry a ref instead:

1. `find`/`form` stamp `data-ja-ref="e3"` onto the elements they return.
   A later `click e3` resolves `[data-ja-ref="e3"]` instantly.
2. The same call writes a descriptor (role, name, tag, ordinal) to
   `data/.browse_refs.json`. If React has re-rendered and blown the
   attribute away, we re-find by that descriptor.

If both fail the ref is reported stale — which is the honest answer, and
means "run find again", not "retry blindly".
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from . import db as D

DEFAULT_CDP = os.environ.get("JOBAGENT_CDP", "http://127.0.0.1:9222")
# 8s was too tight: a real run on 5 Sep lost a click to `Locator.click:
# Timeout 8000ms exceeded` while LinkedIn was still settling. Waiting longer
# costs nothing when the page is quick — the timeout only ever elapses when
# something is actually wrong.
DEFAULT_TIMEOUT_MS = int(os.environ.get("JOBAGENT_BROWSE_TIMEOUT_MS", "15000"))

# How many times to re-run a page read that died because the page navigated
# underneath it. See `_eval`.
NAV_RETRIES = 2

# Keep every string we return bounded. A run should never be surprised by a
# 40k-character "name" pulled off some sprawling LinkedIn card.
MAX_NAME = 140
MAX_VALUE = 200
MAX_OPTIONS = 40
MAX_FIELDS = 60


class BrowserError(RuntimeError):
    """Something about the browser, not the page, is wrong."""


def set_timeout(ms: int | None) -> None:
    """Override the per-call timeout for this process.

    One CLI invocation is one process, so a module-level value is the whole
    of the state — no plumbing needed through every function signature.
    """
    global DEFAULT_TIMEOUT_MS
    if ms and int(ms) > 0:
        DEFAULT_TIMEOUT_MS = int(ms)


# --------------------------------------------------------------------------
# connection
# --------------------------------------------------------------------------

def refs_path(db_path=None) -> Path:
    p = Path(db_path).expanduser() if db_path else D.default_db_path()
    return p.parent / ".browse_refs.json"


def _load_refs(db_path=None) -> dict:
    try:
        return json.loads(refs_path(db_path).read_text(encoding="utf-8"))
    except Exception:
        return {"url": None, "ts": None, "refs": {}}


def _save_refs(payload: dict, db_path=None) -> None:
    try:
        p = refs_path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    except Exception:
        # Ref persistence is a convenience. Never fail a real action over it.
        pass


@contextmanager
def session(cdp_url: str | None = None, tab: str | None = None):
    """Attach to the running Chrome and yield (page, browser).

    `tab` is a substring matched against each open tab's URL. Default
    behaviour prefers a LinkedIn tab, because that is what this agent drives.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover - environment problem
        raise BrowserError(
            "playwright is not installed. Run: pip install playwright "
            "(no browser download needed — we attach to your own Chrome)"
        ) from e

    url = cdp_url or DEFAULT_CDP
    pw = sync_playwright().start()
    try:
        try:
            browser = pw.chromium.connect_over_cdp(url, timeout=DEFAULT_TIMEOUT_MS)
        except Exception as e:
            raise BrowserError(
                f"no Chrome listening on {url}. Start it with "
                f"--remote-debugging-port=9222 and a dedicated --user-data-dir, "
                f"then log into LinkedIn once. ({type(e).__name__})"
            ) from e

        pages = [p for c in browser.contexts for p in c.pages]
        pages = [p for p in pages if not p.is_closed()]
        if not pages:
            raise BrowserError("Chrome is running but has no open tabs")

        want = tab or "linkedin.com"
        chosen = next((p for p in pages if want in (p.url or "")), None)
        if chosen is None and tab:
            raise BrowserError(
                f"no open tab whose URL contains {tab!r}. "
                f"Open tabs: {[p.url for p in pages][:8]}"
            )
        if chosen is None:
            chosen = pages[0]

        chosen.set_default_timeout(DEFAULT_TIMEOUT_MS)
        try:
            yield chosen, browser
        finally:
            # Detach only. Closing the browser would close the user's Chrome.
            try:
                browser.close()
            except Exception:
                pass
    finally:
        try:
            pw.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------
# the in-page collectors
#
# These run inside Chrome. Everything expensive — walking the DOM, scoring
# matches, reading computed styles — happens there, and only the trimmed
# result crosses back. That is the entire point of this module.
# --------------------------------------------------------------------------

_JS_HELPERS = r"""
() => {
  window.__ja = window.__ja || {};

  window.__ja.visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return false;
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none') return false;
    if (parseFloat(s.opacity || '1') < 0.05) return false;
    return true;
  };

  window.__ja.name = (el) => {
    const pick = [
      el.getAttribute('aria-label'),
      (() => {
        const id = el.getAttribute('aria-labelledby');
        if (!id) return null;
        return id.split(/\s+/).map(i => {
          const n = document.getElementById(i);
          return n ? n.innerText : '';
        }).join(' ');
      })(),
      (() => {
        if (!el.id) return null;
        const lab = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
        return lab ? lab.innerText : null;
      })(),
      (() => {
        const lab = el.closest('label');
        return lab ? lab.innerText : null;
      })(),
      el.innerText,
      el.getAttribute('placeholder'),
      el.getAttribute('title'),
      el.getAttribute('alt'),
      el.value,
    ];
    for (const p of pick) {
      const t = (p || '').replace(/\s+/g, ' ').trim();
      if (t) return t;
    }
    return '';
  };

  window.__ja.role = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      const t = (el.type || 'text').toLowerCase();
      if (t === 'checkbox' || t === 'radio' || t === 'file') return t;
      if (t === 'submit' || t === 'button') return 'button';
      return 'textbox';
    }
    return tag;
  };

  window.__ja.box = (el) => {
    const r = el.getBoundingClientRect();
    return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)];
  };

  // The panel the user is actually looking at. LinkedIn's Easy Apply is a
  // dialog; when one is open, nothing outside it is reachable anyway, so
  // scoping to it cuts both noise and the chance of clicking the page behind.
  window.__ja.scope = () => {
    const dialogs = [...document.querySelectorAll('[role="dialog"], dialog[open]')]
      .filter(window.__ja.visible);
    if (dialogs.length) return dialogs[dialogs.length - 1];
    return document.body;
  };

  // A ref label (e.g. "e1") is only unique going forward if no earlier
  // element from a previous find/form/outline call still wears it — stale
  // stamps are never cleared otherwise, so two elements can share one ref
  // and `[data-ja-ref="e1"]` picks whichever is first in DOM order, not the
  // one this call just meant to label. Strip the label off every other
  // element first so the new stamp is the sole holder.
  window.__ja.stamp = (el, ref) => {
    document.querySelectorAll(`[data-ja-ref="${CSS.escape(ref)}"]`).forEach((old) => {
      if (old !== el) old.removeAttribute('data-ja-ref');
    });
    el.setAttribute('data-ja-ref', ref);
  };

  return true;
}
"""

_JS_FIND = r"""
({query, limit, scoped}) => {
  const SEL = 'a,button,input,select,textarea,summary,' +
    '[role=button],[role=link],[role=combobox],[role=checkbox],[role=radio],' +
    '[role=tab],[role=menuitem],[role=option],[role=switch],[contenteditable=true]';
  const root = scoped ? window.__ja.scope() : document.body;
  const q = (query || '').toLowerCase().trim();
  const qTokens = q.split(/\s+/).filter(Boolean);

  const score = (name) => {
    const n = (name || '').toLowerCase();
    if (!q) return 1;
    if (!n) return 0;
    if (n === q) return 100;
    if (n.startsWith(q)) return 85;
    if (n.includes(q)) return 70;
    const hits = qTokens.filter(t => n.includes(t)).length;
    if (!hits) return 0;
    return Math.round(40 * hits / qTokens.length);
  };

  const seen = new Set();
  const out = [];
  for (const el of root.querySelectorAll(SEL)) {
    if (!window.__ja.visible(el)) continue;
    const name = window.__ja.name(el);
    const s = score(name);
    if (s <= 0) continue;
    // Drop a wrapper whose only content is a child we already scored the same.
    const key = window.__ja.role(el) + '|' + name + '|' + window.__ja.box(el).join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ el, name, s });
  }

  out.sort((a, b) => b.s - a.s || a.name.length - b.name.length);
  const top = out.slice(0, limit || 5);

  return top.map((r, i) => {
    const ref = 'e' + (i + 1);
    window.__ja.stamp(r.el, ref);
    return {
      ref,
      role: window.__ja.role(r.el),
      tag: r.el.tagName.toLowerCase(),
      name: r.name.slice(0, 140),
      bbox: window.__ja.box(r.el),
      enabled: !r.el.disabled && r.el.getAttribute('aria-disabled') !== 'true',
      score: r.s,
    };
  });
}
"""

_JS_FORM = r"""
({limit}) => {
  const scope = window.__ja.scope();
  const inDialog = scope !== document.body;

  const labelFor = (el) => {
    let n = window.__ja.name(el);
    if (n) return n;
    // LinkedIn often puts the question in a sibling above the control.
    const grp = el.closest('[data-test-form-element], fieldset, .fb-dash-form-element, div');
    if (grp) {
      const lab = grp.querySelector('label, legend, .fb-dash-form-element__label');
      if (lab) return (lab.innerText || '').replace(/\s+/g, ' ').trim();
    }
    return '';
  };

  const errorFor = (el) => {
    const grp = el.closest('[data-test-form-element], fieldset, div');
    if (!grp) return '';
    const err = grp.querySelector('[role=alert], .artdeco-inline-feedback--error, [class*="error"]');
    if (!err || !window.__ja.visible(err)) return '';
    return (err.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 140);
  };

  const fields = [];
  const radios = new Map();
  let i = 0;

  for (const el of scope.querySelectorAll('input,select,textarea,[contenteditable=true]')) {
    if (!window.__ja.visible(el)) continue;
    const type = (el.type || '').toLowerCase();
    if (type === 'hidden') continue;

    if (type === 'radio') {
      const key = el.name || labelFor(el);
      if (!radios.has(key)) radios.set(key, []);
      radios.get(key).push(el);
      continue;
    }


    i += 1;
    const ref = 'f' + i;
    window.__ja.stamp(el, ref);
    const rec = {
      ref,
      kind: el.tagName.toLowerCase() === 'select' ? 'select'
          : el.tagName.toLowerCase() === 'textarea' ? 'textarea'
          : (type || 'text'),
      label: labelFor(el).slice(0, 140),
      value: String(el.value == null ? '' : el.value).slice(0, 200),
      required: el.required || el.getAttribute('aria-required') === 'true',
    };
    if (type === 'checkbox') rec.checked = el.checked;
    if (el.maxLength && el.maxLength > 0) rec.max_length = el.maxLength;
    if (el.tagName.toLowerCase() === 'select') {
      rec.options = [...el.options].slice(0, 40).map(o => o.text.trim());
    }
    const err = errorFor(el);
    if (err) rec.error = err;
    fields.push(rec);
    if (fields.length >= (limit || 60)) break;
  }

  for (const [key, els] of radios) {
    i += 1;
    const ref = 'f' + i;
    // Stamp each option separately (f7.1, f7.2) so `check` can pick one,
    // and the group itself so `form` can be re-read against it.
    els.forEach((e, n) => window.__ja.stamp(e, ref + '.' + (n + 1)));
    const chosen = els.find(e => e.checked);
    // The question lives on the fieldset's legend, not on the input's name
    // attribute — "reloc" is not something a caller can answer.
    const fs = els[0].closest('fieldset, [data-test-form-element]');
    const legend = fs ? fs.querySelector('legend, label:not(:has(input))') : null;
    const question = legend ? (legend.innerText || '').replace(/\s+/g, ' ').trim() : '';
    const rec = {
      ref,
      kind: 'radio',
      label: (question || String(key || '')).slice(0, 140),
      options: els.map((e, n) => ({ ref: ref + '.' + (n + 1), name: window.__ja.name(e).slice(0, 60) })),
      value: chosen ? window.__ja.name(chosen).slice(0, 60) : '',
      required: els.some(e => e.required || e.getAttribute('aria-required') === 'true'),
    };
    const rerr = errorFor(els[0]);
    if (rerr) rec.error = rerr;
    fields.push(rec);
  }

  // The buttons that move the form forward, so a caller does not need a
  // second round trip just to learn whether it says Next or Submit.
  const actions = [];
  let b = 0;
  for (const el of scope.querySelectorAll('button,[role=button]')) {
    if (!window.__ja.visible(el)) continue;
    const name = window.__ja.name(el);
    if (!name) continue;
    b += 1;
    const ref = 'b' + b;
    window.__ja.stamp(el, ref);
    actions.push({
      ref,
      name: name.slice(0, 80),
      enabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
    });
    if (actions.length >= 12) break;
  }

  return {
    in_dialog: inDialog,
    heading: (scope.querySelector('h1,h2,[role=heading]') || {}).innerText
      ? (scope.querySelector('h1,h2,[role=heading]').innerText || '').replace(/\s+/g, ' ').trim().slice(0, 140)
      : '',
    fields,
    actions,
  };
}
"""

_JS_OUTLINE = r"""
({limit, chars}) => {
  // "What is on this page?" — the one thing `find` and `form` cannot answer,
  // because both need you to already know what you are looking for.
  //
  // A snapshot answered it by returning everything, which is why it kept
  // failing. This answers it in a fixed budget instead: headings, the one
  // repeated block of rows that carries the page's actual content, and the
  // buttons. Bounded by construction, not by luck about the page.

  const headings = [];
  const seenH = new Set();
  for (const h of document.querySelectorAll('h1,h2,[role=heading]')) {
    if (!window.__ja.visible(h)) continue;
    const t = (h.innerText || '').replace(/\s+/g, ' ').trim();
    if (!t || seenH.has(t)) continue;
    seenH.add(t);
    headings.push(t.slice(0, 100));
    if (headings.length >= 8) break;
  }

  // Every list-ish element, grouped by its parent. Siblings under one parent
  // are the repeated structure — search result cards, inbox threads, the
  // hiring-team panel. Nesting sorts itself out: an inner row has a
  // different parent than the card containing it.
  const groups = new Map();
  for (const el of document.querySelectorAll('li,[role=listitem],[role=row],article,[role=article]')) {
    if (!window.__ja.visible(el)) continue;
    const t = (el.innerText || '').replace(/\s+/g, ' ').trim();
    if (!t) continue;
    const p = el.parentElement;
    if (!p) continue;
    if (!groups.has(p)) groups.set(p, []);
    groups.get(p).push({ el, t });
  }

  // Pick the group by total text, not by count. A nav bar has eight items of
  // two words; a results list has eight items of forty. Text is what
  // separates the content from the chrome around it.
  let best = null, bestScore = 0;
  for (const [, items] of groups) {
    if (items.length < 3) continue;
    const score = items.reduce((a, b) => a + Math.min(b.t.length, 300), 0);
    if (score > bestScore) { bestScore = score; best = items; }
  }

  const rows = [];
  if (best) {
    best.slice(0, limit || 30).forEach((it, n) => {
      const ref = 'i' + (n + 1);
      // Stamp the thing worth clicking, not the wrapper around it.
      const target = it.el.querySelector('a[href]') || it.el.querySelector('button') || it.el;
      window.__ja.stamp(target, ref);
      const rec = { ref, text: it.t.slice(0, chars || 120) };
      // The href carries the job id, which is what dedupe runs on — cheaper
      // to hand it over now than to make the caller open the card to find it.
      if (target.tagName === 'A' && target.href) rec.href = target.href.slice(0, 160);
      rows.push(rec);
    });
  }

  const scope = document.querySelector('main, [role=main]') || document.body;
  const actions = [];
  let b = 0;
  for (const el of scope.querySelectorAll('button,[role=button]')) {
    if (!window.__ja.visible(el)) continue;
    const name = window.__ja.name(el);
    if (!name) continue;
    b += 1;
    const ref = 'b' + b;
    window.__ja.stamp(el, ref);
    actions.push({ ref, name: name.slice(0, 60) });
    if (actions.length >= 10) break;
  }

  return {
    headings,
    rows,
    rows_total: best ? best.length : 0,
    actions,
    dialog_open: window.__ja.scope() !== document.body,
  };
}
"""

_JS_STATE = r"""
() => ({
  url: location.href,
  title: document.title,
  viewport: [window.innerWidth, window.innerHeight],
  scroll: [Math.round(window.scrollX), Math.round(window.scrollY)],
  dpr: window.devicePixelRatio || 1,
})
"""


_NAV_ERRORS = (
    "execution context was destroyed",
    "cannot find context with specified id",
    "target closed",
    "frame was detached",
    "navigating and changing the content",
)


def _eval(page, script, arg=None, tries: int = NAV_RETRIES):
    """Run a page script, surviving a navigation that lands mid-read.

    LinkedIn navigates on its own — a click resolves, the SPA swaps the view,
    and a read that started a moment earlier dies with "Execution context was
    destroyed". Two real runs on 5 Sep lost reads exactly this way. It isn't
    a failure of the read; the page simply moved. So wait for the new
    document and ask again.

    Anything that is not a navigation error is re-raised immediately — a
    broken selector should fail loudly, not three times slowly.
    """
    last = None
    for attempt in range(tries + 1):
        try:
            return page.evaluate(script, arg) if arg is not None else page.evaluate(script)
        except Exception as e:  # noqa: BLE001 — Playwright raises broadly here
            msg = str(e).lower()
            if not any(t in msg for t in _NAV_ERRORS) or attempt == tries:
                raise
            last = e
            try:
                page.wait_for_load_state("domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)
            except Exception:
                pass
            page.wait_for_timeout(350)
            # The helpers live on `window`, so a new document has lost them.
            try:
                page.evaluate(_JS_HELPERS)
            except Exception:
                pass
    raise last  # unreachable; keeps the type checker honest


def _prime(page) -> None:
    _eval(page, _JS_HELPERS)


def _state(page) -> dict:
    return _eval(page, _JS_STATE)


# --------------------------------------------------------------------------
# ref resolution
# --------------------------------------------------------------------------

def _remember(page, items, db_path=None) -> None:
    st = _state(page)
    _save_refs(
        {
            "url": st["url"],
            "ts": time.time(),
            "refs": {
                it["ref"]: {
                    # `find` calls it name, `form` calls it label, `outline`
                    # calls it text — all three are what we re-find by.
                    "name": it.get("name") or it.get("label") or it.get("text", ""),
                    "role": it.get("role") or it.get("kind", ""),
                    "tag": it.get("tag", ""),
                }
                for it in items
            },
        },
        db_path,
    )


def _locate(page, ref: str, db_path=None):
    """Resolve a ref to a Playwright locator, re-finding it if React nuked
    the stamped attribute."""
    loc = page.locator(f'[data-ja-ref="{ref}"]')
    try:
        if loc.count() > 0:
            return loc.first
    except Exception:
        pass

    desc = _load_refs(db_path).get("refs", {}).get(ref)
    if not desc or not desc.get("name"):
        raise BrowserError(
            f"ref {ref!r} is stale and cannot be recovered — run `browse find` "
            f"or `browse form` again to get fresh refs"
        )

    _prime(page)
    hits = _eval(page, _JS_FIND, {"query": desc["name"], "limit": 3, "scoped": True})
    for h in hits:
        if h["role"] == desc.get("role") or h["name"] == desc["name"]:
            return page.locator(f'[data-ja-ref="{h["ref"]}"]').first
    raise BrowserError(
        f"ref {ref!r} ({desc['name'][:60]!r}) is no longer on the page — "
        f"the form probably advanced; run `browse form` again"
    )


# --------------------------------------------------------------------------
# public operations
# --------------------------------------------------------------------------

def status(cdp_url=None, tab=None) -> dict:
    with session(cdp_url, tab) as (page, browser):
        pages = [p for c in browser.contexts for p in c.pages if not p.is_closed()]
        return {
            "ok": True,
            "connected": True,
            "cdp": cdp_url or DEFAULT_CDP,
            "active": _state(page),
            "tabs": [{"title": (p.title() or "")[:80], "url": p.url} for p in pages[:10]],
        }


def open_url(url: str, cdp_url=None, tab=None, wait="domcontentloaded",
             new_tab: bool = False) -> dict:
    """Navigate. `new_tab` opens a fresh tab instead of taking over one.

    Default is to reuse the tab, because that is what a run wants — one tab,
    human pace. `--new-tab` is for when you must not disturb what is already
    on screen.
    """
    with session(cdp_url, None if new_tab else tab) as (page, browser):
        if new_tab:
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            page = ctx.new_page()
            page.set_default_timeout(DEFAULT_TIMEOUT_MS)
        page.goto(url, wait_until=wait)
        return {"ok": True, "new_tab": new_tab, **_state(page)}


def find(query: str, limit: int = 5, scoped: bool = True, cdp_url=None, tab=None,
         db_path=None) -> dict:
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        hits = _eval(page, _JS_FIND, {"query": query, "limit": limit, "scoped": scoped})
        _remember(page, hits, db_path)
        st = _state(page)
        return {
            "ok": True,
            "query": query,
            "url": st["url"],
            "scroll": st["scroll"],
            "viewport": st["viewport"],
            "count": len(hits),
            "matches": hits,
        }


def form(limit: int = MAX_FIELDS, cdp_url=None, tab=None, db_path=None) -> dict:
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        res = _eval(page, _JS_FORM, {"limit": limit})
        _remember(page, res["fields"] + res["actions"], db_path)
        st = _state(page)
        return {"ok": True, "url": st["url"], "scroll": st["scroll"], **res}


def outline(limit: int = 30, chars: int = 120, cdp_url=None, tab=None,
            db_path=None) -> dict:
    """Where am I and what is on this page?

    The bounded replacement for a snapshot. `find` needs a search term and
    `form` needs a dialog; this needs neither, which is what the search,
    inbox and hiring-team phases were missing.

    `dialog_open: true` means a modal is up and `browse form` is the better
    call. `rows_total` larger than the rows returned means the list continues
    below — scroll and call again.
    """
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        res = _eval(page, _JS_OUTLINE, {"limit": limit, "chars": chars})
        _remember(page, res["rows"] + res["actions"], db_path)
        st = _state(page)
        # Say what to do next when the outline alone isn't the answer, so a
        # caller doesn't sit there re-running it on a page that has no list.
        if res.get("dialog_open"):
            res["hint"] = "a dialog is open — `browse form` reads its fields and buttons"
        elif not res["rows"]:
            res["hint"] = ("no repeated rows on this page — it's prose, not a list. "
                           "Use `browse text` for the content, `browse find` for a control")
        elif res["rows_total"] > len(res["rows"]):
            res["hint"] = (f"showing {len(res['rows'])} of {res['rows_total']} rows — "
                           f"raise --limit, or `browse scroll` to load more")
        return {
            "ok": True,
            "url": st["url"],
            "title": st["title"][:120],
            "scroll": st["scroll"],
            **res,
        }


def click(ref: str | None = None, text: str | None = None, cdp_url=None, tab=None,
          db_path=None) -> dict:
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        if ref:
            loc = _locate(page, ref, db_path)
        elif text:
            hits = _eval(page, _JS_FIND, {"query": text, "limit": 1, "scoped": True})
            if not hits:
                raise BrowserError(f"nothing on the page matches {text!r}")
            ref = hits[0]["ref"]
            loc = page.locator(f'[data-ja-ref="{ref}"]').first
        else:
            raise BrowserError("click needs --ref or --text")

        name = (loc.inner_text() or "").strip()[:80] or (loc.get_attribute("aria-label") or "")[:80]
        # Scroll first, then measure. Measuring before the scroll would record
        # a box from the old viewport — a coordinate that no screenshot of
        # this moment agrees with, which is exactly the corpus bug we are
        # trying not to reintroduce.
        loc.scroll_into_view_if_needed()
        page.wait_for_timeout(150)
        box = loc.bounding_box()
        before = _state(page)

        # A LinkedIn button can be briefly covered by a toast, a sticky
        # header, or its own loading state. One timeout is not proof the
        # element is unclickable — it usually means "not yet". Retry once
        # against a freshly resolved locator, since the first attempt may
        # have been holding a handle the SPA has since replaced.
        attempts = 0
        while True:
            attempts += 1
            try:
                loc.click()
                break
            except Exception as e:  # noqa: BLE001 — Playwright's timeout type varies
                if attempts > 2 or "timeout" not in str(e).lower():
                    raise
                page.wait_for_timeout(1200)
                try:
                    page.wait_for_load_state("domcontentloaded",
                                             timeout=DEFAULT_TIMEOUT_MS)
                except Exception:
                    pass
                _prime(page)
                loc = _locate(page, ref, db_path)
                loc.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        st = _state(page)
        return {
            "ok": True,
            "clicked": ref,
            "name": name,
            # Viewport CSS pixels, the same frame `browse shot` captures in —
            # so a logged coordinate and its screenshot finally agree.
            "coords": [round(box["x"] + box["width"] / 2), round(box["y"] + box["height"] / 2)] if box else None,
            "bbox": [round(box["x"]), round(box["y"]), round(box["width"]), round(box["height"])] if box else None,
            # The scroll offset the coordinates above belong to. `trace log`
            # needs this to place a click inside its screenshot.
            "scroll": before["scroll"],
            "viewport": before["viewport"],
            "url": st["url"],
            "navigated": st["url"] != before["url"],
            # Surfaced rather than hidden: a click that needed a second go
            # says something about the page, and a corpus that silently
            # smooths over retries teaches a model the page is calmer than
            # it is.
            "attempts": attempts,
        }


def fill(ref: str, value: str, blur: bool = True, cdp_url=None, tab=None,
         db_path=None) -> dict:
    """Type into a field the way LinkedIn's React forms require.

    Select, type, blur — never assign to `.value`. Validation runs on blur
    only, so we blur before reporting, and read back what the field actually
    holds rather than trusting what we sent.
    """
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        loc = _locate(page, ref, db_path)
        loc.scroll_into_view_if_needed()
        loc.click()
        try:
            loc.press("Control+a")
        except Exception:
            pass
        loc.type(value, delay=35)
        if blur:
            loc.press("Tab")
            page.wait_for_timeout(300)
        readback = ""
        try:
            readback = loc.input_value()
        except Exception:
            readback = (loc.get_attribute("value") or "")
        res = _eval(page, _JS_FORM, {"limit": MAX_FIELDS})
        field = next((f for f in res["fields"] if f["ref"] == ref), None)
        return {
            "ok": True,
            "ref": ref,
            "typed": value[:MAX_VALUE],
            "value": readback[:MAX_VALUE],
            "matches": readback.strip() == value.strip(),
            "error": (field or {}).get("error", ""),
        }


def select(ref: str, value: str, cdp_url=None, tab=None, db_path=None) -> dict:
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        loc = _locate(page, ref, db_path)
        loc.scroll_into_view_if_needed()
        try:
            loc.select_option(label=value)
        except Exception:
            loc.select_option(value=value)
        page.wait_for_timeout(250)
        return {"ok": True, "ref": ref, "value": loc.input_value()}


def check(ref: str, on: bool = True, cdp_url=None, tab=None, db_path=None) -> dict:
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        loc = _locate(page, ref, db_path)
        loc.scroll_into_view_if_needed()
        loc.check() if on else loc.uncheck()
        page.wait_for_timeout(200)
        return {"ok": True, "ref": ref, "checked": loc.is_checked()}


def upload(ref: str, path: str, cdp_url=None, tab=None, db_path=None) -> dict:
    """Attach a real file — the capability Browser MCP never had."""
    f = Path(path).expanduser().resolve()
    if not f.exists():
        raise BrowserError(f"no such file: {f}")
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        loc = _locate(page, ref, db_path)
        loc.set_input_files(str(f))
        page.wait_for_timeout(1200)
        return {"ok": True, "ref": ref, "file": f.name, "bytes": f.stat().st_size}


def press(key: str, cdp_url=None, tab=None) -> dict:
    with session(cdp_url, tab) as (page, _):
        page.keyboard.press(key)
        page.wait_for_timeout(250)
        return {"ok": True, "key": key, **_state(page)}


def scroll(dy: int = 600, cdp_url=None, tab=None) -> dict:
    with session(cdp_url, tab) as (page, _):
        page.mouse.wheel(0, dy)
        page.wait_for_timeout(400)
        return {"ok": True, **_state(page)}


def text(selector: str | None = None, limit: int = 4000, cdp_url=None, tab=None) -> dict:
    """Visible text of one region — not the whole page.

    `--limit` exists because the caller is an LLM with a token budget; the
    default of 4000 characters is a paragraph of job description, not a feed.
    """
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        if selector:
            body = page.locator(selector).first.inner_text()
        else:
            body = _eval(page, "() => window.__ja.scope().innerText")
        body = " ".join((body or "").split())
        return {
            "ok": True,
            "chars": len(body),
            "truncated": len(body) > limit,
            "text": body[:limit],
        }


def shot(label: str = "", db_path=None, cdp_url=None, tab=None, full: bool = False) -> dict:
    """Save a viewport screenshot into the trace folder and return its path.

    Viewport, not full page: the coordinates `click` reports are viewport CSS
    pixels, and a full-page capture would put them in a different frame — the
    exact mismatch the 4 Sep corpus audit flagged.
    """
    tdir = D.traces_dir(db_path) / "screenshots"
    tdir.mkdir(parents=True, exist_ok=True)
    with session(cdp_url, tab) as (page, _):
        st = _state(page)
        stamp = hashlib.sha1(f"{st['url']}{time.time()}".encode()).hexdigest()[:12]
        name = f"{D.today()}-{stamp}.png"
        dest = tdir / name
        page.screenshot(path=str(dest), full_page=full)
        return {
            "ok": True,
            "path": str(dest),
            "label": label,
            "viewport": st["viewport"],
            "scroll": st["scroll"],
            "dpr": st["dpr"],
            "url": st["url"],
        }


def dump(label: str, db_path=None, cdp_url=None, tab=None) -> dict:
    """Write the full page text to disk and return only the path.

    This is the honest version of the thing that was breaking: the whole page
    still gets captured for the training corpus, it just goes to a file
    instead of through the conversation.
    """
    pdir = D.traces_dir(db_path) / "pages"
    pdir.mkdir(parents=True, exist_ok=True)
    with session(cdp_url, tab) as (page, _):
        _prime(page)
        st = _state(page)
        body = _eval(page, "() => document.body.innerText") or ""
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:60]
        dest = pdir / f"{D.today()}-{slug}.page_text.txt"
        dest.write_text(body, encoding="utf-8")
        return {
            "ok": True,
            "path": str(dest),
            "chars": len(body),
            "url": st["url"],
            "scroll": st["scroll"],
        }
