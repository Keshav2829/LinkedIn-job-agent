# Browser driver adapter

The agent can drive Chrome three ways. This file is the only place that knows
the difference; every skill talks in the verbs below and lets this table
resolve them.

**Preference order: CDP first, then Claude in Chrome, then Browser MCP.** They
must never drive the same tab in the same run — pick one at the start and keep
it.

CDP is first because it is the only driver whose page reads cannot fail on
size (see "`snapshot` output size" below), and the only one that can attach a
resume everywhere. It is also the only one that needs setup on the user's
machine — `config/cdp.setup.md`, about five minutes, once.

## Choosing a driver

At the top of a run:

1. Run `python -m jobagent browse status`. If it returns `connected: true` →
   **use CDP**. If it reports no Chrome listening, don't try to fix it
   mid-run; fall through, and mention at the end that CDP was unavailable.
2. Else if `mcp__claude-in-chrome__*` tools are present → **use Claude in
   Chrome**.
3. Else if Browser MCP tools are present → **use Browser MCP**, and apply the
   degradations in the last section.
4. Else → stop. Tell the user no browser driver is connected and what to start.

Say which driver you picked in the run summary. When a step fails in a way that
looks driver-specific, name the driver in the error — it is the first thing to
check.

## Verb table

All CDP verbs are `python -m jobagent browse …`, abbreviated `browse …` below.

| Verb the skills use | CDP (`jobagent browse`) | Claude in Chrome | Browser MCP |
|---|---|---|---|
| open a tab | `browse open --url … --new-tab` | `tabs_create_mcp` | (uses the tab you connected the extension to) |
| list tabs | `browse status` | `tabs_context_mcp` | — |
| close a tab | — | `tabs_close_mcp` | — |
| go to URL | `browse open --url` (`--tab <url-substring>` to pick which) | `navigate` | `navigate` |
| back / forward | — (navigate by URL) | `navigate url:"back"` | `goBack` / `goForward` |
| **what's on this page** | `browse outline` | `read_page`, if it fits | `snapshot`, if it fits |
| read structure | `browse form` (the open dialog's fields + buttons) | `read_page` (accessibility tree, refs) | `snapshot` (accessibility tree, refs) |
| read article text | `browse text [--selector] [--limit]` | `get_page_text` | `snapshot`, then read the text nodes |
| find by description | `browse find -q "<text>"` (exact-then-fuzzy on the accessible name) | `find` (natural language) | — **no equivalent**, read `snapshot` yourself |
| click | `browse click --ref e1` (or `--query`) | `computer` `left_click` (ref or coords) | `click` (ref) |
| triple-click | — (`browse fill` already clears the field) | `computer` `triple_click` | `click` ×3, or `type` after selecting |
| type text | `browse fill --ref f1 --value` (clicks, types, blurs) | `computer` `type` | `type` |
| set a form value | `browse fill` / `browse select` / `browse check` | `form_input` | `type` / `selectOption` |
| dropdown | `browse select --ref f3 --value` | `form_input` | `selectOption` |
| hover | — | `computer` `hover` | `hover` |
| press a key | `browse press --key Escape` | `computer` `key` | `pressKey` |
| scroll | `browse scroll --dy 600` | `computer` `scroll` | — |
| wait | (built into each verb) | `computer` `wait` | `wait` |
| screenshot | `browse shot --label <name>` (saves to traces, returns the path) | `computer` `screenshot` | `screenshot` |
| dump full page for the corpus | `browse dump --label <name>` | — (hits the size wall) | — (hits the size wall) |
| console logs | — | `read_console_messages` | `getConsoleLogs` |
| network | — | `read_network_requests` | — none |
| **upload a file** | `browse upload --ref f4 --path` | `file_upload` | — **none** |
| run JS | — | `javascript_tool` | — none |

### `browse outline` — the bounded snapshot

`find` needs a search term and `form` needs an open dialog. `outline` needs
neither, which is why the search, inbox and hiring-team phases need it: it is
the answer to "where am I and what is here".

It returns the page's headings, the one repeated block of rows that carries
the content — search result cards, message threads, the people on the hiring
team — with a ref and an href each, and the main buttons. It picks that block
by total text, so a nav bar of six two-word links loses to a list of six job
cards, and the surrounding chrome never crowds out the content.

The output is bounded by construction: 30 rows × 120 characters by default
(`--limit`, `--chars`), so roughly 1–3k characters on any page. It cannot hit
the cap the way a tree can.

Read the fields it returns:

- **`href` on a row** carries the job id. Run dedupe (`job seen --job-id`)
  straight off it rather than opening the card to find out.
- **`dialog_open: true`** means a modal is up — `browse form` is the better
  call for what's inside it.
- **`rows_total` above the rows returned** means the list continues; raise
  `--limit` or `browse scroll` and call again.
- **`rows: []`** means the page is prose, not a list (a job description, a
  profile). Use `browse text` for the content and `browse find` for controls.
- **`hint`** says which of those cases you're in, in words.

### When CDP retries, and when it doesn't

Two things go wrong on a live LinkedIn page, and both were seen in the 5 Sep
run. Neither is a reason for the agent to retry by hand — the driver already
does, once, and tells you.

**The page navigated mid-read.** LinkedIn is a single-page app; a click
resolves, the view swaps, and a read that started a moment earlier dies with
`Execution context was destroyed`. Every `browse` read now waits for the new
document, re-installs its helpers and asks again (twice, then gives up). A
script error — a bad selector, say — is raised immediately instead, because
failing three times slowly helps nobody.

**A click timed out.** The default was 8s and a real click lost to it while
the page was still settling. It is now 15s (`--timeout`, or
`$JOBAGENT_BROWSE_TIMEOUT_MS`), and a click that still times out is retried
once against a freshly resolved locator — the usual cause is a toast or
sticky header covering the button for a second, not an unclickable element.

`browse click` reports `attempts` in its result. A 2 there is worth keeping in
the trace: it says something true about the page, and a corpus that smooths
over retries teaches a model the page is calmer than it is.

### Refs on CDP

`browse outline` returns `i1, i2…` for rows; `browse find` returns
`e1, e2…`; `browse form` returns `f1, f2…` for fields,
`f6.1, f6.2…` for the options inside a radio group, and `b1, b2…` for the
dialog's buttons. Pass those to `click` / `fill` / `select` / `check` /
`upload`.

Refs survive between commands even though each is a separate process: the
element is stamped in the DOM and its description is cached. If React has
re-rendered past both, the command says the ref is stale — that means run
`browse find` or `browse form` again, **not** retry.

`browse fill` reports the field's value read back after blur, plus any
validation error that appeared. That is the "verify before advancing" rule
below, enforced by the tool instead of remembered by the agent — if `error` is
non-empty, fix it before clicking Review.

Browser MCP tool names are registered without a `browser_` prefix in the source
(`snapshot`, `click`, `type`, `selectOption`, `hover`, `drag`). Your client
namespaces them — `mcp__browsermcp__snapshot` when configured locally, or
`mcp__remote-devices__browsermcp__snapshot` when proxied from the desktop app
into a cloud session. **Confirm the exact names once after install** by listing
your tools; correct this table if they differ.

## Rules that hold on both drivers

These came out of a live run on 29 Aug 2026 and are not driver-specific:

- **Type, don't set.** LinkedIn's React forms ignore a value written straight to
  the DOM. Select the field, type into it, then blur it by clicking the next
  field. Validation only runs on blur, so a stale "Invalid input" proves
  nothing until you have blurred.
- **Verify before advancing.** Screenshot or zoom the field group and confirm no
  red error remains before clicking Review.
- **One tab, human pace**, 25–70 s between meaningful actions.

## Degradations on Browser MCP

| Capability | What to do instead |
|---|---|
| **Resume attachment** (`file_upload`) | Send the resume as a **link**. `profile.resume_url` is rendered into the message via `{resume_note}`; if that field is empty, stop and ask the user for a URL rather than sending a message that promises an attachment it can't add. |
| **Natural-language `find`** | Take a `snapshot` and locate the element yourself. Slower and more tokens; budget for it — and see "`snapshot` output size" below, because on most LinkedIn pages this is not just slower, it can fail outright. |
| **Tab management** | The extension is bound to a tab the user connected. Don't assume you can open or close tabs — navigate the one you have. |
| **Network / console debugging** | `getConsoleLogs` only. No network inspection. |

Everything else — search, scoring, Easy Apply including screening questions,
connection requests, and sending an unattached message — works on both.

## Degradations on Claude in Chrome

None that affect this agent, other than the page-size wall below, which it
hits less often than Browser MCP but still hits on the feed and on long
profiles.

### `snapshot` output size — verified 5 Sep 2026, live run

**The CDP driver is the fix for everything in this section.** `browse find`
and `browse form` cannot hit the cap: matching happens inside Chrome and only
the matches are printed — a few hundred characters where a tree is a hundred
thousand. The paragraphs below stay because they still describe what happens
on the two MCP drivers, which remain the fallback.


**`snapshot` on a real LinkedIn page routinely exceeds the host's max
tool-output size and fails outright, not partially.** Reproduced live: a
`snapshot` call on an ordinary open tab came back as 51,506 characters /
623 lines and was rejected with an "exceeds maximum allowed tokens" error
(a prior run hit the same wall at ~62,000 chars / 780 lines). This is not
transient and not a `jobagent`/browsermcp bug — it's a hard cap the calling
host enforces on any single tool result, and LinkedIn's accessibility tree
is simply bigger than that cap on job pages, profile pages, search results,
and the feed — i.e. most of what this agent looks at.

What actually happens when it fails: the call errors instead of returning
JSON, but the *full* snapshot text is saved to a local file, and the error
names that file's path.

**Do this, not a blind retry:**

- **Don't retry `snapshot` unchanged.** The same page will hit the same
  limit every time — it isn't a timing fluke.
- **Read the saved file instead of the tool result.** Grep it for the
  element's visible text or role (e.g. `grep -i "easy apply"`) rather than
  reading the whole thing — that's almost always enough to recover the
  `ref` you needed. Read it in chunks (offset/limit) only when you
  genuinely need to scan structure, not just find one element.
- **If you have no shell access to that saved file** (a driver context that
  can't reach the local filesystem the snapshot was written to), `snapshot`
  is not usable on that page at all — this is a real capability gap, not a
  step to push through. Fall back to Claude in Chrome for that step if it's
  available (its `find`/`read_page` degrade more gracefully on large
  pages), or narrow what you're looking at first (scroll to the relevant
  section, close panels/modals you don't need) and retry — a smaller,
  scrolled state sometimes produces a small enough tree.
- **On CDP, use `browse dump --label <name>` instead.** It writes the same
  full page text to `data/traces/pages/` and returns only the path, the
  character count and the scroll offset — the capture the corpus wants,
  without the capture crossing the wire.
- **This is also why `trace log`'s `page_text`/`a11y` fields (see
  `trace-schema-v3.md`) can't be "call `read_page`/`snapshot` and paste the
  result inline."** The same size wall applies to logging it as to reading
  it — those fields have to be written to their own file on disk, the same
  way the host already does for a rejected `snapshot`, not carried through
  as a JSON string.
