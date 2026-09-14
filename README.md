# LinkedIn Job Agent

An agent that searches LinkedIn for jobs worth your time, applies to the strong
matches, and works the human side of a job search — recruiters, alumni, mutual
connections, referral asks, follow-ups when invites are accepted, and a nudge
when someone replies.

Two halves:

- **Skills** (`plugin/skills/`) — the judgement. Markdown instructions that
  drive your browser through the browser MCP tools.
- **`jobagent`** (`jobagent/`) — the memory. A SQLite store with a JSON CLI, so
  the agent never re-reads its own history, never applies twice, and never asks
  you the same form question again.

Splitting it this way is what keeps token use flat as the database grows: the
agent asks *"what should I do next?"* instead of re-reading everything it's done.

---

## Install

```powershell
cd C:\local_data\linkedinAgent
python -m jobagent init
```

Python 3.9+, no third-party dependencies.

Then install the plugin so the skills are available to Claude, or just point
Claude at this folder and name the skill you want.

Requirements at run time:

- Chrome, logged into LinkedIn, with a browser driver extension enabled and
  linkedin.com permitted
- the Claude desktop app running (the session drives your real browser)

### Browser drivers

Three are supported, in preference order: **CDP**, then **Claude in Chrome**,
then **Browser MCP**.

**CDP** drives your own Chrome directly via Python (no extension) — it's first
because it's the only driver whose page reads can't fail on size and the only
one that attaches a resume everywhere. It needs a five-minute one-time setup:
`config/cdp.setup.md`. **Claude in Chrome** is the fallback when CDP isn't
connected — it also has file upload and natural-language element finding.
**Browser MCP** (https://github.com/browsermcp/mcp) is the last resort, kept
around for portability: it works in Cursor, VS Code, Windsurf and Claude
Desktop, so the agent runs outside Cowork.

`plugin/references/browser-adapter.md` maps every action to all three drivers
and lists what degrades. Install steps: `config/cdp.setup.md` and
`config/browsermcp.setup.md`.

Only CDP can attach a real file. On the two MCP drivers the agent sends your
resume as a **link** instead. Set one, or it will refuse to send resume-bearing
messages rather than promise an attachment it can't add:

```bash
python -m jobagent profile set --json '{"resume_url":"https://<link-to-your-resume>"}'
```

---

## First run

```
Run linkedin-setup
```

It reads your LinkedIn profile and your resume, then asks — in batches of four —
everything an application form will ever want: notice period, current and
expected CTC, work authorisation, non-compete and confidentiality answers,
whether you have relatives at the company, languages, skills, plus what you
actually want (product vs service, MNC or not, headcount band, locations,
dealbreakers, blacklisted companies).

It stores all of it locally in `data/state.db`. Compensation and phone number
never leave your machine except into a form you approved.

## Daily

```
Run linkedin-daily-run
```

Order is deliberate — inbox, then follow-ups, then search and apply, then new
outreach. Replies come before new invites, always.

Or run a phase on its own: `linkedin-apply`, `linkedin-outreach`,
`linkedin-followup`, `linkedin-inbox`.

## Running it as an agent

`plugin/agents/linkedin-agent.md` defines a subagent that runs the whole cycle
in **its own context window** and returns a ~200-word summary. A full run scrapes
a lot of page text; this keeps all of it out of your main conversation.

```
Use the linkedin-agent subagent to run today's job search
```

It has a deliberately narrow toolset — browser, bash, read — and no Write or
Edit, so a run can never modify the project. It also has no way to ask you a
question mid-run, which is the point: instead of guessing at a screening
question it can't answer, it skips that application with
`--reason needs_human:<question>` and puts the question in its report. Answer it
once with `answers set` and it never comes up again.

To make it recur, create a **scheduled task** (not local cron, which dies with
the session) firing weekday mornings in your timezone, bound to this computer,
with a prompt naming the subagent and this folder. The run needs the machine
awake, the desktop app running, and Chrome logged into LinkedIn — it fails
loudly rather than queuing if not.

---

## What it does and doesn't do on its own

**Applies on its own** to Easy Apply jobs scoring ≥ 70 (configurable), using
only answers you've already given. A form question it can't confidently match
stops that application rather than guessing.

**Never sends a message on its own.** Every connection note, recruiter pitch and
referral ask is drafted, queued, and shown to you. You approve, edit, or reject.
Flip that with `config set --json '{"outreach_requires_approval":false}'` — but
consider that a referral ask carries someone else's name.

**Never states your salary to a recruiter** without you confirming it in the
conversation, even though the number is in the store.

Defaults: 10 applications, 15 invites, 10 messages per day; 80 invites a week;
25–70 seconds between actions.

---

## Layout

```
jobagent/          state + logic
  db.py            schema and migrations
  cli.py           the JSON CLI the skills call
  matching.py      job relevance scoring (explainable, 0-100)
  answers.py       the answer bank + fuzzy question resolution
  templates.py     outreach copy, with LinkedIn's character limits
plugin/
  skills/          linkedin-setup · -apply · -outreach · -followup · -inbox · -daily-run
  references/      browser-playbook.md (LinkedIn URLs, flows, abort signals)
                   cli.md (command reference)
config/
  questions.yaml   the onboarding battery, grouped into ask-in-fours rounds
data/              state.db, your resume — gitignored
tests/             end-to-end smoke test
```

## Useful commands

```bash
python -m jobagent report daily        # today: applied, sent, replies, backlog
python -m jobagent report pipeline     # lifetime counts by status
python -m jobagent report followups    # what's owed a message
python -m jobagent outreach pending    # the approval queue
python -m jobagent run quota           # what's left today
python -m jobagent answers missing     # gaps that will stall an application
python -m jobagent company blacklist --name "Bad Corp"
```

Full reference: `plugin/references/cli.md`.

## Training data for a future local model

Every run already reads pages and finds elements through a large model. Since
LinkedIn's application flow is a small, repeating set of screens (Easy Apply
modal, screening questions, connect/message dialogs), the agent also logs
those (page state, instruction, action) steps as it goes — a free byproduct,
not a new workflow — to `data/traces/<day>/<session>.jsonl` (one folder per day, one file per run). The idea: once there's enough
of a corpus, fine-tune a small vision model on it that can do the mechanical
"find this button / verify no error" work locally, without a large model in
the loop for every click.

Steps are grouped by `task_id` (one job application, one outreach send, one
reply) so a whole episode can be replayed in order later — not just used as
disconnected single actions — opened with a `_task_start` record naming the
instruction that authorized the episode (a request, the daily run, or a
skill default) and closed with a `_task_end` giving its final outcome. Each
step also carries a screenshot by default, plus the click point, target
bounding box, viewport size, and the other candidate elements on screen —
that's what makes a step trainable as a click-prediction example rather than
just a record of what happened. When a value came from `jobagent`'s own
store (an answer lookup, a job score) rather than off the page, that lookup
is logged too — including the times it was wrong and got overridden, which
turn out to be the most useful examples in the corpus.

```bash
python -m jobagent trace stats               # corpus size, step spread, retries, grounding_coverage (% of steps with real coordinates), retrieval_coverage (lookups + overrides)
python -m jobagent trace task --task-id X     # one episode's steps, in order
```

Nothing reads these back today — they don't affect scoring, matching, or any
decision the agent makes. See `plugin/references/browser-playbook.md` §9 for
the full schema and the PII rule (no salary figures, phone numbers, or PII
in logged values, including inside the candidate-elements list).
`data/traces/` is gitignored along with the rest of `data/`.

## Tests

```bash
python -m unittest discover tests -v
```

---

## Honest limits

**LinkedIn's terms prohibit automated access.** This drives your own logged-in
browser at human pace with human-approved messages, which is the mildest form of
it, but the risk of a temporary restriction is real and it is yours. The
conservative defaults exist for that reason — raising them raises the risk
roughly in proportion.

**Selectors rot.** LinkedIn changes its DOM often. The playbook deliberately
navigates by visible text rather than CSS classes, but a redesign will still
break flows; the fix is usually a paragraph in `browser-playbook.md`, not code.

**Volume is not the strategy.** Twelve applications with a referral behind three
of them beats sixty cold ones. The caps and the mandatory
`one_specific_reason` on every message are load-bearing, not decoration.
