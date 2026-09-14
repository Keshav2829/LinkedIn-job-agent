# LinkedIn browser playbook

Everything the skills need to know about driving LinkedIn through a browser
driver. LinkedIn changes its DOM constantly, so **never depend on CSS class
names**. Find elements by their visible text / accessible name — `browse find`
on CDP, `mcp__claude-in-chrome__find` or `read_page` on Claude in Chrome — then
click by ref.

---

> **Three drivers, CDP first.** This file names Claude in Chrome tools for
> readability. Preference order is **CDP, then Claude in Chrome, then Browser
> MCP** — translate every call through the verb table in `browser-adapter.md`,
> which gives the CDP (`jobagent browse …`) and Browser MCP form of each one.
> Read its degradations section too: file upload and natural-language `find`
> don't exist on Browser MCP.

## 0. Session rules — read before every run

- **One tab.** Create one tab at the start (`tabs_create_mcp`), reuse it, close
  it at the end. Multiple tabs racing on linkedin.com trips their bot heuristics.
- **Pace yourself.** Wait a randomised 25–70 s between meaningful actions
  (an application, an invite, a message). `jobagent outreach next` returns the
  configured `pacing_seconds` range. Bursts are the single biggest cause of
  account restriction.
- **Never trigger a native dialog.** Do not click *Discard*, *Delete*,
  *Withdraw*, or *Report*. A `confirm()` freezes the extension and the run dies.
- **Read before you click.** `read_page` after every navigation. If the page
  isn't what you expected, stop and report — do not click blindly.

### Abort signals

Stop the entire run immediately, mark the run ended with a note, and tell the
user, if you see any of:

| Signal | Meaning |
|---|---|
| "Let's do a quick security check" / captcha / puzzle | Bot challenge — stop, hand the browser back |
| "You've reached the weekly invitation limit" | Invite ceiling — stop inviting for the week |
| "You're approaching the monthly limit" (search) | Commercial-use limit — stop searching |
| Any "restricted"/"temporarily suspended" banner | Account action — stop everything, do not retry |
| Login / 2FA screen | Session expired — ask the user to log in, then resume |

Never attempt to solve a captcha, and never retry a blocked action more than
once.

**Log it before you stop.** An abort signal is one of the highest-value
frames in the whole corpus — it's what a local model needs to learn to
*recognize and stop for*, not click through — and as of this writing the
corpus has zero of them despite five documented signals above. See §9,
"Failure and abort trajectories," for the exact steps. Logging it costs one
extra call and never delays the abort itself: detect the signal, stop the
browser action, log it, then end the run.

---

## 1. URLs

### Job search
```
https://www.linkedin.com/jobs/search/?keywords={kw}&location={loc}&f_AL=true&f_TPR=r86400&sortBy=DD
```
| Param | Values |
|---|---|
| `f_AL=true` | Easy Apply only |
| `f_TPR` | `r86400` past 24h · `r604800` past week · `r2592000` past month |
| `f_WT` | `1` on-site · `2` remote · `3` hybrid (comma-joined) |
| `f_E` | `1` intern · `2` entry · `3` associate · `4` mid-senior · `5` director · `6` exec |
| `f_JT` | `F` full-time · `C` contract · `P` part-time |
| `f_C` | company id(s) |
| `sortBy` | `DD` newest · `R` relevance — **use `R`** |
| `start` | pagination offset, steps of 25 (`start=25` is page 2, `start=50` is page 3, …) |

**Use `sortBy=R`, not `DD`.** Verified 29 Aug 2026: newest-first on a broad
keyword returns almost entirely staffing-agency reposts — "AI Engineer" in
India surfaced Power BI, .NET, AEM and frontend-intern roles in its top seven.
Relevance-sorted, the same filters returned genuine LLM/agent roles.

**Keyword breadth matters more than the time window.** Broad titles ("AI
Engineer") match on any token; specific stacks ("LLM AI Agents Engineer")
return far better sets. Prefer a 30-day window on a sharp keyword over a
24-hour window on a vague one.

**Reading the result count:** the number in the page *title* — `(12) AI
Engineer Jobs` — is LinkedIn's unread-notification badge, not a result count.
Read the "N results" element in the page body instead.

**Getting past the first page.** The results pane is a nested scroll
container, not the page — `browse scroll` wheels whatever the mouse happens
to be over, which is not reliably that pane, so it's an unreliable way to
load more cards. `start=N` is not: it's a real navigation, so `browse open`
on the same search URL with `start` incremented by 25 lands on a fresh page
that `browse outline` / `browse find` / `browse text` read exactly like the
first one. Prefer it over scrolling. See linkedin-apply's search step for how
many pages to pull per search.

Individual posting: `https://www.linkedin.com/jobs/view/{jobId}/`
In the search list the selected card sets `?currentJobId={jobId}` — that value
is the `--job-id` used everywhere in `jobagent`.

### People search
```
https://www.linkedin.com/search/results/people/?keywords={kw}&currentCompany=%5B%22{companyId}%22%5D&network=%5B%22S%22%5D
```
| Param | Values |
|---|---|
| `network` | `%5B%22F%22%5D` 1st · `%22S%22` 2nd · `%22O%22` 3rd+ (JSON array, URL-encoded) |
| `currentCompany` | encoded JSON array of company ids |
| `schoolFilter` | encoded JSON array of school ids — this is the alumni filter |
| `geoUrn` | location |

Recruiter keywords that work well: `recruiter`, `talent acquisition`,
`technical recruiter`, `HR`, `hiring manager`, `engineering manager`.

Alumni of your school at a company: your school page →
`https://www.linkedin.com/school/{school-slug}/people/` → filter by company.

### Network & messaging
| What | URL |
|---|---|
| Invitations I sent | `https://www.linkedin.com/mynetwork/invitation-manager/sent/` |
| Invitations received | `https://www.linkedin.com/mynetwork/invitation-manager/` |
| My connections | `https://www.linkedin.com/mynetwork/invite-connect/connections/` |
| Unread messages | `https://www.linkedin.com/messaging/?filter=unread` |
| Notifications | `https://www.linkedin.com/notifications/` |
| My profile | `https://www.linkedin.com/in/me/` |

---

## 2. Easy Apply flow

1. Open the job. Confirm the button reads **"Easy Apply"** — if it reads
   "Apply" it opens an external ATS. Record it with
   `jobagent job skip --job-id X --reason external_ats` unless the user has
   opted into external applications. **Log it** — §9's failure-trajectory
   pattern: a `detect_external_ats` step, then close the episode with
   `_task_end` `outcome: "skipped:external_ats"`. This is a routine, expected
   outcome, not an error — logging it the same way as a success is what
   teaches a model the difference between "Apply" and "Easy Apply" buttons.
2. Click **Easy Apply**. A modal opens with a step indicator.
3. Steps, in the usual order:
   - **Contact info** — name, email, phone country code, phone. Prefilled;
     verify against the answer bank.
   - **Resume** — pick the most recent uploaded resume if it matches
     `profile.resume_name`, else upload with `file_upload`.
   - **Additional questions** — the screening battery. See §3.
   - **Work authorization / EEO** — answer from the bank; EEO questions
     default to whatever the user chose during setup.
   - **Review** — read the whole modal back before submitting.
4. Buttons advance in this order: **Next** → **Continue to next step** →
   **Review your application** → **Submit application**.
5. **Uncheck "Follow {company}"** on the review step unless the user opted in.
6. Success looks like a dialog: *"Your application was sent to {company}"*.
   Only after seeing it, call `jobagent job applied`.
7. Close the confirmation with its **Not now** / ✕ button (LinkedIn upsells a
   profile update on this screen — decline it).

**Harvest the hiring team while you're here.** The job page carries a "Meet the
hiring team" panel and a "People you can reach out to" panel, naming the job
poster and employees, with degree badges and alumni flags visible. These are
the best outreach targets you will find — they are attached to *this* req.
`contact upsert` them during the application, so the outreach phase doesn't
have to run a people search to rediscover them.

If a step cannot be completed (a required question with no stored answer and
no safe default), click **✕** then **Discard**? — **no**: instead leave the
modal by pressing Escape, and record
`jobagent job skip --job-id X --reason needs_human:<question>`. Never submit a
partially-guessed application. **Log it** the same way as the external-ATS
case above — close the episode with `_task_end`
`outcome: "skipped:needs_human:<question>"` (the literal question text, so
the corpus shows exactly what stumped it) rather than leaving the episode
without an ending record.

---

### Filling fields — verified the hard way, 29 Aug 2026

**`form_input` alone does not work on Easy Apply screening fields.** It sets the
DOM value, but LinkedIn's React form never sees an input event: validation
doesn't re-run, stale "Invalid input" errors persist, and the value may not
reach form state at all. A submission can go through with fields the employer
receives as empty.

Always fill a screening field like this:

1. `computer` → `triple_click` on the field's ref (selects existing content)
2. `computer` → `type` the value
3. **Blur it** by clicking the next field — validation only runs on blur
4. **Log it right now** — `trace log` for `fill_screening_field`, with the
   screenshot you just took to check for a red error. Don't hold it and
   write it later; see "log each field as you go" in §3.

Then screenshot or `zoom` on the field group and confirm no red "Invalid input"
remains *before* clicking Review. `form_input` is still fine for `<select>`
dropdowns and the contact-info step.

**A field that comes back red is not a dead end to fix silently — it's a
retry chain, and retry chains are the single most useful pattern missing
from the corpus today (zero `attempt > 1` records so far).** Log the failed
attempt (`fill_screening_field`, `attempt: 1`, `outcome: "error_shown"`,
screenshot of the red state) and the corrected one (same `task_id`/`step`,
`attempt: 2`, `outcome: "success"`) as two separate records. That pair is
what teaches a local model to notice its own mistake and correct it, instead
of only ever seeing the clean first-try version.

### Compensation fields are numeric-only

Indian postings ask **CCTC** (current) and **ECTC** (expected) "Per Annum".
These validate as plain integers in rupees — `32 LPA`, `45-50 LPA` and
`3.2 million` are all rejected; `3200000` is accepted. Consequences:

- A **range is impossible.** One number must be committed.
- **Fill it yourself from the stored `current_ctc` / `expected_ctc` — do not
  stop to ask, and do not skip with `needs_human`.** The user already set
  these values once at setup; that is the confirmation. Convert units at
  fill time (`38 LPA` → `3800000`).
- **If the stored value is a range** (e.g. "35-40 LPA"), use the **upper
  bound** (`4000000` here) and proceed — don't surface it for a
  per-application decision.
- Only fall back to `needs_human:<label>` when the field is asking for
  something `current_ctc`/`expected_ctc` doesn't cover at all (e.g. a
  distinct "minimum acceptable CTC" figure with nothing stored for it).

## 3. Screening questions

For each question in the "Additional questions" step:

```bash
python -m jobagent answers resolve --question "<the exact label>" --options "Yes|No"
```

- `found: true` and `confidence >= 0.6` → fill with the returned `answer`.
- `found: true` but `confidence < 0.6` → fill it, and add the question to the
  `--screening` array passed to `job applied` so the user can audit it later.
- `found: false` → **do not guess.** Two cases:
  - The run is interactive: ask the user, then
    `jobagent answers set --key <new_key> --value "<answer>" --question "<label>" --alias "<label>"`
    so it's never asked again.
  - The run is unattended: skip the job with
    `--reason needs_human:<label>` and surface it in the daily report.

Numeric "how many years of experience with X" questions: if X is in the
profile's skills list, answer with the years recorded for it; otherwise resolve
against `total_experience_years` only when the skill is a core one, else ask.

**Log each field as you go — don't wait for the end of the episode.** A
screening battery can run to a dozen questions; log `fill_screening_field`
(step 4 of the fill procedure above) right after each one is filled and
blurred, not as a batch once the whole modal is done or at `_task_end`.
Batching loses the pairing between a field's screenshot/coords and the
retry/override records that belong to it, and if the run aborts partway
through the form (a validation error you can't resolve, an abort signal),
every field you filled before the crash goes unlogged instead of standing
as real steps in the trace.

---

## 4. Sending a connection request

1. Open the profile URL.
2. Find the **Connect** button. If it isn't visible, it's behind **More** →
   **Connect**. If the primary button is **Follow** and Connect is absent, the
   person only accepts follows — record `contact mark --relation declined`
   with a note and move on. Log the `find_connect_button` step with
   `outcome: "na"` (nothing is broken — Connect genuinely isn't offered), then
   close the episode with `_task_end` `outcome: "skipped:follow_only"`.
3. Click **Add a note**. LinkedIn caps the note at **300 characters**;
   `jobagent` caps drafts at 280.
4. Paste the approved body with `form_input` into the note textarea.
5. Click **Send**. Then `jobagent outreach sent --id N`.

If a modal says *"You've reached the weekly invitation limit"* — abort per §0
(and log it per §0/§9 before you do).

Free accounts get a limited number of *personalised* invites per month. When
LinkedIn shows the "You have X personalized invites left" upsell, prefer
sending remaining invites **without** a note rather than burning the quota,
and tell the user. This is a recoverable degradation, not an abort — log the
blocked note attempt (`fill_connect_note`, `outcome: "blocked"`, a note
naming the upsell, **plus a screenshot and the coords/element you clicked** —
a block with no grounding on it is exactly as untrainable as a success with
no grounding on it), then the successful no-note send as its own fully
grounded step (its own coords/element/screenshot — it's a different button
in a different place, not a continuation of the blocked click), and close
the episode with `_task_end` `outcome: "sent"` and
`result: {"note_included": false}` — not a made-up `outcome` like
`"sent_no_note"`; the detail belongs in `result`. See §9, "Every step in a
recovery sequence is its own record," for why this matters beyond this one
case.

---

## 5. Sending a message (with resume attached)

1. Open the profile → **Message**, or go straight to the thread in
   `/messaging/`.
2. Type the body (`browse fill` on CDP, `form_input` on Claude in Chrome) into
   the message box.
3. Attach the resume: click the **paperclip / Attach a file** control, then use
   `browse upload` on CDP or `mcp__claude-in-chrome__file_upload` on Claude in
   Chrome with the resume path.
4. Verify the attachment chip shows the right filename **before** clicking Send.
5. Click **Send**, then `jobagent outreach sent --id N`.

Only 1st-degree connections can be messaged for free. For a 2nd-degree contact
with no InMail credits, the invite note *is* the message — keep it to 280 chars.

---

## 6. Checking whether invites were accepted

Go to `/mynetwork/invitation-manager/sent/`. Everyone still listed there is
**pending**. So:

```bash
python -m jobagent contact pending-invites          # who we think is pending
```

Compare. Anyone in `pending-invites` who is **no longer** on the sent list has
either accepted or withdrawn/expired. Confirm by opening their profile: a
**Message** button and "1st" degree badge means accepted.

```bash
python -m jobagent contact mark --url <profile> --relation connected
```

Withdraw nothing automatically — pending invites older than ~3 weeks can be
withdrawn manually by the user if they want the quota back.

---

## 7. Reading replies

`/messaging/?filter=unread` lists unread threads. For each:

1. Read the thread. Capture the newest inbound message only.
2. `jobagent reply add --contact-url <profile> --snippet "<first 300 chars>" --sentiment positive|neutral|negative`
   (duplicate snippets are ignored, so re-scanning is safe).
3. **Do not auto-reply.** Draft a response and queue it for approval as
   `--kind thanks` or `--kind followup`.

Auto-responders ("I'm on leave", "we received your application") should be
tagged `--sentiment auto` so they don't count as real replies.

---

## 8. Reading your own profile (setup only)

`https://www.linkedin.com/in/me/` → `get_page_text`. Pull: name, headline,
location, current company + title, full experience list with dates, education
(school names matter — they drive alumni search), skills, and the public
profile URL. Then `Details → Skills` (`/details/skills/`) for the full list,
which the main page truncates.

---

## 9. Logging steps for a future local model

The long-run goal is a small vision model, fine-tuned on LinkedIn's own
screens, that can take over the mechanical parts of this playbook (find the
button, verify no red error, click it) without a call out to a large model
each time. That model doesn't exist yet — this section is about generating
the training data for it as a free byproduct of runs you're doing anyway.

**Cut over to schema 3 on 5 Sep 2026.** Schema 2 made records *valid*; it
did not make them *learnable* — a clean, fully-conformant 55-record day
still measured **0.0% usable training examples**
(`tools/audit_training_yield.py`), because the record never stored what the
model actually saw (no DOM, no accessibility tree, no scroll position) and
its action space didn't cover most of what a run actually does (scrolls,
`find`, `read_page` left no trace at all). `trace-schema-v3.md` is the full
spec; this section is the day-to-day summary of what changed in the call
shape.

**The flat v2 fields are retired.** `action_type`, and top-level
`coords`/`bbox`/`element`/`candidates`/`url`, are rejected outright on any
non-reserved step — `trace log` names the replacement in the error rather
than silently accepting the old shape. Two new blocks replace them:

- **`action`** — what you did. `{"kind": "click"|"type"|"select"|"key"|
  "scroll"|"navigate"|"read"|"tool_call"|"ask_human"|"stop", ...}` plus
  `ref`/`element`/`coords`/`bbox`/`candidates`/`value` as before, now nested
  under it. `verify`/`na` from v2 are gone — a look that doesn't act (a
  `find`, `read_page`, `get_page_text` call) is a `read`, and it now gets its
  own record instead of leaving no trace. A `jobagent` CLI lookup
  (`answers resolve`, `job.score`, ...) is `tool_call`, with
  `tool: {name, args, result}`.
- **`observation`** — the page state the action was taken against:
  `url`, `title`, `screenshot`, `screenshot_dims`, `viewport`, `scroll`,
  `page_dims`, `a11y` (path to the `read_page` output for this state,
  saved to its own file — see "why these can't be inlined" below),
  `page_text` (path to `get_page_text` output, same reason). Required on
  every non-reserved step except `tool_call` — a lookup has no page state to
  report — and as the closing state on `_task_end`, so the last transition
  in the episode isn't a silent gap.

**`trace log` rejects rather than warns on the things that make a record
untrainable** — none of this touches app behavior; a rejected call still
means the actual apply/outreach/inbox flow continues, only the log line is
refused, so fix the record and log it again (or move on):

- Every non-reserved step needs an `action` block with a valid `kind`, and
  (except reserved steps) a `plan_step` — the current sub-goal this action
  serves, so a planner is trainable above the single-action executor, not
  just the executor itself.
- Every non-reserved, non-`tool_call` step needs an `observation` block with
  `a11y` and `page_text` set.
- Whenever `action.coords` or `action.bbox` is set, `observation.viewport`,
  `observation.scroll` and `observation.page_dims` must all be present too —
  without `scroll`, a `bbox` names a place on the screen, not on the page.
- A `click`/`type`/`select` step **must** carry a screenshot (`--image-b64`)
  or the call fails, and `observation.screenshot_dims` must agree with
  `observation.viewport` (or `scale` must say why it doesn't) — this is what
  stops the v2 failure mode where recorded coords silently lived in a
  different frame than the stored image.
- `action.kind` `tool_call` requires `action.tool.name` and `action.tool.args`.
- `driver` is **required** on any step whose `action.kind` touches the
  browser (everything except `tool_call`/`ask_human`/`stop`), and is
  normalized to `cdp`, `claude_in_chrome`, `browsermcp`, or
  `claude_browser_pane` (the Claude desktop app's in-app browser pane) —
  `claude-in-chrome` and similar variants are accepted and rewritten, not
  rejected.
- `outcome` on an ordinary step is restricted to `success|error_shown|
  blocked|na`, and must be a string. A richer payload (a confirmation
  string, an application id, a match score) goes in a new `result` object
  instead — `outcome` is a label, `result` is detail.
- `_task_start` carries `source` (`user_request|scheduled_daily_run|
  skill_default`), never `outcome`, and no `action`/`observation` — it's
  the one record that only names what authorized the episode.
- `_task_end`'s `outcome` is `applied|sent|not_applied|error|aborted` or
  `skipped:<reason>` (same rule as above), carries no `action`, and now
  requires a final `observation` — the last page state before the episode
  closes.

**Why `a11y`/`page_text` are paths, not inline text.** `read_page`/`snapshot`
output on a real LinkedIn page routinely runs 50,000+ characters — reliably
past the size a single tool call or a single trace record can carry (see
`browser-adapter.md`'s "`snapshot` output size" note, reproduced live on
5 Sep 2026: an ordinary tab's snapshot came back at 51,506 characters and
was rejected outright). Save the `read_page`/`get_page_text` output to a
file under `data/traces/pages/` yourself (e.g.
`pages/<day>-<hash>.a11y.yaml` / `.txt`) and put that path in
`observation.a11y`/`observation.page_text` — `trace log` checks that the
fields are set, not that it can read the files itself. Reuse the same pair
of files across consecutive steps that share a page state instead of
re-saving duplicates (`observation.content_hash`, optional, is for exactly
this — hash `a11y` + `url` and skip the write if it's unchanged).

**Attach a screenshot (`--image-b64`) to every logged step, not just
verification steps.** A coordinate or bounding box with no image behind it
can't train a vision model — it's a number with nothing to show it applies
to. You're already taking a screenshot to see the page before most actions
(`computer screenshot`); reuse that same capture for the log call instead of
skipping it to save a step. **Store it at native resolution, unscaled** — a
downscaled or JPEG-compressed capture makes LinkedIn's small text
unreadable, which is exactly the thing a grounding model most needs to read;
if you keep a smaller copy for your own context-window reasons, that's a
derived artifact, not what goes in the record. `screenshot_dims` is computed
for you from a PNG or JPEG; for any other format, set it yourself. The one
exception to attaching a screenshot at all is a step where you only ever
read text (`get_page_text`, `read_page` with no visual check) and never
looked at a screenshot yourself — don't take one solely to satisfy the
logger, and log that step's `action.kind` as `read`, not `click`/`type`/
`select`, so the screenshot requirement doesn't apply to it. Any common
image format is fine; the CLI sniffs it, it doesn't assume PNG.

**`trace_schema` and `persona_id` are filled in for you.** Every record
written now carries `trace_schema: 3`; anything without that field predates
this fix — schema 2 shows up as `2` and anything before that as
`v1_legacy` in `trace stats`' `by_schema` breakdown. **v2's grounding is
dead, not just thinner** — recorded `coords` lived in a 1568×744 frame
while the stored images were 784×372, and `scroll` was never recorded, so
there's no way to re-anchor a v2 `bbox` onto a page after the fact. Treat
v2 records as corpus for episode structure and `retrieval`/override
examples, never as coordinate-supervision labels; don't mix schemas in one
training pull. `persona_id` is a stable per-machine tag generated once and
cached, so a corpus that later merges more than one account's runs can
still be filtered or weighted by whose answers (and whose CTC, phone
number, etc.) actually produced a step.

**Backfilling an existing step doesn't duplicate it anymore.** Pass
`"backfill": true` with the same `task_id` and `step` as an already-logged
record, and `trace log` updates that record in place (setting
`backfilled_at`) instead of appending a near-duplicate line. If no matching
record exists, it logs fresh and adds `backfill_target_not_found: true` so
you can tell a typo'd `task_id`/`step` from a genuinely new step.

```bash
python -m jobagent trace log --json '{
  "skill": "linkedin-apply",
  "step": "find_easy_apply_button",
  "task_id": "<see table below>",
  "attempt": 1,
  "goal": "Apply to strong-match Easy Apply roles from today's search, capped at the daily quota",
  "plan_step": "Open the Easy Apply modal for this posting",
  "instruction": "<the exact instruction you gave find, or what you were verifying>",
  "rationale": "<why this action, one line — optional>",
  "observation": {
    "url": "<current URL>", "title": "<page title>",
    "screenshot": "<set for you by --image-b64>",
    "viewport": {"width": 1280, "height": 800},
    "scroll": {"x": 0, "y": 0},
    "page_dims": {"width": 1280, "height": 3400},
    "a11y": "pages/<day>-<hash>.a11y.yaml",
    "page_text": "pages/<day>-<hash>.txt"
  },
  "action": {
    "kind": "click",
    "element": {"role": "button", "name": "Easy Apply", "ref": "<the ref find/read_page returned>"},
    "coords": {"x": 940, "y": 210},
    "bbox": {"x": 900, "y": 190, "w": 90, "h": 36},
    "candidates": [{"role": "button", "name": "Save", "bbox": {"x": 700, "y": 190, "w": 60, "h": 36}}],
    "expected_effect": "the Easy Apply modal opens over the job page"
  },
  "outcome": "success|error_shown|blocked|na",
  "driver": "cdp|claude_in_chrome|browsermcp|claude_browser_pane",
  "confidence": "<answers.resolve confidence, when this step is filling a screening field>",
  "retrieval": [{
    "source": "answers.resolve",
    "query": {"question": "How many years of work experience do you have with Docker Products?"},
    "result": {"found": true, "key": "total_experience_years", "value": "4", "confidence": 0.95},
    "accepted": false,
    "override_reason": "Docker isn'\''t in the stored skills list; the fuzzy matcher wrongly fell back to total_experience_years. Asked the user directly instead."
  }]
}' --image-b64 "<base64 screenshot>"
```

Field notes:

- **`goal`/`plan_step`/`rationale`** — the hierarchy above the single action.
  `goal` is constant for the whole episode (copy it from `_task_start`'s
  `instruction`); `plan_step` is the current sub-goal and changes a few times
  per episode; `rationale` is optional, one line on why this action. Training
  `instruction → action` alone only ever produces an executor that still
  needs a large model to tell it what to do — `plan_step` is what makes the
  planning layer trainable too.
- **`action.coords`** — the actual click point, in the raw pixel coordinates
  you clicked at (or the field you typed into), in `observation.viewport`'s
  frame. **`action.bbox`** — the target element's bounding box, if
  `read_page`/`find` gave you one. Give `coords` or `bbox` whenever the step
  was a click/type/select — without one of them, the step can record *that*
  something happened but not *where*. `action.coords_norm`/`bbox_norm`
  (0–1 scaled) are computed for you when `observation.viewport` is present —
  don't set them yourself.
- **`action.element`** — the accessibility node you acted on (role,
  accessible name, ref). Cheap to include since `find`/`read_page` already
  gave it to you, and it's a cleaner label than free text.
- **`action.candidates`** — the *other* interactive elements you saw and
  didn't pick (e.g. "Follow" sitting next to "Connect"). Optional, but this
  is what teaches a grounding model to discriminate between similar buttons
  instead of only ever seeing the right answer with nothing to contrast it
  against.
- **`action.expected_effect`** — one short line, written *before* acting,
  on what should happen next ("the Easy Apply modal opens"). Cheap, and
  comparing it to the next record's `observation` is how a failure gets
  auto-labelled without a second annotation pass.
- **`attempt`** — 1 for a clean first try; bump it if you had to retry after
  a miss, a stale element, or a validation error. Defaults to 1 if omitted.
  A retry chain (attempt 1 fails, attempt 2 succeeds, same `step`/`task_id`)
  is some of the most useful data you can log — it's what would teach a
  local model to notice and recover from its own mistakes.
- **`confidence`** — when the step is filling an answer from `answers
  resolve`, pass its `confidence` through. Lets training later filter out
  the answers the app itself already flags as uncertain.
- **`driver`** — always set it on a browser-facing action. `cdp`,
  `claude_in_chrome`, `browsermcp` and `claude_browser_pane` use different
  coordinate/action
  semantics; mixing them unlabeled would corrupt any coordinate-based
  training. Not required for `tool_call`/`ask_human`/`stop` — those aren't
  tied to a browser driver.
- **`retrieval`** — how you actually got the value you're filling in, when it
  came from `jobagent`'s own store rather than being read off the page or
  typed from the instruction alone. One entry per lookup you made:
  `source` (the CLI call, e.g. `"answers.resolve"`, `"job.score"`,
  `"contact.seen"`), `query` (the args you passed), `result` (what it
  returned, verbatim), and `accepted` (did you actually use that value
  as-is?). **When you didn't** — because the match was wrong, low-confidence,
  or you used your own judgment over it — set `accepted: false` and fill
  `override_reason` with what you actually did instead and why. This is not
  a corner case to skip: overrides are the highest-value examples in the
  whole corpus, because they're the ones that show a local model *when not
  to trust its own retrieval*. The example above is a real one — the answer
  bank fuzzy-matched "years with Docker Products" to `total_experience_years`
  at 0.95 confidence, which was wrong, and got caught and corrected during a
  live run. Log it exactly like that when it happens again, don't just fix it
  silently.

**`step` naming is a closed vocabulary, and it is now enforced.** This
section used to call it a "suggested starting set", and by the 4 Sep audit
five invented names had leaked into the corpus — `select_resume`,
`view_job_posting`, `click_easy_apply`, `fill_screening_questions`,
`review_application` — each splitting examples off a canonical step that
already existed. `trace log` now rejects an unknown `step` (with a
near-match suggestion) and rewrites those five to their canonical names.
`skill` is closed the same way, and `linkedin-apply` `task_id`s that arrive
as a bare number or a `/jobs/view/…` URL are normalized to `job_<job_id>`.

```bash
python -m jobagent trace steps    # the live vocabulary, plus the aliases
```

The set, by phase:

| Phase | Steps |
|---|---|
| session (`skill: linkedin-daily-run`, `task_id: run_<run_id>`) | `capture_user_intent`, `open_run`, `open_linkedin`, `verify_session`, `open_jobs_tab` |
| search (`task_id: search_<slug>`) | `type_search_query`, `apply_filter`, `search_jobs`, `paginate_results`, `screen_job_candidate`, `open_job_card` |
| one job (`task_id: job_<job_id>`) | `check_job_seen`, `open_job`, `read_job_description`, `score_job`, `record_score_decision`, `detect_external_ats`, `find_easy_apply_button`, `verify_no_error`, `upload_resume`, `fill_screening_field`, `select_work_auth`, `harvest_hiring_team`, `click_review`, `submit_application`, `close_confirmation_dialog` |
| outreach / follow-up | `find_connect_button`, `fill_connect_note`, `send_invite`, `open_message_thread`, `fill_message_body`, `attach_resume`, `send_message` |
| inbox / setup | `read_reply`, `read_own_profile` |
| any task, any time | `detect_abort_signal` |

`detect_abort_signal` and `detect_external_ats` exist specifically for
failure and abort trajectories — see below. Don't invent a new name per
failure; a captcha and a restriction banner are both `detect_abort_signal`
with a different `notes` string, not two different steps.

**The search phase and the scoring decision are the ones that go missing.**
The first three days logged 154 records and not one came from a search, and
`score_job` — the lookup that gates every application — had zero records
across 31 episodes. An episode, in that corpus, began at "the Easy Apply
button is on screen": a model trained on it could fill a form and could not
find a job to fill it for. Everything from the user's request through
`record_score_decision` is as much a step as clicking Submit.
`data-generation-checklist.md` has the per-step field lists, the
three-level task model (session → search → job), and a flow table to walk
after a run; `trace stats` now reports `never_logged_steps` and
`off_vocabulary_steps` so a missing phase shows up without a hand audit.

**Logging can be switched off for a run** — `jobagent trace disable` (or
`config.trace_logging: false`) makes `trace log` a reported no-op that
writes nothing and returns `{"ok": true, "skipped": …}`. Keep calling
`trace log` unconditionally; don't branch on config per step, and don't
treat the skipped response as an error. It defaults to on.

**Always set `task_id`** to whatever identifies the one episode this step
belongs to — it's what lets the steps of a single application be replayed in
order later, instead of sitting in the log as unconnected single actions.
`step_index` is assigned for you (count of prior steps with the same
`task_id`), so don't pass it yourself. Use, per skill:

| Skill | `task_id` |
|---|---|
| `linkedin-daily-run` | `run_<run_id>` — the session itself, from the user's request to the final report |
| `linkedin-apply` (search) | `search_<slug>` — one per distinct keyword/filter combination, dead ends included |
| `linkedin-apply` (one job) | `job_<job_id>` (e.g. `job_4021553311`) — opened at the dedupe check, **not** at the Easy Apply button, so a job rejected on score still leaves a complete episode |
| `linkedin-outreach` | the contact's `profile_url` |
| `linkedin-followup` | the contact's `profile_url` |
| `linkedin-inbox` | the thread URL |
| `linkedin-setup` | a constant like `setup` (one episode, once) |

These nest: a session task opens one or more search tasks, and each search
task hands off to one job task per card it opens. Steps never share a
`task_id` across levels.

Log every step of one task under the *same* `task_id` from open to
close/submit/skip, even if other tasks' steps land in between in the file —
`trace task --task-id X` groups and orders them regardless of interleaving.

**Open every task with one `_task_start` record** — the reserved `step` name
for what actually authorized this episode. Every other record in the task
carries a micro-instruction ("find the Easy Apply button"); this is the one
place the *governing* instruction lives — the request or policy that made
this episode happen at all, which a model would otherwise have no way to
recover from the steps alone:

```bash
python -m jobagent trace log --json '{
  "skill": "linkedin-apply", "step": "_task_start", "task_id": "job_4456855664",
  "goal": "User-directed test run: apply to this specific queued job (score 61, below the 70 auto-apply bar) and reach out to its hiring team",
  "instruction": "User-directed test run: apply to this specific queued job (score 61, below the 70 auto-apply bar) and reach out to its hiring team",
  "source": "user_request"
}'
```

`instruction` should be specific enough that someone reading only this one
record understands why the episode happened — not just "apply to job" but
what made *this* job get applied to (auto-apply threshold clear? explicit
user pick? a daily-run phase acting on the queue?). Copy the same text into
`goal` — every other record in the episode reuses it verbatim, so a planner
has the constant it's supposed to condition on. `source` is one of
`user_request` (a person in the conversation asked for this one directly),
`scheduled_daily_run` (the normal `linkedin-daily-run` cycle picked it up
off the queue under standing config), or `skill_default` (a skill's own
built-in behavior, not a specific request). This has to be the *first*
record logged for a `task_id` — logging it after other steps already exist
defeats the point, and `trace log` will warn if `step_index` isn't 0.
`_task_start` carries no `action`/`observation` of its own.

**Close every task with one `_task_end` record** — the reserved `step` name
for an episode-level summary, once the task is fully done (submitted,
skipped, sent, or aborted). Unlike every other reserved-step rule,
`_task_end` **does** require an `observation` — the last page state before
the episode closes, so that final transition isn't a silent gap (there's no
`action` alongside it; the episode is over):

```bash
python -m jobagent trace log --json '{
  "skill": "linkedin-apply", "step": "_task_end", "task_id": "job_4021553311",
  "outcome": "applied|skipped:needs_human:<question>|skipped:external_ats|error",
  "observation": {
    "url": "<current URL>", "title": "<page title>",
    "a11y": "pages/<day>-<hash>.a11y.yaml", "page_text": "pages/<day>-<hash>.txt"
  }
}'
```

`total_steps` is filled in for you. Without this record you can still read
back a task's steps with `trace task`, but you can't tell — without replaying
every step — whether it actually finished, and how. This is what lets later
training filter to only-successful trajectories, or specifically study how
failures unfolded.

### Failure and abort trajectories

A corpus of only clean successes teaches a model to click confidently and
never to stop. As of an audit on 30 Aug 2026, that's exactly what this
corpus was: 59 records, zero retries, one non-terminal `blocked`, no
captcha, no external-ATS skip, no needs_human skip, no session-expired
screen — despite every one of those being a documented, expected path above.
This is not lower priority than the happy path; a model that has never seen
a captcha will drive straight through one.

**Abort signals (§0).** The moment you recognize one of the five signals,
before you stop the browser action:

```bash
python -m jobagent trace log --json '{
  "skill": "linkedin-apply", "step": "detect_abort_signal",
  "task_id": "job_4021553311", "plan_step": "verify page state before the next screening field",
  "instruction": "verify page state before continuing to the next screening field",
  "action": {"kind": "read"}, "outcome": "blocked",
  "observation": {
    "url": "<current URL>", "a11y": "pages/<day>-<hash>.a11y.yaml",
    "page_text": "pages/<day>-<hash>.txt"
  },
  "notes": "weekly invitation limit banner shown"
}' --image-b64 @screenshot.b64
```

Attach a screenshot here even though `read` steps don't strictly require
one — recognizing an abort screen *by sight* is the entire point of logging
it. If a task was open when the signal fired, close it:

```bash
python -m jobagent trace log --json '{
  "skill": "linkedin-apply", "step": "_task_end", "task_id": "job_4021553311",
  "outcome": "aborted", "result": {"reason": "weekly_invitation_limit"},
  "observation": {
    "url": "<current URL>", "a11y": "pages/<day>-<hash>.a11y.yaml",
    "page_text": "pages/<day>-<hash>.txt"
  }
}'
```

If nothing was open (the signal fired between jobs, before any `_task_start`),
the `detect_abort_signal` record stands alone — there's no episode to close.

**Expected skips** (`needs_human:<question>`, `external_ats`, `follow_only`,
and similar) are not errors and shouldn't read like one. Log the moment you
detect the condition with a plain `outcome` (`"na"` if nothing actually broke,
`"success"` if the detection itself is the point), then close the episode
with `_task_end` `outcome: "skipped:<reason>"` — see the worked examples in
§2 and §4 above. These are some of the most trainable records in the whole
corpus, because "correctly declining to guess" is exactly the judgment a
fine-tuned model needs to inherit, not just "correctly clicking."

**Retry chains** (attempt 1 fails, attempt 2 succeeds, same `task_id`/`step`)
are the other major gap — see the validation-error pattern in §3. Whenever
you catch and fix your own mistake mid-episode, that correction is worth
logging explicitly rather than only keeping the record of the fix.

**Every step in a recovery sequence is its own record — never a narrative
buried in a later step's `instruction`.** Checked against real §4 episodes
(sending an invite when free personalized notes are exhausted): four
real occurrences of "click Add a note → LinkedIn blocks it with a Premium
upsell → click Send without a note instead" produced one record with
`outcome: "blocked"` (and even that one had no screenshot or coords), one
where the blocking click has **no `outcome` at all** — it reads as a plain
successful click, and the fact that it triggered the upsell only exists as
prose in the *next* step's `instruction` — and two where the pattern isn't
in the data at all because a later step in the same run "knew" notes were
exhausted and skipped attempting the note entirely. None of the four shows
what the blocked screen looked like, whether anything had to be dismissed,
or where the recovery button sat relative to it. That's not a one-off gap;
it's what happens whenever a block-and-recover sequence gets compressed
into one step's prose instead of logged as the separate moments it was:

1. **The action that got blocked keeps its own `outcome`** —
   `"blocked"` or `"error_shown"`, on that record, with a screenshot and the
   coords/element you clicked. Never leave `outcome` unset on a step just
   because a *later* step's `instruction` explains what happened to it —
   that later prose is invisible to anything that isn't reading every
   record's free text.
2. **If dismissing the block takes its own action** — closing a popup,
   pressing Escape, clicking away — log that as its own step with its own
   `action.kind` and, if it was a click, `action.coords` and a screenshot.
3. **The recovery action is a new, fully grounded step**, not a continuation
   of the blocked one — it's usually a different button in a different
   place ("Send without a note" is not "Add a note"). Give it its own
   coords/element/screenshot exactly as you would for a first-try success.
4. **Never call `_task_end` before you've actually verified the outcome.**
   One real episode logged `_task_end` `outcome: "sent"` immediately after
   the send click, then logged a second `_task_end` after `verify_no_error`
   actually ran — two summary records for one task, the first written before
   confirmation. `job applied`/`outreach sent` already carry this rule
   ("only record what you saw confirmed") — it applies exactly as much to
   `trace log`.

```bash
python -m jobagent trace log --json '{...}' --image-b64 "<base64 image>"
```

Rules:

- **Attach a screenshot to every step by default** (see above) — this is the
  single biggest lever on whether the corpus is trainable at all, so don't
  skip it to save a call.
- **Never log field values that are salary figures, phone numbers, or full
  answer text with PII.** `action.value` is for short, generic things like a
  button label or "Yes"/"No" — not the compensation number or a message body.
  The same applies inside `action.candidates` — descriptions and bounding
  boxes only, never PII pulled off the page. A screenshot can still contain
  PII visible on the page (your own contact info, a recruiter's profile) —
  that's expected and fine, it's what the model needs to see; just don't
  caption or transcribe that PII into a text field. The same restraint
  applies to a saved `a11y`/`page_text` file — it's the page's own text, not
  a place to add commentary that names a person.
- This is additive and best-effort for the *quality* nudges. A failed
  `trace log` call should never abort or slow down the actual
  apply/outreach flow — if it errors, ignore it and move on. `trace log`
  also returns a `warnings` list (no `action.coords`/`bbox`, no
  `action.element`, no `action.candidates`, no `expected_effect`, a
  `_task_start` logged out of order, or a task's first step missing both an
  instruction and a `_task_start`) — worth a glance, never worth blocking
  on. The hard requirements in this section (action/observation shape,
  screenshot + dims, plan_step, tool_call fields) are not in that list —
  those reject the call outright rather than warn, same as before.
- Records land in `data/traces/<day>/<session>.jsonl` (gitignored, same as the rest of
  `data/`). Nothing here is read back by the agent or affects its decisions —
  `python -m jobagent trace stats` reports corpus size, step-name spread,
  what fraction of actionable steps actually have coordinates
  (`grounding_coverage`) and how often a retrieved value got overridden
  (`retrieval_coverage`) — both worth checking after a run — and
  `trace task --task-id X` is for spot-checking one episode's steps came out
  in the right order.
