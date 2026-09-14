# LinkedIn Job Agent

An agent that searches LinkedIn, applies to strong matches, and runs recruiter
and referral outreach with human approval. It is **one folder, no install** —
every client reads the same files.

Run every command from this folder (the one containing `jobagent/`).

Every command below is written as `python -m ...`. Most macOS and Linux
installs (stock macOS with no Homebrew, most Debian/Ubuntu systems) have no
`python` on `PATH`, only `python3` — if `python -m jobagent ...` says
"command not found," use `python3` instead for every command in this repo's
docs. Windows almost always has `python` (and it's what the setup docs use).

## Where the substance lives

| | |
|---|---|
| `jobagent/` | the state store + CLI. All memory lives here. |
| `jobagent/browser.py` | the CDP driver — drives your own Chrome, returns small answers |
| `jobagent/harvest.py` | builds the training corpus from Claude Code's own transcripts |
| `annotation_suite/` | standalone (no `jobagent` import) browser UI to review/edit/reward that corpus and export a clean fine-tune set — `python -m annotation_suite serve <file>.steps.jsonl` |
| `plugin/skills/<name>/SKILL.md` | the six procedures. **Single source of truth.** |
| `plugin/references/` | browser playbook, driver adapter, CLI reference |
| `data/state.db` | everything the agent knows |

Client-specific folders (`.claude/skills/`) are **copies for auto-discovery**,
not sources. Edit `plugin/`, then re-copy — see "Keeping copies in sync".

## Triggering, by client

| Client | How |
|---|---|
| **Cursor** | Reads this file automatically. Say "run the LinkedIn daily cycle" — or `@plugin/skills/linkedin-daily-run/SKILL.md` to point at a phase directly. |
| **Claude Code** | Skills auto-discover from `.claude/skills/`. Say `run linkedin-daily-run`, or "use the linkedin-agent subagent to run today's job search". |
| **Cowork** | "Run today's LinkedIn job cycle from this folder." |
| **Anything else** | Read `plugin/skills/linkedin-daily-run/SKILL.md` and follow it. The phases are plain files. |

Phases can be run alone: `linkedin-setup` (first time), then `linkedin-inbox`,
`linkedin-followup`, `linkedin-apply`, `linkedin-outreach` — that order.

## Before any run

1. Chrome open and logged into LinkedIn.
2. A browser driver connected. **CDP** first — `python -m jobagent browse
   status` returning `connected: true` — because it is the only driver whose
   page reads can't fail on size and the only one that attaches a resume
   everywhere. Then **Claude in Chrome**, then **Browser MCP**. Pick one per
   run; two drivers on one tab is a broken run. See
   `plugin/references/browser-adapter.md`, and `config/cdp.setup.md` for the
   one-time CDP setup.
3. `python -m jobagent run start` must return `setup_complete: true`.

On the two MCP drivers there is no file upload, so the resume goes as a link.
Set one or resume-bearing messages are refused rather than sent hollow (CDP
does not need this — `browse upload` attaches the actual file):

```bash
python -m jobagent profile set --json '{"resume_url":"https://..."}'
```

## The state store

Query it; never re-derive it. That is what keeps token use flat as it grows.

```bash
python -m jobagent report daily        # today: applied, sent, replies, backlog
python -m jobagent report followups    # accepted invites owed a message
python -m jobagent outreach pending    # the approval queue
python -m jobagent run quota           # what's left today
python -m jobagent answers missing     # gaps that will stall an application
```

Full reference: `plugin/references/cli.md`.

**The database is per-machine.** A cloud session runs the CLI in its own
container; commit `data/state.db` back here at the end, or that day's work is
lost. Two clients running at once on different copies will diverge — don't.

## Rules that are not negotiable

- **Never guess a form answer.** `answers resolve` returning `found: false`
  means ask the user, or skip the job with `--reason needs_human:<question>`.
- **Never send a message without approval**, unless
  `config.outreach_requires_approval` is false.
- **Compensation fields (CCTC/ECTC) auto-fill from the stored answer bank.**
  `current_ctc` / `expected_ctc` were already confirmed once during
  `linkedin-setup` — filling them again on every application is not a new
  disclosure and needs no fresh confirmation, and is not something to block
  or skip on. If the stored value is a range (e.g. "35-40 LPA"), convert to
  a single number using the **upper bound** — LinkedIn's numeric field
  forces one number anyway. Only fall back to `needs_human:<label>` when no
  CTC value is stored at all. **Recruiter messages that state a salary
  figure still go through the outreach approval queue** like any other
  drafted message — that's review, not a block, and is unaffected by this.
- **Only record what you saw confirmed** — the "Your application was sent to X"
  dialog, not the Submit click.
- **Type into fields; don't set them.** LinkedIn's React forms ignore values
  written to the DOM. Select, type, blur. Validation runs on blur only.
- Daily caps and 25–70 s pacing are load-bearing, not decoration.
- **Log steps for future local-model training as you go** — `trace log` after
  every `find` call and verification step, per `browser-playbook.md` §9,
  always with the `task_id` of the episode it belongs to (job_id / profile_url
  / thread URL), `coords`/`bbox`/`element` when the step was a
  click/type/select, and **`--image-b64` with a screenshot on every step** —
  a coordinate with no image behind it can't train anything. **Every
  non-`tool_call` step also needs `observation.a11y` and `observation.page_text`
  set, or `trace log` rejects it** — these are *paths*, not inline text: call
  `read_page`/`get_page_text` for the state you're acting on, save the output
  to a file under `data/traces/.../pages/` (e.g. `pages/<day>-<hash>.a11y.yaml`
  / `.txt`), and put that path in the record. Do this **before** the action
  that will be logged, not after — capture the pre-click page state, then act.
  A click/type/select with `coords`/`bbox` also needs `observation.viewport`,
  `observation.scroll` and `observation.page_dims`. This applies to
  `submit_application` and every other ordinary step just like it applies to
  `find_easy_apply_button` — there is no exemption for steps near the end of
  an episode; only `_task_start`/`_task_end` skip the a11y/page_text
  requirement. This includes
  each screening-question field during Easy Apply: log `fill_screening_field`
  right after you fill and blur it, not batched at the end of the episode —
  see `browser-playbook.md` §3. **Open** every
  task with a `_task_start` record naming the instruction that authorized it
  (a user request, the daily run, or a skill default) and **close** it with a
  `_task_end` record giving its final outcome. Whenever a value came from
  `jobagent`'s own store (`answers resolve`, `job score`, `contact seen`)
  rather than off the page, log it under `retrieval` — and when you didn't
  trust that value as-is (it was wrong, low-confidence, or you used judgment
  over it), set `accepted: false` with an `override_reason` instead of just
  silently fixing it. Never let a failed `trace log` call block or slow the
  actual run, and never log a salary figure, phone number, or other PII as
  `action_value` or inside `candidates`.
- **Log the search and the decision, not just the application.** The first
  three days of corpus held 154 records and zero from any search, and
  `score_job` — the lookup gating every application — had zero records across
  31 episodes. A run has three nested episodes: the session (`run_<run_id>`,
  from the user's request through opening the Jobs tab), each search
  (`search_<slug>`, dead ends included), and each job (`job_<job_id>`, opened
  at the dedupe check rather than at the Easy Apply button, so a job rejected
  on score still leaves a complete episode). `step` and `skill` are closed
  vocabularies the CLI enforces — `jobagent trace steps` prints them, and
  `plugin/references/data-generation-checklist.md` is the per-step lookup.
  After every run read `trace stats`: `off_vocabulary_steps` empty, and
  `never_logged_steps` containing nothing the run actually did.
- **Logging can be turned off for a run** — `jobagent trace disable` (or
  `config.trace_logging: false`) makes `trace log` a reported no-op that
  writes nothing. Keep calling it unconditionally; don't branch per step.
  It defaults to on.
- **After a run, harvest it** — `python -m jobagent harvest`. `trace log`
  captures what you remembered to write down; the harvester reads Claude
  Code's own session transcripts and captures what happened. On 5 Sep that
  was 123 browser actions against 1 logged step. It finds the transcripts
  itself, joins them to `data/traces/` for the skill/step/task_id labels,
  resolves the screenshots, redacts PII, and marks every record with whether
  its reasoning survived (`thinking.state`) so a later pass can fill the gaps
  without ever confusing generated reasoning for real. See
  `plugin/references/harvest.md`. Keep calling `trace log` regardless — it is
  the only source of the semantic labels the transcript cannot know.
- **The corpus documents runs; it cannot yet train one.** A 4 Sep audit
  of 199 records found no page state stored on any record, scrolls and
  reads absent from the action space, coordinates in a different frame
  from the screenshots, and no scroll offset. `trace-schema-v3.md`
  specifies what a trainable record needs. It is a spec, not shipped —
  keep logging the v2 way; v3 is additive and v2 records stay valid.

## Known LinkedIn behaviour (verified 29 Aug 2026, live run)

- **`sortBy=R`, never `DD`.** Date sort on a broad keyword returns
  staffing-agency reposts — an "AI Engineer" search surfaced Power BI, .NET and
  frontend-intern roles in its top seven.
- Sharp keywords beat tight time windows. "LLM AI Agents Engineer" over 30 days
  beats "AI Engineer" over 24 hours.
- The `(12)` in the page title is the notification badge, not a result count.
- **CCTC/ECTC fields are numeric-only**, rupees per annum. A range is
  impossible; convert the stored `current_ctc`/`expected_ctc` to a single
  number yourself (upper bound of a range) and fill it — don't ask, don't
  skip.
- The job page's "Meet the hiring team" panel names the recruiter and employees
  attached to that req — the best outreach targets available, free. Capture
  them with `contact upsert` during the application.

## Keeping copies in sync

`plugin/` is the source. After editing a skill:

**macOS / Linux:**
```bash
cp -r plugin/skills/* .claude/skills/
cp plugin/agents/linkedin-agent.md .claude/agents/
```

**Windows (PowerShell):**
```powershell
Copy-Item plugin\skills\* .claude\skills\ -Recurse -Force
Copy-Item plugin\agents\linkedin-agent.md .claude\agents\ -Force
```

Cursor needs no copy — it reads this file and the `plugin/` paths directly.
