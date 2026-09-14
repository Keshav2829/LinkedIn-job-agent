# `jobagent` command reference

Run from the project root — wherever this repo is checked out:

```bash
python -m jobagent <group> <action> [flags]
```

Every command prints **one line of JSON**. That is deliberate: read the JSON,
act, move on. Never dump the database into the conversation.

Set `LINKEDIN_AGENT_DB` to override the store location (defaults to
`data/state.db`).

---

## Setup

| Command | Returns |
|---|---|
| `init` | creates/upgrades the DB, seeds the canonical question list |
| `profile get` / `profile set --json '{...}'` | your profile (merged, not replaced) |
| `prefs get` / `prefs set --json '{...}'` | job preferences |
| `config get` / `config set --json '{...}'` | caps, thresholds, approval switches |
| `answers missing` | every question still without an answer — drives onboarding |
| `answers set --key K --value V [--question Q] [--alias A]` | store an answer |
| `answers resolve --question "..." [--options "Yes\|No"]` | **the form-filling call** |
| `answers list [--category identity\|work_auth\|employment\|legal\|eeo\|narrative]` | |

`answers resolve` returns `{found, key, value, answer, confidence}`. `answer`
is already coerced to the option set you passed. `found:false` means *ask the
human* — never guess.

## Jobs

| Command | Use |
|---|---|
| `job seen --job-id X` | cheap dedupe before opening a card |
| `job upsert --json '{job_id,title,company_name,location,url,easy_apply,workplace,posted,description}'` | record a discovery |
| `job score --job-id X` | score it; returns `{score, reasons, status, auto_apply}` |
| `job queue [--min-score N] [--limit N]` | what to apply to now, already quota-capped |
| `job applied --job-id X [--answers '{}'] [--screening '[]'] [--notes]` | after the confirmation dialog only |
| `job skip --job-id X --reason R` | external ATS, needs_human, closed, etc. |
| `job list --status applied\|queued\|skipped` | |

## Companies & contacts

| Command | Use |
|---|---|
| `company upsert --name N --json '{kind,headcount,industry,linkedin_url}'` | |
| `company blacklist --name N` | never apply again |
| `contact seen --url U` | returns `recommend: ok\|skip` (cooldown + do-not-contact) |
| `contact upsert --json '{profile_url,name,headline,company_name,role_type,degree,mutual_count,is_alumni,alumni_school}'` | `profile_url` is required — it's the unique key contacts are matched on. If a "Meet the hiring team" panel entry has no clickable profile link, it can't be recorded as a contact; skip it. |
| `contact mark --url U --relation connected\|declined\|withdrawn` | |
| `contact pending-invites [--older-than-days 7]` | invites awaiting acceptance |
| `contact awaiting-followup` | accepted, but we haven't sent the real message |
| `contact list --relation connected --company "Stripe"` | |

`role_type` is one of `hr`, `hiring_manager`, `employee`, `alumni`, `peer`.

## Outreach (draft → approve → send)

| Command | Use |
|---|---|
| `outreach draft --contact-url U --kind K [--job-id X] [--template T] [--extra '{}'] [--body "..."]` | renders and queues |
| `outreach pending` | the approval queue — show this to the user |
| `outreach approve --id N [--body "edited text"]` / `--all` | `--body` updates the sent copy only — the original draft stays in `body_original`, untouched, for every row (see below) |
| `outreach reject --id N [--reason R]` | |
| `outreach next [--kind invite] [--limit N]` | approved + within quota = send these |
| `outreach sent --id N` | **only after the browser confirms** |
| `outreach fail --id N --error "..."` | |

`kind` ∈ `invite`, `hr_pitch`, `referral_ask`, `referral_ask_existing`,
`intro_peer`, `followup`, `thanks`.
Templates are named for the kind; override with `--template invite_alumni` etc.
Available: `invite_hr`, `invite_employee`, `invite_alumni`, `invite_mutual`,
`hr_pitch`, `referral_ask`, `referral_ask_existing`, `intro_peer`,
`followup_nudge`, `thanks_reply` (`template list` to confirm).

Every drafted row stores the rendered text twice: `body` (what actually gets
sent — `approve --body` may overwrite this) and `body_original` (the
first-drafted text, set once at `draft` time and never touched again). A
human editing a draft before sending it is the highest-value signal in the
whole store for a preference-tuned writer model — `outreach list` returns
both columns so that pair is never lost to an in-place overwrite. Rows drafted
before this field existed carry `body_original: null`; that's expected, not
a bug — there's no way to recover a draft that was never recorded separately.

`--extra '{"one_specific_reason":"..."}'` is where the personal sentence goes.
**Always fill it.** A message without it reads like a mail merge.

## Replies & reporting

| Command | Use |
|---|---|
| `reply add --contact-url U --snippet "..." [--sentiment positive]` | idempotent |
| `reply unnotified` | what to tell the user about |
| `reply mark-notified --all` | after you've told them |
| `run start` | opens a run, returns quota + setup status |
| `run quota` | what's left today |
| `run end [--notes "..."]` | |
| `report daily` | today's applications, sends, replies, approval backlog |
| `report pipeline` | lifetime counts by status |
| `report followups` | stale invites · accepted-but-unmessaged · silent applications |

## Training-data logging

Not app state — doesn't feed scoring, quotas, or dedupe. Appends
(page state, instruction, action) records to `data/traces/<day>/<session>.jsonl` for a
future local-model fine-tune. See `browser-playbook.md` §9 for what to log
and when.

| Command | Use |
|---|---|
| `trace log --json '{...}' --image-b64 B64` | append (or backfill-merge — see below) one step (full field list below); `--image-b64` saves a screenshot alongside it, format sniffed (PNG/JPEG/GIF/WEBP). `step_index` (scanned across all days, not just today), `coords_norm`/`bbox_norm`, `total_steps` (on `_task_end`), `trace_schema`, and `persona_id` are auto-filled. **Rejects the call** (returns `{"ok":false,"error":...}`, writes nothing) rather than warning when: `driver` is set to something outside `cdp`/`claude_in_chrome`/`browsermcp`/`claude_browser_pane`; a step with an `action_type` has no `driver` at all; `outcome` is used outside its closed vocabulary for that record type (see below); a `click`/`type`/`select` step has no screenshot; or `skill`/`step` is outside its closed vocabulary (an unknown `step` comes back with a near-match suggestion; the five names that leaked into the first corpus — `select_resume`, `view_job_posting`, `click_easy_apply`, `fill_screening_questions`, `review_application` — are rewritten to their canonical step instead of rejected, and a `linkedin-apply` `task_id` given as a bare number or a `/jobs/view/…` URL is normalized to `job_<job_id>` with a warning). A rejected log call never blocks the actual apply/outreach/inbox flow — only the log line is refused. When `config.trace_logging` is false the call is a reported no-op: `{"ok":true,"skipped":"trace_logging disabled…"}`, nothing written. Returns `warnings` (still non-blocking) for softer gaps: missing `coords`/`bbox`/`element`, or a `_task_start` out of order. **A base64 screenshot over ~130KB will hit the OS per-argument exec limit if passed inline** — write it to a file and pass `--image-b64 @<path>` instead. |
| `trace stats` | corpus size by day/skill/driver/**schema** (`by_schema`: records tagged `v1_legacy` predate the schema-2 validation above — exclude them from a real training pull), step-name spread (`top_steps`), task/episode counts (`tasks_with_start_record`, `tasks_with_end_record`), retry counts, `grounding_coverage` — % of click/type/select steps with real coordinates, and `retrieval_coverage` — how many steps cite a data lookup and how many of those got overridden. **`never_logged_steps`** lists canonical steps with zero records anywhere in the corpus and **`off_vocabulary_steps`** lists names that predate the vocabulary check — read both after every run; the missing search phase was invisible for three days behind a healthy-looking record count |
| `trace list` | which days have traces |
| `trace task --task-id X` | reconstruct one episode's steps in order — spot-check a trajectory |
| `trace steps` | print the live `skill`/`step` vocabularies and the alias map, plus whether logging is currently on |
| — | *Schema 3 is specified but not implemented — see `trace-schema-v3.md`. It adds `observation` (page text, a11y tree, scroll offset, native-resolution screenshot) and a real `action` block, which is what the corpus needs before it can train a policy rather than just document one.* |
| `trace enable` / `trace disable` | flip `config.trace_logging`. While off, `trace log` writes nothing and says so — keep calling it unconditionally rather than branching per step. Defaults to on |

`--json` fields: `skill`, `step` (closed vocabulary — see playbook §9;
`_task_start`/`_task_end` reserved), `task_id`, `attempt`, `url`,
`instruction`, `element{role,name,ref}`, `coords{x,y}`, `bbox{x,y,w,h}`,
`viewport{width,height}`, `candidates[{role,name,bbox}]`, `action_type`,
`action_target`, `action_value`, `outcome`, `result{...}`, `driver`,
`confidence`, `backfill` (bool),
`retrieval[{source,query,result,accepted,override_reason}]`.

`outcome`'s vocabulary now depends on `step`: an ordinary step takes
`success|error_shown|blocked|na`; `_task_start` takes no `outcome` at all
(use `source`: `user_request|scheduled_daily_run|skill_default`);
`_task_end` takes `applied|sent|not_applied|error|aborted` or
`skipped:<reason>`. Anything richer — a confirmation string, an application
id, a match score, a company name — goes in `result`, not `outcome`.

`task_id` groups one episode's steps (a job_id for an application, a
profile_url for outreach/followup, a thread URL for inbox) so they can be
replayed in sequence later, not just used as isolated grounding examples.
Open every task with a `step: "_task_start"` record naming the instruction
that authorized the episode (a user request, the scheduled daily run, or a
skill's own default behavior), and close it with a `step: "_task_end"`
record giving the episode-level `outcome`. Use `retrieval` whenever a value
came from `jobagent`'s own store (`answers resolve`, `job score`, `contact
seen`, …) rather than being read off the page — and set `accepted: false`
with an `override_reason` whenever the retrieved value was wrong or you used
judgment over it instead of trusting it as-is; those overrides are the
highest-value examples in the corpus.

Pass `"backfill": true` with the same `task_id`/`step` as an already-logged
record to update that record in place (adding a `backfilled_at` timestamp)
instead of appending a near-duplicate line. No match found → logs fresh and
adds `backfill_target_not_found: true`.

See `browser-playbook.md` §9 for the
full schema and the PII rule (no salary figures, phone numbers, or PII
inside `candidates`).
