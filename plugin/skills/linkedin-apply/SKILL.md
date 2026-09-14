---
name: linkedin-apply
description: Searches LinkedIn for jobs matching the user's stored profile, scores each one, and submits Easy Apply applications for strong matches while skipping duplicates. Use when the user asks to find jobs, search for roles, apply to jobs on LinkedIn, or run today's applications.
---

# Search, score, apply

The application half of the agent. Discovery is cheap and deduped; applying is
capped and audited. Read `plugin/references/browser-playbook.md` before the first
browser call and `plugin/references/cli.md` for command details.

**Capturing runs is how the training corpus gets made** — the goal is a
small local model that can run the mechanical half of this on its own, and
a run that isn't captured produces nothing toward it. Capture the discovery
half, not just the applying half: the first three days held 154 records and
not one came from a search or a scoring decision, so a model trained on
them could fill an Easy Apply form and could not find a job to fill it for.
Each step below names what to capture as it happens;
`plugin/references/data-generation-checklist.md` says which training example
each one feeds and what the record must carry for that example to exist.
`python -m jobagent trace disable` turns logging off for a run that
shouldn't contribute to the corpus.

## 1. Open the run

Log `capture_user_intent` **first**, before any tool call — the request
near-verbatim plus what it parsed into (titles, locations, filters, limit).
Open the session task `run_<run_id>` with `_task_start` as soon as you have
the run id.

```bash
python -m jobagent run start
```

Returns the run id, today's remaining quota, and `setup_complete`. If setup is
incomplete, stop and route the user to `linkedin-setup`. If
`applications.left` is 0, skip straight to outreach or report and stop.

Read `prefs` and `config` from that response — don't re-query per job.

**Log `open_run`** with that response as a `retrieval` entry
(`source: "run.start"`). A run that stops right here because setup is
incomplete or quota is spent is a real trajectory — log it with
`outcome: "blocked"` and close the session task, rather than leaving
nothing behind.

On the way to the search, three more session steps: `open_linkedin` (the
navigate), `verify_session` (logged in, no captcha or banner), and
`open_jobs_tab` (the click, with the sibling nav items as `candidates`).

## 2. Search

Build one search URL per (title × location) pair from
`prefs.target_titles` and `prefs.locations`. Always include `f_AL=true`
(Easy Apply) and **`sortBy=R`**.

**Do not sort by date.** Verified 29 Aug 2026: `sortBy=DD` on a broad keyword
returns almost pure staffing-agency reposts — an "AI Engineer" search in India
surfaced Power BI, .NET, AEM and frontend-intern roles in its top seven, and
nothing worth applying to. Relevance sort on the same filters returned real
LLM/agent roles.

**Prefer sharp keywords over tight time windows.** "AI Engineer" matches on any
token; "LLM AI Agents Engineer" returns a usable set. A 30-day window
(`f_TPR=r2592000`) on a specific keyword beats a 24-hour window on a vague one.
Start at `r604800`; widen to `r2592000` when a query returns under five results.

The `(12)` in the page title is LinkedIn's notification badge, not a result
count — read the "N results" element in the page body.

Open **one** tab and reuse it for every search.

**Read more than the first page.** A page holds 25 cards; stopping there
after one search means most of a broad result set was never looked at, not
that it was screened and rejected. Keep pulling pages with `start=25`,
`start=50`, … (browser-playbook §1, "Getting past the first page" — a real
`browse open` navigation, not a scroll) until **any** of these is true:
you've screened 3 pages (75 candidates) for this keyword/location, you've
gone a full page with zero unseen jobs (`job seen` all `known: true` — the
rest of this query is jobs you've already processed), or you have enough
`auto_apply` candidates queued to fill what's left of today's quota. A page
that returns fewer than 25 cards is the last page — stop there regardless.

**Each distinct keyword/filter combination is its own logged episode**,
`task_id: search_<slug>` — open it with `_task_start` and close it with
`_task_end`, *including* the ones that return nothing.
`outcome: "skipped:no_matching_jobs"` on a dead-end search is a trajectory
worth as much as a successful one: it's what teaches a model to recognize a
bad query and move on rather than grinding. Within it, log
`type_search_query` when you type into the search box, one `apply_filter`
record **per filter** you toggle, `search_jobs` when the results URL loads
(with the real "N results" count — not the title badge), and
`paginate_results` **every time you pull another page** — it's a decision
record ("the first page wasn't enough, so I kept looking"), not an optional
extra, so log it whether or not a later page turns up anything.

For each result card, read: job id (`currentJobId`), title, company, location,
workplace type, posted age, Easy Apply badge.

**Log `screen_job_candidate` for every card you actually read** — the
rejects as readily as the shortlist, with `verdict` and `reject_reason`. The
corpus currently has zero of these, which means it contains only jobs that
were applied to and no example of judgment being exercised. When you click a
shortlisted card, log `open_job_card` with the neighbouring cards you didn't
click as `candidates`.

**Dedupe before you open anything:**

```bash
python -m jobagent job seen --job-id 4021553311
```

`known: true` → move on, no matter its status. Never re-open a job the store
already has.

This lookup decides whether the job gets looked at at all, so it opens that
job's episode: `_task_start` on `job_<job_id>`, then `check_job_seen` with
the response as a `retrieval` entry (`source: "job.seen"`). On
`known: true`, close it immediately with
`_task_end` `outcome: "skipped:already_seen"`.

## 3. Score

For genuinely new jobs, open the card, read the description, then:

```bash
python -m jobagent job upsert --json '{"job_id":"4021553311","title":"Senior Backend Engineer",
  "company_name":"Stripe","location":"Bengaluru, India","workplace":"hybrid",
  "url":"https://www.linkedin.com/jobs/view/4021553311/","posted":"2 days ago",
  "easy_apply":1,"description":"<first ~2000 chars>"}'
python -m jobagent job score --job-id 4021553311
```

The response tells you what to do:

| Response | Action |
|---|---|
| `skip_reason` set | Nothing. It's blacklisted / excluded / wrong location. |
| `auto_apply: true` | Apply in step 4. |
| `status: queued`, `auto_apply: false` | Borderline. Collect these and show the user at the end; apply only if they say so. |
| `status: scored` | Below the review threshold. Leave it. |

Store the description you read — it is what makes the outreach personal later,
and re-fetching it costs another page load.

**Log all four of these, every time:** `open_job` (the navigate),
`read_job_description` (what you extracted — seniority, must-haves — not the
raw text), `score_job` with the `job score` response as a `retrieval` entry
(`source: "job.score"`), and then `record_score_decision` — the branch
itself, carrying `decision` (`apply_now` | `queued_for_review` |
`below_threshold` | `excluded`) and `next_action` (`open_easy_apply` |
`await_user` | `next_job`).

`score_job` had **zero** records across 31 episodes before this was written.
It is the lookup that gates every application, and without it the corpus
shows applications happening for no visible reason. `record_score_decision`
is the "if it matches, apply; otherwise next job" branch — log it for jobs
that go no further too, then close that episode with `_task_end`
`outcome: "not_applied"`. A rejection is a complete episode, not a
non-event.

## 4. Apply

```bash
python -m jobagent job queue --limit 10
```

This is already capped by the daily quota and excludes anything applied to.
Work the list **one job at a time**, pausing 25–70 s between applications.

Follow browser-playbook §2 for the Easy Apply modal and §3 for screening
questions. The two rules that matter:

- **Never guess an answer.** `answers resolve` returning `found: false` means
  ask the user (interactive) or skip with `--reason needs_human:<question>`
  (unattended). Log the skip itself, not just the skipped job — browser-
  playbook §9, "Failure and abort trajectories."
- **Only record a submission you saw confirmed.** The "Your application was
  sent to X" dialog is the trigger — not clicking Submit.

```bash
python -m jobagent job applied --job-id 4021553311 \
  --answers '{"notice_period":"60 days","expected_ctc":"38 LPA"}' \
  --screening '["Years of experience with Go? — answered 2 at confidence 0.51"]'
```

Uncheck "Follow {company}" on the review step unless the user opted in.

If the button says **Apply** rather than **Easy Apply**, it leaves LinkedIn:

```bash
python -m jobagent job skip --job-id X --reason external_ats
```

Mention these in the summary — some are worth the user doing by hand. Log
this skip too (browser-playbook §2/§9) — it's a routine, expected outcome,
not an error, and the corpus currently has none of them.

## 5. Hand off to outreach

Every company you applied to is now a target. Say so and offer to run
`linkedin-outreach` for them — recruiter outreach on the same day as the
application is the whole point of the agent.

## 6. Close out

```bash
python -m jobagent report daily
python -m jobagent run end --notes "..."
python -m jobagent trace stats
```

Close the session task with `_task_end`, `result` carrying the run's shape
(`searches`, `candidates_screened`, `jobs_scored`, `applied`,
`queued_for_review`, `skipped`).

Then **read `trace stats` before you report**: `off_vocabulary_steps` should
be empty, and `never_logged_steps` should contain nothing this run actually
did. That single check is what would have caught the missing search phase on
day one instead of three days later.

Report back in a few lines: how many applied and where, the borderline ones
awaiting a decision, anything skipped for a missing answer, and quota left.
Do not paste the whole JSON.

## Stop conditions

Abort the run on any signal in browser-playbook §0 — captcha, limit banner,
restriction notice, logged-out state. Mark the run ended with the reason and
tell the user what happened. Never retry a blocked action twice. Log the
signal itself before you stop — browser-playbook §0/§9 — an abort a model
never saw is an abort it will drive straight through later.
