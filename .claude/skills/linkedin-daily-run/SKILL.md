---
name: linkedin-daily-run
description: Runs the full LinkedIn job-agent cycle in order — inbox, follow-ups, search and apply, then new outreach — respecting daily caps and ending with a summary. Use when the user asks to run the job agent, do today's job search, run the daily cycle, or when a scheduled job-search task fires.
---

# Daily run

One command, the whole cycle. Order matters: **replies before new outreach**.
A recruiter waiting on an answer is worth more than ten new invites, and
answering them first stops the agent from looking like a bot that only talks.

## The order

| # | Phase | Skill | Why here |
|---|---|---|---|
| 1 | Inbox | `linkedin-inbox` | Someone may be waiting. Cheapest, highest value. |
| 2 | Follow-ups | `linkedin-followup` | Accepted invites are warm right now and go cold fast. |
| 3 | Search & apply | `linkedin-apply` | New postings, best in their first 24 hours. |
| 4 | Outreach | `linkedin-outreach` | For companies applied to in step 3. |
| 5 | Report | — | One summary, then stop. |

## Run it

```bash
python -m jobagent run start
```

Check the response before anything else:

- `setup_complete: false` → stop, route to `linkedin-setup`.
- `unanswered_questions > 0` → note it; forms will hit gaps.
- Quota all zero → skip to step 1 and 5 only (inbox and report). Don't burn a
  browser session applying to nothing.

Open **one** browser tab and keep it for all four phases. Close it at the end.

**The session is itself a logged episode.** Open `task_id: run_<run_id>` with
`skill: linkedin-daily-run` before touching the browser — `capture_user_intent`
first (what was asked, and what it parsed into), then `open_run`,
`open_linkedin`, `verify_session`, `open_jobs_tab`. Close it with `_task_end`
carrying the run's shape (searches, candidates screened, jobs scored, applied,
queued, skipped). Everything before the first search URL lives in this task,
and none of it existed in the corpus before 4 Sep 2026 — see
`plugin/references/data-generation-checklist.md`. `jobagent trace disable` turns
logging off for a run that shouldn't contribute to it.

Budget the session: at conservative caps a full run is roughly 10 applications,
15 invites and 10 messages, paced 25–70 s apart — about 45–60 minutes of
browser time. If the user is watching and wants it shorter, cut phase 3's limit
first, never the pacing.

## Attended vs unattended

**Attended** (the user is responding): show the approval queue between phases 2
and 4 and let them approve in one pass. Ask about borderline jobs.

**Unattended** (scheduled fire, or the user has gone quiet): apply and record
normally, queue all outreach without sending, skip anything needing a human
answer with `--reason needs_human:<question>`, and put everything in the final
report. Never send a message no one approved, and never guess a form answer to
avoid stopping.

## Stop conditions

Abort the entire run — don't just skip a phase — on any signal in
`plugin/references/browser-playbook.md` §0: captcha or security check, weekly invite
limit, commercial-use search limit, a restriction banner, or a logged-out
session. Mark it and tell the user plainly what LinkedIn said.

```bash
python -m jobagent run end --notes "stopped: weekly invite limit reached"
```

## Report

```bash
python -m jobagent report daily
python -m jobagent report followups
```

Write the summary yourself in six lines or so — not the raw JSON:

- applied to N jobs (name the three best, with scores)
- M invites sent, K messages sent
- new replies, and which need the user personally
- P messages waiting for approval
- what got skipped and why
- quota left today, and invites left this week

End there. Don't propose a second run — the caps exist for a reason.

## Scheduling

Depends on the client. In Cowork, use the scheduled-task tools — **not** local
cron, which dies with the session. In Claude Code or Cursor, use the OS
scheduler (Windows Task Scheduler, `cron`) invoking the client headlessly from
the project folder. Either way the prompt names this skill and the folder, and
the run needs the machine awake with Chrome logged into LinkedIn.

Whichever client runs it, the phases below are files on disk — a client without
a skill-invocation tool simply reads each `plugin/skills/<name>/SKILL.md` in
order and follows it.
