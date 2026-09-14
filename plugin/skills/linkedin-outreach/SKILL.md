---
name: linkedin-outreach
description: Finds recruiters, hiring managers, alumni and mutual-connection employees at target companies on LinkedIn, then drafts connection notes and referral asks for the user's approval before sending. Use when the user asks to reach out to recruiters or HR, ask for referrals, connect with people at a company, or build their network for a job search.
---

# Outreach

Applications go into a pile. A referral gets read. This skill finds the right
people at companies the user has applied to, and drafts something worth
replying to.

**Nothing is sent without approval** (unless `config.outreach_requires_approval`
is false). The flow is always: find → draft → user approves → send → record.

## 1. Pick the companies

Default targets, in priority order:

```bash
python -m jobagent job list --status applied --limit 20
```

Companies applied to today first, then this week's. The user can also name one
directly.

## 2. Find the people

For each company, work these four buckets in order. `config.contacts_per_company`
(default 4) is the total across all buckets — quality over volume.

| Bucket | How | Why first |
|---|---|---|
| **1st-degree already** | `contact list --company "Stripe" --relation connected` | Costs no invite quota. Ask them directly for a referral. |
| **Alumni** | People search with `schoolFilter` = the user's school, `currentCompany` = target | Highest accept rate of any cold invite |
| **Mutual connections** | People search, 2nd degree, sorted by mutual count | The shared name does the vouching |
| **Recruiters / HR** | Keywords `recruiter`, `talent acquisition`, `technical recruiter`, `HR` | They own the req |

URLs and filter encodings: browser-playbook §1. Also add the **hiring manager**
when the job posting names one, or when someone's title matches the team
(e.g. "Engineering Manager, Payments" for a payments role).

Before touching anyone:

```bash
python -m jobagent contact seen --url https://www.linkedin.com/in/priya-sharma
```

`recommend: skip` means do-not-contact or still inside the cooldown window
(`config.min_days_between_touches`, default 5 days). Respect it.

Then record them:

```bash
python -m jobagent contact upsert --json '{"profile_url":"https://www.linkedin.com/in/priya-sharma",
  "name":"Priya Sharma","headline":"Technical Recruiter at Stripe","company_name":"Stripe",
  "title":"Technical Recruiter","role_type":"hr","degree":"2nd","mutual_count":4,
  "is_alumni":0,"location":"Bengaluru"}'
```

`role_type` ∈ `hr` · `hiring_manager` · `employee` · `alumni` · `peer`.

## 3. Draft

Match the template to the relationship:

| Situation | Command |
|---|---|
| 2nd/3rd degree recruiter | `--kind invite --template invite_hr` |
| 2nd/3rd degree employee | `--kind invite --template invite_employee` |
| Alumni | `--kind invite --template invite_alumni` |
| Strong mutual connections | `--kind invite --template invite_mutual` |
| **Already a 1st-degree connection** | `--kind referral_ask_existing` — no invite needed, message them |

```bash
python -m jobagent outreach draft \
  --contact-url https://www.linkedin.com/in/priya-sharma \
  --kind invite --template invite_hr --job-id 4021553311 \
  --extra '{"one_specific_reason":"I built the idempotency layer for a payments API doing 4M txns/day"}'
```

### The one rule that matters

**`one_specific_reason` is mandatory and must be different every time.** Write
it yourself from the job description and the person's profile — a project they
shipped, a system in the posting you've built, a detail from their headline. A
template with the blanks filled and nothing else is what gets people marked as
spam, and it doesn't work.

Also:

- Invite notes are capped at 280 characters. The draft response tells you
  `within_limit` — if false, rewrite shorter rather than letting it truncate.
- Never ask for a referral in the *invite note*. The note earns the connection;
  the ask comes after they accept (that's `linkedin-followup`).
- The exception is an existing 1st-degree connection, where the ask is the
  message.
- Duplicate drafts are refused. If you get `duplicate: true`, that person is
  already in flight — move on.

## 4. Get approval

```bash
python -m jobagent outreach pending
```

Show the user a compact list — name, company, role, and the message body. Let
them approve all, approve individually, edit, or reject:

```bash
python -m jobagent outreach approve --all
python -m jobagent outreach approve --id 12 --body "<their edit>"
python -m jobagent outreach reject --id 13 --reason "wrong team"
```

If the user is away, leave the queue and say how many are waiting. Do not send.

## 5. Send

```bash
python -m jobagent outreach next --limit 10
```

Returns only what's approved and within quota, plus the pacing range. For each,
one at a time, with a 25–70 s pause between:

- **invite** → browser-playbook §4 (Connect → Add a note → Send). Whenever
  free personalized notes run out mid-run, the block and the no-note
  recovery are each their own fully grounded `trace log` step (§4/§9) — not
  folded into a later step's `instruction` text.
- **message kinds** → browser-playbook §5 (Message → type → attach resume → Send)

Record the outcome truthfully — the store is only useful if it matches reality:

```bash
python -m jobagent outreach sent --id 12
python -m jobagent outreach fail --id 13 --error "Connect button not available"
```

If a profile offers only **Follow**, mark the contact `declined` with a note.
If LinkedIn shows the weekly invite limit, stop inviting entirely for the
week. Both are worth logging, not just recording in the store — a "Follow
only" profile is browser-playbook §4's `skipped:follow_only` pattern, and
the invite limit is an abort signal (§0/§9). A `fail --error "..."` is also
worth a matching `trace log` entry (`outcome: "error_shown"`, or `_task_end`
`outcome: "error"`) rather than only living in `jobagent`'s own error column
— the trace corpus is where a model would learn to recognize the failure
*on the page*, which the CLI's `--error` string alone can't teach.

## 6. Report

Say who was contacted at which companies, how many are still awaiting approval,
and the invite quota left for the week. Then note that acceptances get picked up
by `linkedin-followup` on the next run — the user doesn't have to watch for them.
