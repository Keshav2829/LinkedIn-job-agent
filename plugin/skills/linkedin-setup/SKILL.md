---
name: linkedin-setup
description: One-time onboarding for the LinkedIn job agent — reads the user's own LinkedIn profile and resume, then asks the questions needed to fill job applications and target the right companies. Use when the user says set up the job agent, onboard me, configure my job search, re-run setup, or update my profile/preferences/resume.
---

# Setup

Run once. It builds three things the rest of the agent depends on: a **profile**
(who the user is), an **answer bank** (what every application form will ask),
and **preferences** (what they actually want). Everything lands in
`data/state.db` — nothing is re-derived later.

Project root is the folder containing `jobagent/`. All commands run from there.

## 0. Ground rules

- Ask in **batches of four**, following the rounds in `config/questions.yaml`.
  Never dump thirty questions as prose. Use `AskUserQuestion` where the client
  offers it (Cowork, Claude Code); in Cursor or any client without it, ask four
  numbered questions in a message and wait. The batching is the point, not the
  widget.
- **Pre-fill everything you can.** If you scraped it from the profile or parsed
  it from the resume, present it as a confirm/correct, not a blank.
- Compensation and phone number stay in the local database. Say so once, plainly,
  before asking for them.

## 1. Initialise

```bash
python -m jobagent init
```

If it reports `setup_complete: true`, ask whether the user wants a full re-run
or just to update a section, and skip to that section.

## 2. Read their LinkedIn profile

Open one tab (see `plugin/references/browser-playbook.md` §0 and §8):

- `https://www.linkedin.com/in/me/` → `get_page_text`
- `https://www.linkedin.com/in/{slug}/details/skills/` → the full skills list
- `.../details/experience/` and `.../details/education/` if the main page truncates

Extract: name, headline, location, current company + title, every role with
dates, education (school names drive alumni search later), skills, public URL.

Store it:

```bash
python -m jobagent profile set --json '{"name":"...","headline":"...","current_title":"...",
  "current_company":"...","experience_years":5,"location":"...","linkedin_url":"...",
  "skills":["..."],"education":[{"school":"...","degree":"...","year":"..."}],
  "experience":[{"company":"...","title":"...","from":"...","to":"..."}]}'
```

`profile set` mirrors the overlapping fields straight into the answer bank, so
those questions are already handled.

## 3. Get the resume

Ask for the latest resume file. Preferred: a path inside the project folder
(`data/resume.pdf`). Read it and reconcile with the profile — resumes are
usually more current and more specific than the LinkedIn page. Where they
disagree, ask which is right.

Record the path so applications and attachments can find it:

```bash
python -m jobagent profile set --json '{"resume_path":"C:\\local_data\\linkedinAgent\\data\\resume.pdf","resume_name":"Keshav_Agrawal_Resume.pdf"}'
```

The `resume_name` matters: on the Easy Apply resume step the agent picks the
already-uploaded file whose name matches, instead of re-uploading every time.

## 4. Work the question rounds

```bash
python -m jobagent answers missing
```

That is the to-do list. Walk `config/questions.yaml` round by round, asking in
batches of four, then storing:

```bash
python -m jobagent answers set --key notice_period --value "60 days"
python -m jobagent prefs set --json '{"locations":["Bengaluru","Remote"]}'
python -m jobagent config set --json '{"daily_applications":10}'
```

Notes on specific rounds:

- **EEO round** — default every option to "Prefer not to say" and say that these
  are optional on every form. Don't editorialise beyond that.
- **`headcount_band`** — the chosen option carries `min_headcount` /
  `max_headcount`; write those into prefs, not the label.
- **`mnc_preference`** — store as `prefs.mnc_preference`; it steers company
  research, it is not a hard filter.
- **Narrative round** — open text, not multiple choice. These become the seed
  for per-job personalisation, never the final copy.

## 5. Confirm the search plan

Before finishing, show the user what the agent will actually do, and get a yes:

- the search queries you'll run (titles × locations)
- the auto-apply score threshold and what a job at that score looks like
- daily caps: applications / invites / messages
- that outreach is drafted and queued for their approval, not sent

Then verify:

```bash
python -m jobagent answers missing     # should be empty or only voluntary fields
python -m jobagent run start           # setup_complete must be true
python -m jobagent run end --notes "setup"
```

## 6. Offer the schedule

Ask whether to run daily. How you schedule it depends on the client: in Cowork
use the scheduled-task tools (never local cron — it dies with the session); in
Claude Code or Cursor, Windows Task Scheduler or cron calling the client
headlessly. Either way it fires `linkedin-daily-run` on weekday mornings in the
user's timezone, and it needs the machine awake with Chrome logged in.

Suggest waiting until a few runs have been watched before scheduling anything.

## Re-running later

- New resume → repeat §3 and re-check §4.
- Changed targets → re-run only the `targets` / `company_shape` rounds.
- Changed caps → `config set` directly, no need for the whole flow.
