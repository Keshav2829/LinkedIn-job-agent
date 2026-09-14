# Data generation checklist

**This file is not about logging.** Its only purpose is to produce
supervised training data for a small local model that will eventually run
the mechanical half of this agent — find the button, judge the card, decide
whether to apply, fill the field, recognise when to stop. Every rule below
exists because an SLM needs something specific, and the rule says what.

The test of a record is not "does it have the required fields." It is:

> **Can a training pair be built from this record — and is the *input* side
> of that pair actually present, not just the target?**

A record can satisfy every field rule and teach nothing. That is the
failure mode this file is written against.

---

## The measurement that produced this rewrite

4 Sep 2026, on a clean 119-record day that scored **26/26 against the old
conformance checklist**:

| Example type | Records | Usable | Why the rest yield nothing |
|---|---:|---:|---|
| grounding | 19 | **0** | no scroll offset; image frame ≠ coordinate frame; no page text |
| judgement | 38 | **0** | no bbox — which card was judged is unrecoverable; card text never stored |
| decision | 6 | **0** | JD text lives in `state.db`, not the record; `score_job` is a CLI output, not a behaviour |
| stop / abort | 2 | **0** | no screenshot — recognising an abort screen by sight is the whole point |
| tool use | 9 | **0** | no `action_type` for a tool call, so calling the store is not an emittable action |
| annotation | 39 | — | `verify`/`na` are notes about what happened, not actions to predict |
| episode scaffolding | 18 | — | `_task_start` / `_task_end`, correctly not examples |
| **Total** | **119** | **0** | **0.0% of records yield a usable training example** |

On the 19 grounding candidates, three requirements each failed on **100%**
of records — screenshot, coords and viewport were present every time, but
scroll offset, frame agreement and page text were present *never*. Dropping
any single requirement changes the yield from 0 to 0. There is no partial
credit available; the gap is structural.

`tools/audit_training_yield.py` is this table. Run it, not the conformance
audit, to know whether a day was worth capturing.

---

## Provenance: the question underneath all of it

Before "is this a usable pair", a narrower question the corpus has to
answer: **this record claims X — where did X come from?**

Measured across the whole corpus, 158 records carrying **483 individual
assertions**:

| Source of the claim | Assertions | |
|---|---:|---|
| **SELF** — derivable from the record or its episode (URL params, the action taken, counts of sibling records, a tool's own error string) | 164 | 34.0% |
| **CITED** — a `retrieval` entry supports it | 24 | 5.0% |
| **IN_SHOT** — visible in the attached screenshot; checkable by eye, but nothing links the claim to the element it came from and no text form is stored | 171 | 35.4% |
| **EP_SHOT** — visible only in a screenshot attached to a *different* record of the same episode | 31 | 6.4% |
| **NO_SHOT / RECALLED** — no evidence anywhere; came from the annotating model's priors | 46 | 9.5% |
| unclassified | 47 | 9.7% |

**Only 5% of assertions cite their source.** That is the direct answer to
"the data doesn't show how the result was generated" — for the most part,
it doesn't, and where it does, it is by accident of the claim being
self-evident rather than by the record pointing at anything.

Worst offenders, by step:

| Step | Assertions | Recalled | Note |
|---|---:|---:|---|
| `screen_job_candidate` | 130 | **45** | the `reject_reason` problem below |
| `verify_no_error` | 29 | 0 | `validation_errors: 0` is a claim about the page with no page state to check it against |
| `read_job_description` | 25 | 0 | `must_haves` / `seniority` are the annotator's extraction; the JD text itself is not in the record |
| `find_easy_apply_button` | 2 | 0 | zero SELF, zero CITED — the whole record is an unsourced assertion about the page |

Note what the screenshot does and does not buy. It is legible — a 768×320
JPEG of a results list still shows titles, companies, "4 hours ago" and
the Easy Apply badge clearly enough to check a claim by eye. What it does
**not** do is say *which* element a claim refers to, or provide the text in
a form anything can join on. 171 assertions sit in that state: verifiable
by a human squinting, useless as supervision.

---

## The five example types

Everything worth capturing is one of these. If a step produces none of
them, it is scaffolding — log it for episode structure, but do not expect
it to train anything.

### 1. Grounding — `(page state, instruction) → where and how to act`

The core skill: given what is on screen and a micro-instruction, emit an
action and its target.

**Target:** `action.kind` + `coords`/`bbox` (or `ref`).

**The input must contain all of:**

| Required | Because |
|---|---|
| screenshot, **native resolution, PNG** | not for legibility — a 768×320 JPEG of a results list turns out to be *readable*, titles, companies, "4 hours ago" and the Easy Apply badge all intact. The reason is fidelity of the label: coordinates recorded in a 1536×639 frame against a 768×320 image are off by 2×, and a lossy resize moves element edges relative to the `bbox` that is supposed to name them |
| `viewport` | the coordinate frame; without it a number is not a place |
| `screenshot_dims` **equal to `viewport`** | the 4 Sep corpus stored coords in a 1568×744 frame against 784×372 images — every coordinate off by 2× |
| `scroll` `{x, y}` + `page_dims` | a `bbox` at y=280 names a place on the *screen*; a model needs the place on the *page*, and needs to know what is off-screen |
| `page_text` **and** `a11y` tree | pixels alone can't learn ref-based action; the tree alone can't see what isn't in it — and on 4 Sep the post-submit dialog rendered on screen while being **absent from the tree**, three times across three companies |
| `candidates` — the elements *not* picked | without contrast, a card click teaches "any card is the right card" |

**Disqualifiers:** any one of the above missing. Not negotiable — a
coordinate with no frame, no scroll and no page state is a number with
nothing behind it.

### 2. Judgement — `(one candidate) → keep or reject, with a reason`

Screening a results card. **The largest block of the 4 Sep day (38 records,
29 of them rejects) and none of it is learnable**, because only the verdict
was stored.

**Target:** `verdict` (`candidate`|`reject`) + `reject_reason`.

**The input must contain:**

- **`bbox` of the card being judged**, in the recorded frame. Without it,
  which of ~25 cards this verdict refers to is unrecoverable.
- **The card's own text**, captured verbatim — not a written summary of it.
  A `result` saying `{"title": "AI Engineer", "company": "EIRIS"}` is the
  annotator's paraphrase; the model needs what was on screen.
- The **profile/prefs state** the judgement was made against (or a stable
  reference to it), since "too senior" is only meaningful relative to a CV.

#### The provenance rule — the one that matters most here

> **Every claim in a verdict must resolve to either the captured
> observation or a `retrieval` entry. If it resolves to neither, the
> verdict is `needs_lookup`, never `reject`.**

Measured on the corpus at 158 records: **19 of 36 reject reasons cite
facts that are on no page, in no store and in no retrieval entry.**

```jsonl
{"step":"screen_job_candidate","action_target":"Senior Deep Learning Engineer at Nanonets",
 "result":{"verdict":"reject","reject_reason":"small AI startup, well under 2000 employees"}}
```

A LinkedIn result card renders title, company, location, workplace,
posted age, the Easy Apply badge, applicant count and sometimes salary.
**It does not render headcount.** So "well under 2000 employees" came from
the annotating model's background knowledge of Nanonets. The `companies`
table has a `headcount` column and no row for that company; there is no
`retrieval` entry; there is no page state. The number is unsourced.

Two things follow, and both are severe:

1. **An SLM trained on this learns to fabricate.** It sees a company name
   and a confident numeric claim about its size, with nothing in the input
   that could have produced the number. The pattern it can actually learn
   is *"emit a plausible headcount assertion and reject on it"* — applied
   with equal confidence to companies it has never heard of.
2. **The policy is not even self-consistent.** `prefs.min_headcount` is
   **3000**. One record cites "the 3000-headcount preference", another
   "well under 2000 employees". The same stored threshold, recalled
   differently twice, because it was recalled rather than read.

The fix is not a bigger `reject_reason`. It is:

- capture the card text, so card-visible claims are checkable;
- for anything not on the card, **look it up and cite it** — the
  `companies` table already has `headcount`, `size`, `is_mnc` and `kind`
  columns, so `company.seen` belongs in `retrieval` exactly as `job.score`
  does;
- when the lookup has no answer, the verdict is `needs_lookup` with the
  question named. **A reject with no evidence is not a cheap reject; it is
  a poisoned example**, and one is worse than none.

Card-visible rejects are fine and valuable — "not Easy Apply", "posted 3
months ago", "title is Director", "already Applied" are all readable off
the card and need no lookup. It is the invisible attributes that need
sourcing.

### 3. Decision — `(JD + score + thresholds) → apply / queue / skip`

**Target:** `decision` + `next_action` + reasons.

**The input must contain the job description text.** On 4 Sep the JD went
into `state.db` via `job upsert` and never into the record, so the decision
records have a target and no input. Either inline the JD (or its first
~2000 chars) in the record, or store a content hash plus a resolvable
reference — but the pair must be reconstructable from the corpus alone.

**`score_job` is not a decision example.** It is a deterministic CLI
output. What is worth learning is `record_score_decision` — and especially
the cases where the scorer was *overridden*: on 4 Sep, `job.score` returned
87/auto-apply for a posting demanding 5+ years against a ~4-year profile.
That override is the highest-value example in the whole corpus, and it
only trains anything if the JD that justified it is in the record.

### 4. Field fill — `(question label, answer bank) → value`

**Target:** `action_value` + `confidence`.

**The input must contain:** the exact question label as rendered, the
`retrieval` entry showing what the answer bank returned, and — when
`accepted: false` — the `override_reason`. Overrides here teach *when not
to trust retrieval*, which nothing else in the corpus does.

Never put a salary figure, phone number or other PII in `action_value` or
`candidates`.

### 5. Stop / abort — `(page state) → stop, and why`

**Target:** the `stop` action + reason.

**The input must contain a screenshot and page state**, because
recognising a captcha, a limit banner or a logged-out screen *by sight* is
the entire skill. A `stop` example with no image is worthless.

A policy with no `stop` in its action space cannot decline. This is the
type the corpus is thinnest on — 2 records, both unusable — and the one
whose absence is most dangerous in deployment.

---

## What is not an example

- **`verify` and `na` records.** They annotate what happened; there is no
  action to predict. 39 of 119 records on 4 Sep. Keep them for episode
  structure, but they do not count toward yield. Under schema v3 a look
  becomes a `read` **action** and *does* become an example — "decide to
  read before acting" is a real behaviour.
- **Tool calls, today.** `check_job_seen`, `open_run` and
  `capture_user_intent` record what a lookup *returned*, never that the
  lookup was *chosen*. Until `tool_call` is an emittable action with
  `name` + `args`, a model can use a score it is handed but cannot learn
  to ask for one.
- **`_task_start` / `_task_end`.** Episode scaffolding. Valuable for
  filtering trajectories at training time; not pairs.

---

## The capture contract

One `observation` block per record, shared by every example type. This is
the change that moves yield off zero.

```json
"observation": {
  "url": "...", "title": "...",
  "screenshot": "screenshots/....png",
  "screenshot_dims": {"width": 1536, "height": 639},
  "viewport":        {"width": 1536, "height": 639},
  "scale": 1.0,
  "scroll": {"x": 0, "y": 1240},
  "page_dims": {"width": 1536, "height": 4820},
  "a11y": "pages/<hash>.a11y.yaml",
  "page_text": "pages/<hash>.txt",
  "content_hash": "7f3a1c9e"
}
```

`content_hash` dedupes: most steps in a form share a page state that
changes only locally, so the files are stored once and referenced many
times. Full field semantics and validation rules are in
`trace-schema-v3.md` §1 and §8 — this file says *why each field is needed*;
that one says what it must contain.

**Log every action, including scrolls and reads.** Not for completeness —
because transitions come free only if the action sequence has no gaps. A
skipped scroll makes the next click appear to happen from the wrong scroll
position, which is the hardest error to detect after the fact. On 4 Sep the
job episode logged 2 of ~15 real interactions.

---

## Step catalog, by what it trains

Task model, `task_id` conventions and the closed step vocabulary are
unchanged — see `trace-schema-v3.md` and `browser-playbook.md` §9. What
changes is why each step is worth capturing.

| Step | Example type | Notes |
|---|---|---|
| `capture_user_intent` | — (context) | supplies `goal` for every downstream pair |
| `open_run` | tool use¹ | quota/config gate the whole session |
| `open_linkedin`, `verify_session` | stop / grounding | `verify_session` is the happy-path twin of `detect_abort_signal` |
| `open_jobs_tab` | grounding | siblings in `candidates` — near-identical nav targets |
| `type_search_query` | grounding | keyword is safe in `action_value` |
| `apply_filter` | grounding | one record per filter; the pill row re-renders and invalidates refs, so retry chains land here |
| `search_jobs` | decision | `result_count` from the page body, never the title badge |
| `paginate_results` | decision | "the first page had nothing and I kept looking" is a behaviour |
| `screen_job_candidate` | **judgement** | needs the card bbox *and* the card text — see §2 |
| `open_job_card` | grounding | `candidates` matter most here: ~25 near-identical targets |
| `check_job_seen` | tool use¹ | gates whether the job is opened at all |
| `open_job`, `read_job_description` | decision | the JD is the input to §3 — it must be in the record |
| `score_job` | — | deterministic CLI output, not a behaviour |
| `record_score_decision` | **decision** | the branch itself; log it for jobs that go no further too |
| `detect_external_ats` | decision | routine expected outcome, not an error |
| `find_easy_apply_button` | grounding | `Save` sits beside it; a second "Easy Apply" label sits lower on the page |
| `verify_no_error` | stop | error-state recognition |
| `upload_resume` | grounding + field fill | |
| `fill_screening_field`, `select_work_auth` | **field fill** | one per question, right after fill+blur |
| `harvest_hiring_team` | judgement | |
| `click_review` | grounding | Follow-company checkbox state belongs here |
| `submit_application` | grounding | `result.confirmation_text` is the only proof the outcome was seen |
| `close_confirmation_dialog` | grounding | the dialog is **not in the a11y tree** — a pixels-only example, and a good one |
| `detect_abort_signal` | **stop** | attach a screenshot even though `verify` doesn't require one |

¹ becomes a real example only once `tool_call` is an action kind.

---

## Failure and refusal are the scarce class

4 Sep, across 273 records: **4 retries, 1 `error_shown`, 7 `blocked`.**
Every documented recovery path is at or near zero.

Three mechanics, none requiring anything artificial:

1. **`expected_effect`** — one short string written *before* acting.
   Comparing it to the next record's observation labels failures
   automatically. This is the only realistic way to fix the retry drought.
2. **Never discard a failed attempt.** A stale ref or a red validation
   error is `attempt: 1, outcome: "error_shown"` followed by `attempt: 2,
   outcome: "success"` — two records, same `step` and `task_id`. When
   backfilling one of them, pass `step_index` to target it: `(task_id,
   step)` alone is ambiguous exactly where steps repeat.
3. **Log refusals as `stop` / `ask_human` actions**, never as an absence.

`outcome` currently has no value for a *mechanical* action failure — a
stale element reference is not `success`, `error_shown`, `blocked` or `na`.
Until v3 adds one, use `error_shown` and put the detail in `result`.

---

## Before a run

- [ ] Trace logging on (`jobagent trace steps` reports it).
- [ ] Know which example types this run is expected to produce. A run that
      will only fill forms produces no judgement or stop examples, and
      that is a reason to plan a different run, not to skip the check.

## During a run

- [ ] Every action logged, scrolls and reads included — gaps corrupt
      transitions silently.
- [ ] Every actionable record carries the full `observation` block.
- [ ] Every judgement carries the card's bbox **and** its text.
- [ ] Every decision carries the JD text it was made from.
- [ ] **Every claim in a verdict resolves to the observation or a
      `retrieval` entry.** Nothing recalled. `needs_lookup` when the fact
      isn't available.
- [ ] Every override carries a real `override_reason`.
- [ ] Every failed attempt kept, paired with its recovery.
- [ ] **Written one record at a time, as each step happens.** Records
      sharing a single timestamp were written from memory afterwards — the
      screenshot for each step no longer exists by then, which is why every
      batched screening record in this corpus has none. Check clusters, not
      the span of the episode: run 17 wrote 9 records in one second inside a
      373-second task, and a span-based check waved it through.

## After a run

```bash
python tools/audit_training_yield.py data/traces/<day>/<session>.jsonl data/traces/screenshots
```

- [ ] Yield is above zero. **Record count is not yield** — 119 records
      scoring 26/26 on field conformance produced 0 usable examples.
- [ ] Yield per example type is non-zero for the types the run exercised.
- [ ] `jobagent trace stats` → `off_vocabulary_steps` empty,
      `never_logged_steps` contains nothing the run actually did.

---

## Status

Schema is still 2; the `observation` and `action` blocks specified in
`trace-schema-v3.md` are **not implemented**. Until they are, every day
captured scores zero yield by construction, and the corpus accumulates
episode structure and override provenance — genuinely useful for
filtering and for eval replay — but no trainable pairs.

The honest summary: the corpus today is a well-formed record of what the
agent did. It is not yet a dataset.
