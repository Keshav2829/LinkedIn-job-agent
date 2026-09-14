# Trace schema v3 — a spec for training data, not an audit log

**Status:** implemented 5 Sep 2026. `TRACE_SCHEMA_VERSION` is 3;
`cmd_trace`'s `log` action in `jobagent/cli.py` enforces the §8 validation
below (`observation`/`action` required per the rules there, `action.kind`
closed to the §2 vocabulary, `tool_call` requires `tool.name`/`tool.args`,
`_task_end` requires a final `observation`, `outcome` must be a string).
`plugin/references/browser-playbook.md` §9 is the day-to-day summary for
callers; this file stays the reference for *why* each field exists. Not yet
done: `tools/audit_training_yield.py` (and the other `tools/audit_*.py`
scripts) still read the old flat v2 shape only — they will under-report a
schema-3 day's real yield until they're updated to read `action`/
`observation` too. Treat their output as valid for v2/v1_legacy records and
misleading for v3 ones until that follow-up lands.

Schema 2 made records *valid*. It did not make them *learnable*. This spec
is the change from a corpus a human can audit to a corpus a policy can be
trained on. Read `data-generation-checklist.md` for what each example type needs;
this file is what the record itself has to become.

---

## Why v2 is not enough

Measured against the 199-record corpus (29 Aug – 4 Sep 2026), after the
4 Sep vocabulary fix. None of these are volume problems.

| Finding | Evidence |
|---|---|
| **No observation is stored at all.** `page_text` is in the documented schema and has never been written. No DOM, no accessibility tree. | `page_text` present in **0 / 199** records |
| **The logged action space isn't the real one.** Scrolls, `find`, `read_page`, `get_page_text` are the majority of real interactions and none are actions in the schema. | The 4 Sep job episode logged **2 click/type/select** out of ~15 real browser interactions (6 `verify`, 3 `na`) |
| **No state transitions.** One image per record, no s₍t+1₎, so "did that click work?" is unlearnable. | 20–58 s of unlogged activity between consecutive logged steps in the same episode |
| **Coordinate frames disagree.** Raw `coords` are 2× the image they are attached to. | recorded viewport `1568×744`; **24 of 26** stored screenshots are `784×372` |
| **Scroll position is never recorded**, so a `bbox` cannot be located on the page. | scroll/offset key present in **0 / 199** records |
| **Grounding is sparse even after the fix.** | of 58 actionable steps: `coords` 24, `bbox` 6, `candidates` 5, `viewport` 23 |
| **`instruction` is written after the fact and is the wrong input.** Training (instruction → action) yields an executor that still needs a large model above it to produce the instruction. | every `instruction` string in the corpus |
| **Store lookups are not actions.** `retrieval` records the answer; there is no way to learn *when* to call `job.score`. | `retrieval` in 26 records, `tool_call` action type does not exist |
| **Almost no failure data.** | `attempt > 1`: **2**; `error_shown`: **1**; `blocked`: **7** — out of 199 |
| **Residual label corruption in pre-enforcement rows.** | **7** `_task_end` records carry a dict in `outcome`; `sent_no_note` is off-vocabulary |

**The one-line diagnosis:** v2 records *what was decided*. A policy has to
learn *what was seen, what was done, and what changed*. v2 stores none of
those three faithfully.

---

## The v3 record

Three blocks — `observation`, `action`, `outcome` — plus the episode
scaffolding v2 already has. Everything else is context.

```json
{
  "trace_schema": 3,
  "persona_id": "persona_ab12cd34ef",
  "ts": "2026-09-04T18:53:17Z",
  "day": "2026-09-04",
  "run_id": 15,

  "skill": "linkedin-apply",
  "task_id": "job_4455832919",
  "step": "find_easy_apply_button",
  "step_index": 6,
  "attempt": 1,

  "goal": "Apply to strong-match Easy Apply roles from today's AI Agents Engineer search, capped at the daily quota",
  "plan_step": "Open the Easy Apply modal for the Persistent Systems posting",
  "instruction": "Locate the Easy Apply button, distinguishing it from the adjacent Save button",
  "rationale": "Score 87 cleared the bar and the user approved despite the 5+ yrs line; the button reads Easy Apply so this stays in LinkedIn",

  "observation": {
    "url": "https://www.linkedin.com/jobs/view/4455832919/",
    "title": "Agentic AI Engineer | Persistent Systems | LinkedIn",
    "screenshot": "screenshots/2026-09-04-10372fd9427c.png",
    "screenshot_dims": {"width": 1568, "height": 744},
    "viewport":        {"width": 1568, "height": 744},
    "scale": 1.0,
    "scroll": {"x": 0, "y": 0},
    "page_dims": {"width": 1568, "height": 4820},
    "a11y": "pages/2026-09-04-7f3a1c9e.a11y.yaml",
    "page_text": "pages/2026-09-04-7f3a1c9e.txt",
    "content_hash": "7f3a1c9e"
  },

  "action": {
    "kind": "click",
    "ref": "ref_70",
    "element": {"role": "button", "name": "Easy Apply to this job"},
    "coords": {"x": 484, "y": 190},
    "bbox": {"x": 455, "y": 181, "w": 58, "h": 18},
    "candidates": [
      {"role": "button", "name": "Save", "bbox": {"x": 523, "y": 181, "w": 32, "h": 18}},
      {"role": "button", "name": "Show match details", "bbox": {"x": 455, "y": 295, "w": 90, "h": 16}}
    ],
    "expected_effect": "the Easy Apply modal opens over the job page"
  },

  "outcome": "success",
  "result": {"modal_opened": true, "modal_kind": "single_step"},
  "driver": "claude_in_chrome",
  "retrieval": [ ... ]
}
```

Fields inherited unchanged from v2: `skill`, `step`, `task_id`,
`step_index`, `attempt`, `outcome`, `result`, `driver`, `retrieval`,
`confidence`, `notes`, `backfill`, `persona_id`, `run_id`, `ts`, `day`.

---

## 1. `observation` — the block that does not exist today

**Required on every record that follows a page.** This is the single most
important change; without it there is nothing to learn a policy *from*.

| Field | Required | Notes |
|---|---|---|
| `url`, `title` | yes | cheap, and `url` already carries most of the search state |
| `screenshot` | on any record where the page was looked at | **PNG, native resolution, unscaled** — see §4 |
| `screenshot_dims` | with `screenshot` | must equal `viewport` unless `scale` says otherwise |
| `viewport` | yes | the CSS-pixel frame all coordinates live in |
| `scale` | when the stored image is not 1:1 with the viewport | `0.5` means the image is half-size; coordinates stay in viewport pixels |
| `scroll` | **yes** | `{x, y}` document scroll offset. Without it a `bbox` names a place on the screen, not on the page |
| `page_dims` | yes | full scrollable document size — lets a model know how much is off-screen |
| `a11y` | yes | path to the `read_page` accessibility tree for this state |
| `page_text` | yes | path to the `get_page_text` output for this state |
| `content_hash` | yes | hash of `a11y` + `url`; identical consecutive states point at the same files instead of duplicating them |

**Why both `a11y` and `page_text`, and both plus the screenshot.** The
4 Sep run produced a state where they disagree: LinkedIn's post-submit
upsell dialog rendered on screen and was **absent from the accessibility
tree**, so `find` returned "no confirmation dialog" and it had to be
dismissed by raw coordinate. A model trained only on refs would hang
there forever; one trained only on pixels can't use refs at all. Store
both and the disagreement itself becomes a training example.

**On volume:** `content_hash` deduping matters more than it looks — the
majority of steps in a form-filling episode share a page state that
changes only locally. Store the files once, reference them many times.

---

## 2. `action` — the model's real action space

v2's `action_type` (`click|type|select|verify|navigate|na`) describes
what a *reviewer* cares about. `verify` and `na` are not actions; they
are annotations. A policy emits something on every turn, including
turns where all it does is look.

```json
"action": { "kind": "...", ... }
```

| `kind` | When | Key fields |
|---|---|---|
| `click` | any click | `ref` and/or `coords`, `bbox`, `element`, `candidates` |
| `type` | typing into a field | plus `value` (never salary/phone/PII) |
| `select` | a `<select>` / dropdown choice | plus `value` |
| `key` | Return, Escape, Tab | `value` = the key sequence |
| `scroll` | **every scroll, including the ones that felt incidental** | `direction`, `amount`, and the resulting `observation.scroll` on the next record |
| `navigate` | URL change | `value` = destination URL |
| `read` | `get_page_text` / `read_page` / `find` — reading *is* an action, it consumes a turn and the model must learn when it's worth one | `query` (for `find`), `returns` (count of matches / chars) |
| `tool_call` | any `jobagent` CLI call | `tool.name`, `tool.args`, `tool.result` — see §5 |
| `ask_human` | the run needs an answer it does not have | `question` |
| `stop` | terminal: done, blocked, or refusing | `reason` |

`ask_human` and `stop` are load-bearing. A policy with no `stop` in its
action space cannot decline, and the abort trajectories in §0 of the
playbook become unlearnable no matter how many captchas it sees.

**`expected_effect`** (short string, written *before* acting) is cheap
and pays for itself: comparing it to the next record's observation
generates failure labels automatically, which is the only realistic way
to fix the 2-retries-in-199 problem.

**Every action gets a record.** The 4 Sep job episode logged 2 of ~15
real interactions. Two modal scrolls, a `scroll_to`, a "Show all" click,
7 `find` calls, 3 `read_page` and 5 `get_page_text` left no trace. Those
gaps are not noise around the interesting steps — for a policy that has
to decide *whether to scroll or click*, they are most of the signal.

---

## 3. Transitions come free — if every action is logged

v3 deliberately does **not** add an `observation_after` block. Within a
`task_id`, records are ordered by `step_index`, so record *n*'s
`observation` is the next state after record *n−1*'s `action` — provided
no action happened in between. That proviso is the whole reason §2
insists on logging everything.

The one exception: `_task_end` carries a final `observation` with no
`action`, so the last transition is closed.

```
step_index:  6            7            8
             obs₆         obs₇         obs₈
             action₆  →   action₇  →   action₈
             ╰──── transition ────╯
```

Any gap in the action sequence silently corrupts every transition that
spans it. A skipped scroll makes the next click look like it happened
from the wrong scroll position — which, given §4, is exactly the error
that is hardest to detect later.

---

## 4. One coordinate frame, stated explicitly

The current corpus is unusable for coordinate supervision: `coords` are
recorded in a 1568×744 frame while 24 of 26 images are 784×372, and
`scroll` is absent entirely.

The v3 contract:

1. **All of `coords`, `bbox`, `candidates[].bbox` are in CSS pixels of
   `observation.viewport`**, origin at the viewport's top-left.
2. **Store screenshots at native resolution, as PNG.** JPEG at 0.5 scale
   makes LinkedIn's small text unreadable, and unreadable text is the
   thing a grounding model most needs to read. If a downscaled copy is
   kept for context-window reasons, it is a derived artifact, not the
   stored record.
3. If the stored image is not 1:1, `scale` must say so and coordinates
   still stay in viewport pixels. Never store coordinates in image pixels.
4. `coords_norm` / `bbox_norm` remain auto-derived from `viewport`.
5. **`scroll` plus `page_dims` are mandatory** whenever `coords` or
   `bbox` is set. Document position is `scroll.y + bbox.y`; without
   `scroll` that is not recoverable, and neither is "is the target
   currently on screen?"

The CLI should reject a record carrying `coords`/`bbox` without
`viewport`, `scroll` and `screenshot_dims`, the same way v2 rejects a
click without a screenshot.

---

## 5. Tool calls become actions

Today `retrieval` records what a lookup returned. It never records that
a lookup *was chosen*. So a model can learn to use a score it is handed,
and cannot learn to ask for one.

```json
"action": {
  "kind": "tool_call",
  "tool": {
    "name": "job.score",
    "args": {"job_id": "4455832919"},
    "result": {"score": 87, "status": "queued", "auto_apply": true}
  }
}
```

`retrieval` stays, and keeps its distinct job: **provenance and
override**. `accepted: false` + `override_reason` is the highest-value
supervision in the whole corpus — the 4 Sep run produced 12 overrides,
including the one that matters most, where `job.score` returned 87 /
auto-apply for a posting whose JD demands 5+ years against a ~4-year
profile. That record teaches a model *not to trust its own retrieval*,
which nothing else in the dataset does.

Rule: `tool_call` says the call happened; `retrieval` says whether its
answer was believed.

---

## 6. Goal, plan step, instruction — the hierarchy has to be visible

The deepest problem with v2 is not a missing field. It is that
`instruction` is written by the planner *after* it has already decided,
and then trained as if it were the input.

Train on `(instruction → action)` and you get a competent low-level
executor that still needs a large model to tell it what to do — i.e. not
an independent agent, no matter how much data you collect.

v3 records all three levels so both layers are trainable:

| Field | Scope | Trains |
|---|---|---|
| `goal` | constant for the episode, copied from `_task_start` | — (context) |
| `plan_step` | the current sub-goal, changes a few times per episode | **the planner**: `(goal, observation, history) → plan_step` |
| `instruction` | this one action | **the executor**: `(plan_step, observation) → action` |
| `rationale` | why this action, in one line | reward modelling / eval; optional |

The planner is the layer that makes the model independent, and it is the
layer the current corpus contains no supervision for at all.

---

## 7. Failure and refusal, deliberately farmed

2 retries, 1 `error_shown`, 7 `blocked` out of 199. Every documented
recovery path — captcha, external ATS, `needs_human` skip, validation
error, weekly limit, logged-out session — is at or near zero examples.

Three mechanics, none requiring anything unpleasant:

1. **`expected_effect` mismatch auto-labels failures.** When the next
   observation doesn't match, that's a failure record with no extra work.
2. **Never discard a failed attempt.** A red validation error is
   `attempt: 1, outcome: "error_shown"` followed by `attempt: 2,
   outcome: "success"` — two records, same `step` and `task_id`.
3. **Log refusals as `stop` / `ask_human` actions**, not as an absence.
   The 4 Sep aborted run (`run_14`: logged-out session → `verify_session`
   blocked → `detect_abort_signal` → `_task_end: aborted`) is the shape
   to repeat; it is currently the corpus's only complete abort.

---

## 8. Validation the CLI should enforce at v3

Additions to the v2 rejection list:

- `observation.viewport`, `observation.scroll`, `observation.page_dims`
  required whenever `action.coords` or `action.bbox` is set.
- `observation.screenshot_dims` required with `observation.screenshot`;
  must equal `viewport` unless `scale` is present and consistent.
- `observation.a11y` and `observation.page_text` required on every record
  with an `action.kind` other than `tool_call`.
- `action.kind` closed vocabulary (§2); `verify` and `na` are **removed**
  — a look is a `read`, and a record with nothing to emit shouldn't exist.
- `tool_call` requires `tool.name` and `tool.args`.
- `plan_step` required on every non-reserved step.
- `_task_end` requires a final `observation`.
- `outcome` must be a string (7 dicts slipped through pre-enforcement).

---

## 9. Migration

**Nothing is rewritten and nothing is deleted.** `trace_schema` already
tags every record, and `trace stats` already breaks down `by_schema`.

| Corpus | Rows | Keep for | Do **not** use for |
|---|---|---|---|
| `v1_legacy` (untagged) | 59 | episode grammar, step-vocabulary priors | anything supervised |
| `2` | 140 | episode grammar, `retrieval` overrides (26 records, 12 overrides), task/step ordering, eval replay | behaviour cloning, grounding, coordinate prediction |
| `3` | — | everything | — |

Specifically salvageable from v1/v2, and worth protecting:

- **Episode structure.** 34 of 35 tasks have matching `_task_start` /
  `_task_end`. That's a clean grammar of what a legal episode looks like.
- **The retrieval/override records.** Provenance and `override_reason`
  don't depend on observation quality.
- **The step vocabulary itself**, now closed and enforced.

Specifically dead:

- **All v2 grounding.** 784×372 JPEGs cannot be upscaled, and without
  `scroll` the coordinates cannot be re-anchored. No backfill is
  possible — the page states are gone. Treat v2 `coords`/`bbox` as
  metadata about what happened, never as labels.

**Cut-over: done 5 Sep 2026.** `TRACE_SCHEMA_VERSION` is 3 and §8's
validation is enforced in `cmd_trace`. `trace task`/`trace stats` keep
reading both schemas (`by_schema` separates the two corpora); `stats`'
`grounding_coverage` now also checks the nested `action.kind`/`coords`/
`bbox`/`element` so it doesn't silently zero out on an all-v3 day. Do not
mix schemas in one training pull.

---

## What v3 still does not give you

Worth being honest about the ceiling, so this isn't mistaken for a
finished plan:

- **Behaviour cloning alone plateaus.** A policy trained only on
  successful human-driven episodes drifts as soon as it lands in a state
  no episode covered. Closing that needs on-policy correction (run the
  model, log where it goes wrong, add those states) — which v3 supports
  but does not itself perform.
- **LinkedIn's DOM moves.** The 4 Sep run alone found the results URL
  changed (`/jobs/search-results/`), the filter row became radio pills,
  and `get_page_text` on the results surface returns the list rather than
  the detail pane. Any grounding model has a shelf life; the a11y-tree
  half of the observation is what makes re-training on a changed site
  cheap rather than a restart.
- ~~**`driver` still has no value for the desktop browser pane.**~~ Fixed
  in the same cut-over: `claude_browser_pane` and `cdp` are both in
  `VALID_DRIVERS` alongside `claude_in_chrome`/`browsermcp`. `cdp` is now
  the preferred driver in live runs (see `browser-adapter.md`); the desktop
  browser pane still has zero records.
- **The audit tooling lags the schema.** `tools/audit_training_yield.py`,
  `audit_checklist.py` and `audit_provenance.py` were all written against
  the flat v2 shape and don't know about `action`/`observation` yet — they
  will read a well-formed schema-3 record as if it were an empty one. Fix
  them before trusting their output on a mixed or all-v3 corpus.
