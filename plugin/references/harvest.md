# Harvesting a training corpus

`jobagent trace log` records what the agent remembered to write down.
`jobagent harvest` records what actually happened.

The gap is not small. Measured on the two real runs of 5 Sep 2026:

| | |
|---|---|
| Browser actions the agent took | **123** |
| Non-reserved steps it logged in that window | **1** |
| Records `harvest` produces from the same window | **265** (123 of them browser actions) |

Nothing about the agent's discipline changes this. Claude Code already wrote
every one of those actions to disk as it happened — the verbatim command, the
verbatim result, the assistant's own words, a timestamp. `harvest` reads those
files instead of asking the agent to describe itself afterwards.

## Running it

```bash
python -m jobagent harvest
```

With no arguments it finds this project's sessions under `~/.claude/projects/`
by itself (Windows: `%USERPROFILE%\.claude\projects\`, resolved the same way
via `Path.home()` — no flag needed on any OS), joins them to `data/traces/`,
and writes to `data/sft/`.

Both `data/traces/` and `data/sft/` are organised one folder per calendar day, one file (or file-set) per session inside it — `data/traces/<day>/<session-time>.jsonl` for trace logs, `data/sft/<day>/<harvest-time>-harvest.*` for harvest output — so multiple sessions on the same day never commingle or silently overwrite one another.

It does **not** guess the project folder's slug — the slugify rule has changed
between Claude Code versions. Every transcript records its own `cwd`, so the
files are asked which project they belong to.

```bash
# a corpus ready for supervised fine-tuning, browser actions only,
# with the screenshots copied in so the folder is self-contained
python -m jobagent harvest --format messages --kind browse --copy-images

# a specific session, or a folder you copied elsewhere
python -m jobagent harvest --transcript path/to/<session-id>.jsonl
python -m jobagent harvest --transcript path/to/folder/

# --claude-dir overrides where ~/.claude is looked up, for any OS — e.g.
# Claude Code running under WSL keeps its config in the Linux home, not the
# Windows one this same command would otherwise resolve to from PowerShell
python -m jobagent harvest --claude-dir /home/you/.claude
```

`--claude-dir` (or the `CLAUDE_CONFIG_DIR` env var) takes any path your shell
can express, so a WSL config reached from Windows would instead be
`\\wsl$\Ubuntu\home\you\.claude` — that form is WSL/Windows-specific; on
macOS or native Linux the default `~/.claude` already resolves correctly with
no override needed.

Useful flags: `--kind` (repeatable — `browse` for every browser action, or an
exact kind like `browse.click`), `--subagents-only` (the real work happens in
the sub-agent; the parent is mostly orchestration), `--max-chars`,
`--no-redact`, `--out`, `--session`.

## Where the data comes from

```
~/.claude/projects/<slug>/<session>.jsonl        the parent session
~/.claude/projects/<slug>/<session>/
    subagents/agent-*.jsonl                      where the work happened
    tool-results/*.txt                           results too big to inline
data/traces/<day>/<session>.jsonl                skill / step / task_id labels
data/traces/screenshots/*.png                    what the page looked like
```

The transcript knows what was done. The trace knows what it *meant* — which
skill, which step in that skill's vocabulary, which job. Neither is enough
alone, so records are joined by time: `_task_start`/`_task_end` bracket an
episode and everything between them inherits its `task_id`, `skill` and
`goal`; a trace record within 15 seconds of an action also contributes its
`step` and `plan_step`.

## The record

One JSON object per line, `schema: "jobagent-sft/1"`.

```json
{
  "schema": "jobagent-sft/1",
  "episode_id": "agent-a9fdc9335c2093a8d",
  "step_index": 49,
  "ts": "2026-09-05T16:23:55.830Z",

  "source":   { "transcript", "session_id", "is_subagent", "cwd",
                "model", "claude_version", "uuid" },

  "task":     { "prompt", "goal", "skill", "step", "task_id",
                "run_id", "plan_step" },

  "observation": {
    "kind": "result_of:browse.form",     // what produced this state
    "text": "…",                          // verbatim, what the model saw
    "chars": 1452, "truncated": false,    // real length, before any cap
    "url", "title", "viewport", "scroll",
    "screenshot": { "path", "abs_path", "dataset_path", "width", "height",
                    "sha256", "bytes", "exists", "age_steps",
                    "frame_matches_viewport" }
  },

  "thinking": { "state": "stripped", "text": null,
                "placeholder": "<THINKING_STRIPPED>", "generated": false },

  "rationale": "Contact info is prefilled correctly. Proceeding.",

  "action": {
    "tool": "Bash",
    "kind": "browse.click",
    "command": "python -m jobagent browse click --ref b2",
    "args":   { "verb": "click", "ref": "b2" },
    "effect": { "coords": [859,494], "bbox": [832,470,54,48],
                "scroll": [0,0], "viewport": [1034,646],
                "name": "Next", "navigated": false }
  },

  "result": { "ok": true, "error": null, "text": "…", "chars": 287,
              "truncated": false, "from_spill_file": false }
}
```

Four fields deserve their own note.

**`observation.images` is the agent looking at a picture.** The agent reads
its own screenshots back with `Read`, and those tool results are image blocks
with no text at all. Counting only text made 21 of them look like empty
observations, when they are the most informative records in the corpus — the
ones where the model saw the page before deciding. The record stores the file
path rather than a megabyte of base64, and sets `screenshot.from: "read"`
with `age_steps: 0`, because a screenshot the model actually looked at beats
the last one it happened to take.

**`observation.text` is what the model saw, not what the page contained.** The
model never saw the page — it saw a tool result. That result is the input half
of the training pair, so it is stored verbatim rather than as a path to a page
dump. This only became possible with the CDP driver: `browse form` returns
about 900 characters where an accessibility tree returns 100,000.

**`action.effect` carries the coordinates with their frame.** `coords` and
`bbox` are meaningless without `scroll` and `viewport` — a coordinate on its
own names a place on a screen, not a place on a page. The 4 Sep corpus audit
found exactly this defect: coordinates in one frame, screenshots in another.
Records now carry all four together, and
`observation.screenshot.frame_matches_viewport` says out loud whether the
image agrees.

**`thinking.state` is always one of three values.** `stripped` means a
thinking block existed and Claude Code emptied it on write — the model did
reason there, and the words are gone for good. `absent` means there was no
block. `present` means text survived. On the two real runs: 127 stripped, 138
absent, 0 present.

## Filling in the thinking later

`thinking.text` is null wherever the reasoning was lost, and
`thinking.placeholder` holds a token so those records are trivial to select:

```python
rows = [json.loads(l) for l in open("data/sft/<day>/<time>-harvest.steps.jsonl")]
needs = [r for r in rows if r["thinking"]["state"] != "present"]
```

A generation pass has everything it needs to write plausible reasoning — the
goal, the observation, the rationale, the action, and the result that
followed. When it writes one, it must set `thinking.text` **and** flip
`thinking.generated` to `true`. That flag is the only thing keeping
synthesised reasoning distinguishable from real reasoning once the two are
mixed in one file, so nothing should ever set the text without it.

`rationale` is different and should not be overwritten: it is what the model
actually said at the time. Roughly a quarter of actions have one.

## Reviewing, rewarding and backfilling: `annotation_suite`

`tools/view_harvest.py` (below) is read-only — it tells you a corpus is
broken but not how to fix it. `annotation_suite/` (a sibling top-level
folder, not part of `jobagent` — it imports nothing from this package, only
the record shape documented above) is the write side: a browser UI over one
`.steps.jsonl` that a human uses to turn a harvested corpus into an actual
fine-tuning dataset.

```bash
python -m annotation_suite serve data/sft/2026-09-06/14-32-05-harvest.steps.jsonl
```

Per record it can: edit any field worth correcting (`observation.text`,
the `task` labels, `rationale`, `action.command`/`args`, `result.text`/`ok`);
mark a step kept or deleted; reward it -2..2; classify it against the five
example types and the provenance classes `data-generation-checklist.md`
already defines (grounding/judgement/decision/field_fill/stop/scaffolding,
and SELF/CITED/IN_SHOT/EP_SHOT/NO_SHOT) — the two signals that checklist
says the corpus has no way to record on its own; and generate the missing
`thinking` block by shelling out to the `claude` CLI itself
(`claude -p ... --output-format json --restricted`), writing
`thinking.generated: true` and `thinking.author` exactly per the contract
above, one record at a time or in a bulk pass over everything matching the
current filter.

Nothing is overwritten in place. Every change is appended as one line to
`<input>.annotations.jsonl` — the same log-and-join idiom `trace log` and
`harvest` already use — so the review pass has a full audit trail and can be
re-run or resumed safely. `python -m annotation_suite export <file>.steps.jsonl`
applies that log (drops deleted steps, applies edits and rewards) and writes
`<file>.annotated.steps.jsonl` (or `--format messages`) plus a manifest, with
no UI needed for a scripted pull. `python -m annotation_suite stats <file>.steps.jsonl`
prints corpus + annotation counts as JSON. See `annotation_suite/README.md`.

## The `messages` format

`--format messages` emits chat-shaped rows instead:

```json
{"messages": [
   {"role": "system",    "content": "You are a LinkedIn job-application agent…"},
   {"role": "user",      "content": "Goal: …\nURL: …\nScroll: …\n\nLast result …"},
   {"role": "assistant", "content": "<thinking>\n<THINKING_STRIPPED>\n</thinking>\n\nContact info is prefilled correctly. Proceeding.\n\n<action kind=\"browse.click\">\npython -m jobagent browse click --ref b2\n</action>"}],
 "meta": {"kind", "skill", "step", "task_id", "thinking_state", "ok",
          "has_screenshot", "episode_id", "step_index", "session_id"}}
```

The placeholder is emitted rather than dropped, so a filled-in corpus is the
same file with the token replaced, and `meta.thinking_state` lets a training
run weight or exclude those rows.

## Redaction

On by default. Phone numbers, email addresses and salary figures are masked
(`<PHONE>`, `<EMAIL>`, `<SALARY>`) across free text **and** parsed fields —
`action.args.value` and `action.effect.value` are where a typed phone number
would otherwise sit in plain sight. Ten-digit job ids are deliberately kept;
they are what dedupe runs on. `--no-redact` disables all of it.

This is not optional politeness: `AGENTS.md` forbids putting a salary figure
or phone number in the corpus, and the transcript contains every one of them
verbatim.

## Looking at what you harvested

A 265-record JSONL is not something you can eyeball, and the parts most
likely to be broken look fine in a text editor. `tools/view_harvest.py`
renders a corpus as a browsable page and runs the checks:

```bash
python tools/view_harvest.py data/sft/2026-09-06/14-32-05-harvest.steps.jsonl
python tools/view_harvest.py data/sft/2026-09-06/14-32-05-harvest.steps.jsonl --kind browse --only-serious
python tools/view_harvest.py data/sft/2026-09-06/14-32-05-harvest.messages.jsonl --embed
```

It writes `<input>.html` beside the input and prints a tally. Each record
becomes a card: context, the observation (JSON pretty-printed), thinking or
its placeholder, the rationale rendered as markdown, the action, the effect,
and the screenshot inline. Filter by kind, by reasoning state, by task, by
warnings, by session, or search the text. `--embed` inlines the images as
data URIs so the page is one portable file; without it they load from disk.

A single harvest run with no `--session` flag pulls in every session under
the project root (see "Running it" above), so one `.steps.jsonl` routinely
mixes several unrelated conversations together. The **session** dropdown
filters the page down to one of them, keyed on `source.session_id` — always
present in `.steps.jsonl`. The `.messages.jsonl` format only carries it in
`meta.session_id` from this change onward; a `.messages.jsonl` harvested
before it won't offer session filtering.

What it flags, and what it deliberately doesn't: a missing screenshot file, a
screenshot whose size disagrees with the viewport, a stale screenshot, an
empty observation, a click with no coordinates or no frame, an unlabelled
record, a failed action. Missing reasoning is **not** flagged — it is the
expected state of nearly every record, and flagging it matched all 123 and
buried the real problems. It is a filter and a tag instead.

Both formats are accepted; the shape is detected from the first record. The
`messages` format has no `effect` field, so the coordinate checks are skipped
there rather than reported as failures.

## The manifest

Every run writes `<time>-harvest.manifest.json` next to the corpus (in that day's `data/sft/<date>/` folder): record
counts by action kind, the thinking-state breakdown, join statistics,
screenshot resolution counts, and `yield_vs_trace_log` — harvested records
against hand-logged ones over the same time window. Read it before trusting a
corpus; `screenshots.missing` far above `resolved`, or `join.unlabelled` near
the record count, means something upstream moved.
