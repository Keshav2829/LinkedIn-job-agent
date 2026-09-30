# orchestrator

A small, local, model-agnostic agent harness that drives the LinkedIn Job
Agent (`../jobagent`) the way Claude Code's `linkedin-agent` subagent does —
by reading the same `plugin/skills/*/SKILL.md` procedures and running the
same `python -m jobagent ...` CLI — except it talks to **any locally hosted
model behind an OpenAI-compatible API** (vLLM, llama.cpp's server, LM
Studio, Ollama's `/v1` shim, ...) instead of Claude, needs no client
integration to run, and keeps its own session history and memory.

**This folder is fully standalone.** It does not import `jobagent`, does
not touch `data/state.db`, and does not modify anything under `plugin/` or
`.claude/`. It only *reads* `plugin/skills/` and shells out to the
`jobagent` CLI, exactly as a human operator would from a terminal.

## Why this exists, and the design decisions behind it

**Problem:** the LinkedIn agent's "brain" (the skills, the CLI, the
allowlisted rules) is client-agnostic by design — Cursor, Claude Code and
Cowork can all run it because it's just files and a CLI. But all three of
those are hosted/paid clients. If you want to run the same cycle against a
model you're serving yourself, on your own hardware, with full control over
cost and data locality, you need a small harness that can hold a
conversation with that model and act on its output. That's all this is.

**Design decisions and why:**

1. **A text-based action protocol, not native OpenAI function-calling.**
   Function-calling only works when the serving stack has a tool-call
   parser wired up for that exact model family (vLLM needs
   `--enable-auto-tool-choice --tool-call-parser <name>`; plenty of local
   setups — llama.cpp's server, a base model, an older vLLM build — don't
   have one). Instead, the system prompt asks the model to end every reply
   with one fenced ` ```action ` block of JSON:
   `{"tool": "...", "input": {...}}` or `{"final": "..."}`. This works
   against *any* `/chat/completions` endpoint, fails loudly and
   debuggably when a model gets the format wrong (instead of a silent
   400), and is exactly the shape of thing you can fine-tune a small local
   model to do reliably — see "Fine-tuning data" below. See `agent.py` for
   the full rationale and the parser.

2. **A small, mostly-read-only tool surface** (`tools.py`): `list_skills`,
   `read_skill`, `read_file`, `memory`, and `run_shell`. `run_shell` is the
   only tool that can change anything, and it's allowlisted to
   `python -m jobagent ...` / `python -m orchestrator ...` plus read-only
   `git status/log/diff/branch` — so a confused or adversarial local model
   cannot turn "run today's search" into something destructive. This
   mirrors the job agent's own philosophy (never guess, never bypass the
   approval queue) rather than reinventing it.

3. **Its own SQLite database** (`orchestrator/data/orchestrator.db`),
   entirely separate from `jobagent/data/state.db`. Three tables:
   `sessions` (one row per run), `messages` (the full transcript — also the
   raw material for fine-tuning export), and `memory` (small key/value
   facts the agent chooses to remember across sessions via the `memory`
   tool — e.g. "the last run ended with two invites still pending" — kept
   separate from the job agent's own answer bank on purpose: this is
   memory about *how to operate*, not the user's job-search profile).

4. **stdlib only.** This project has no dependency manifest anywhere and no
   venv requirement; `orchestrator` keeps that property. `urllib` talks to
   the model server, `sqlite3` is the database, `http.server` serves the
   UI. Nothing to `pip install`.

5. **A session per run, a database per machine**, matching `jobagent`'s own
   convention (see the root `AGENTS.md`) — this is per-machine local state,
   not something to sync or commit.

6. **Screenshots are opt-in and additive, not a second code path.** The
   `screenshot` tool doesn't drive a browser itself — it shells out to
   `jobagent browse shot`, which already owns the CDP connection and the
   viewport-vs-full-page distinction, then reads the resulting PNG back and
   attaches it to the *next* turn as a standard OpenAI `image_url` content
   block. It only exists in the tool list at all when `vision: true` is
   configured, so a text-only local model's system prompt and tool surface
   are completely unaffected by the feature being built. See "Screenshots
   (vision)" below.

7. **Asking a question pauses the session; it never blocks a thread.**
   `interactive: true` lets the model emit `{"ask": "..."}` instead of a
   tool call or `final`. The session's status flips to `waiting_for_user`
   and `drive_session` simply returns — no thread sits waiting on `input()`
   or a socket. Whoever answers (the CLI's `input()` prompt, or a POST to
   the UI's answer endpoint) appends the reply as a plain `user` turn and
   flips the status back to `running`; the *next* call to `drive_session`
   picks the loop back up exactly where it left off, using the session's
   own pinned `interactive`/`vision` flags rather than whatever config that
   later call happens to be holding. Off by default, for the same reason
   `vision` is off by default: a scheduled or unattended run should never
   be able to end up stuck waiting on someone who isn't there. See
   "Interactive Q&A" below.

## Quickstart

1. Serve a model with an OpenAI-compatible API, e.g. with vLLM:

   ```bash
   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
   ```

2. Point the orchestrator at it — either copy `config.example.json` to
   `config.json` and edit it, or use flags/env vars (see Configuration
   below):

   ```bash
   cp orchestrator/config.example.json orchestrator/config.json
   # edit base_url / model to match what you served
   ```

3. Run a task from the project root (the folder containing `jobagent/`):

   ```bash
   python -m orchestrator run "List the available skills, then read linkedin-daily-run and summarize what it does."
   ```

4. Or use the local web UI instead:

   ```bash
   python -m orchestrator ui
   ```

   This opens `http://127.0.0.1:8787/` — a form to start a run, a list of
   past sessions, and a live-updating transcript view.

## Configuration

Precedence, low to high: built-in defaults → `orchestrator/config.json`
(gitignored — your local override, not committed) → environment variables →
explicit CLI flags.

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `provider` | `ORCHESTRATOR_PROVIDER` | `openai` | `openai` (any OpenAI-compatible `/chat/completions` server), `anthropic` (Claude's native Messages API), or `claude-cli` (shells out to the `claude` CLI — no API key needed) — see "Driving a session with Claude" below |
| `base_url` | `ORCHESTRATOR_BASE_URL` | `http://localhost:8000/v1` for `openai`, `https://api.anthropic.com` for `anthropic`, unused for `claude-cli` | Base URL for whichever provider is in effect. Only defaults per-provider when nothing else set it — an explicit value (file/env/CLI) always wins, **even across a `--provider` switch**, so pass `--base-url` alongside `--provider` if your `config.json`/env value was written for a different provider |
| `api_key` | `ORCHESTRATOR_API_KEY` | `EMPTY` | Sent as `Authorization: Bearer <key>` for `openai` (vLLM ignores it by default) or `x-api-key` for `anthropic` (falls back to `$ANTHROPIC_API_KEY` when left at `EMPTY`). Unused for `claude-cli` — it authenticates however `claude` itself is already logged in |
| `model` | `ORCHESTRATOR_MODEL` | `local-model` | Model name exactly as the server expects it: an API model id for `openai`/`anthropic`, or a CLI model alias like `sonnet`/`opus`/`haiku` for `claude-cli` |
| `temperature` | `ORCHESTRATOR_TEMPERATURE` | `0.2` | |
| `max_tokens` | `ORCHESTRATOR_MAX_TOKENS` | `1024` | Per reply |
| `max_steps` | `ORCHESTRATOR_MAX_STEPS` | `25` | Hard cap on tool calls per session |
| `max_consecutive_errors` | `ORCHESTRATOR_MAX_CONSECUTIVE_ERRORS` | `3` | Abort after this many protocol/tool errors in a row |
| `request_timeout` | `ORCHESTRATOR_REQUEST_TIMEOUT` | `120` | Seconds, per LLM call |
| `shell_timeout` | `ORCHESTRATOR_SHELL_TIMEOUT` | `300` | Seconds, per `run_shell` call |
| `vision` | `ORCHESTRATOR_VISION` | `false` | Registers the `screenshot` tool — see "Screenshots (vision)" below |
| `interactive` | `ORCHESTRATOR_INTERACTIVE` | `false` | Lets the model pause and ask a question — see "Interactive Q&A" below |
| `browse` | `ORCHESTRATOR_BROWSE` | `false` | Lets the model drive `jobagent browse ...` directly instead of declining — see "Driving a session with Claude" and `plugin/skills/linkedin-apply/SKILL.md`'s approval gate |
| `send_images` | `ORCHESTRATOR_SEND_IMAGES` | `true` | Whether a stored screenshot is actually sent on replay — not session-pinned, safe to turn off on a resume call; see "Screenshots (vision)" below |
| `effort` | `ORCHESTRATOR_EFFORT` | unset | Reasoning effort for `--provider claude-cli` (`low`/`medium`/`high`/`xhigh`/`max`). Session-pinned like `vision`/`interactive`/`browse` — see below |
| — | `ORCHESTRATOR_REPO_ROOT` | parent of this folder | The LinkedIn agent project root |
| — | `ORCHESTRATOR_DB` | `orchestrator/data/orchestrator.db` | This project's own sqlite file |

`python -m orchestrator config show` prints the effective config (API key
masked).

## CLI reference

```
python -m orchestrator run "<task>" [--finetune] [--vision] [--interactive] [--model M] [--base-url URL] ...
python -m orchestrator run -                      # read the task from stdin

python -m orchestrator sessions list [--limit N] [--json]
python -m orchestrator sessions show <session_id> [--json]
python -m orchestrator sessions answer <session_id> "<answer>"   # resume a paused session
python -m orchestrator sessions retry <session_id>                # resume a failed/aborted/stopped session
python -m orchestrator sessions stop <session_id>                 # stop a running session at its next step
python -m orchestrator sessions delete <session_id> [--delete-screenshots]
python -m orchestrator sessions review <session_id> --reward N [--note "..."]   # -2..2
python -m orchestrator sessions review <session_id> --clear

python -m orchestrator ui [--port 8787] [--no-browser] [--vision] [--interactive]

python -m orchestrator finetune export --out FILE [--all] [--status S ...] [--any-status] [--val-frac F] [--strip-metadata] [--require-review] [--min-reward N]
python -m orchestrator finetune validate FILE

python -m orchestrator config show

python -m orchestrator memory get <key>
python -m orchestrator memory set <key> <value>   # value is parsed as JSON if possible
python -m orchestrator memory list
```

## Driving a session with Claude

`agent.py`'s whole loop — system prompt, the ` ```action ` text protocol,
tool dispatch — is model-agnostic by design (see its docstring): it only
needs something that can hold a chat and follow the instructions to end
every reply with one fenced action block. `--provider anthropic` or
`--provider claude-cli` points that same loop at Claude instead of a local
OpenAI-compatible server, so a *live* session — real `jobagent` output,
real tool errors, real recoveries — can be driven by Claude:

```bash
# with an Anthropic API key:
python -m orchestrator run "run today's job search" \
  --finetune --provider anthropic --model claude-sonnet-5 --api-key "$ANTHROPIC_API_KEY"

# with a Claude Code subscription and no separate API key -- shells out to
# the `claude` CLI (same pattern annotation_suite/annotate.py already uses
# for backfilling reasoning: `claude -p ... --output-format json
# --restricted`), one call per turn:
python -m orchestrator run "run today's job search" \
  --finetune --provider claude-cli --model sonnet
```

`claude-cli` needs `claude` on PATH and already logged in; nothing else to
configure. It has no file/tool access of its own (`--restricted`) — every
step still goes through the orchestrator's own tool dispatch exactly as
with any other provider, Claude just plays the role of the model. One real
limitation: it can't receive image attachments, so a `--vision` session run
this way gets a `[screenshot omitted]` placeholder in place of each
screenshot turn instead of the actual image — use `--provider anthropic`
for a vision-enabled session.

**Reasoning effort is unset by default**, meaning every call inherits
whatever `claude`'s own ambient config says (`~/.claude/settings.json`'s
`modelSettings.<model>.effortLevel`, or its built-in default) — fine
interactively, a real gap for generating training data: the "same" task
could silently run at a different effort level on a different machine, or
after a settings change, with nothing recording which. Pass `--effort` to
make it explicit and reproducible instead:

```bash
python -m orchestrator run "run today's job search" \
  --finetune --provider claude-cli --model sonnet --effort high
```

Pinned to the session exactly like `vision`/`interactive`/`browse` (a
resumed session keeps whatever effort it started with, regardless of what
the resuming call's config says), and recorded in the exported record's
`effort` field when set, so a training example's provenance includes what
it was actually generated at.

The point of doing this is distillation: Claude follows the protocol far
more reliably than the small local model this project is meant to end up
running on, so its sessions make a much better training set than that local
model's own attempts at itself. Flag them with `--finetune` as usual, then
export and fine-tune a local model on them exactly the way the rest of this
README describes — `--provider` only changes who answers each turn, not the
transcript shape, the validation in `finetune.py`, or anything downstream.

## Fine-tuning data

Pass `--finetune` to `run` (or check the box in the UI) to flag a session;
on completion its full transcript is written to
`orchestrator/data/finetune/<session_id>.jsonl` as one
`orchestrator-sft/1`-schema record with a `messages` list in plain OpenAI
chat format. To combine everything flagged so far into one training file:

```bash
python -m orchestrator finetune export --out orchestrator/data/finetune/export.jsonl
python -m orchestrator finetune export --out all.jsonl --all   # every session, flagged or not
python -m orchestrator finetune export --out export.jsonl --val-frac 0.1   # + export.val.jsonl
python -m orchestrator finetune validate orchestrator/data/finetune/export.jsonl
```

`export` only includes `completed` sessions by default — a `failed` /
`aborted` / `max_steps_reached` session never reached a real final answer,
so it's excluded unless you pass `--status <s>` (repeatable) or
`--any-status` to deliberately build a negative/error-recovery slice. Every
included session is checked (`finetune.validate_trajectory`) before it's
written — a structurally broken one is skipped and reported, never silently
shipped — and the printed summary breaks kept sessions down by trajectory
type (single tool call, multi-step, no tool needed, clarification, error
recovery, multi-turn), so a lopsided export is visible instead of assumed.
`--val-frac` splits by each session's literal task string, not by session
row, so two retries of the same task can't land on both sides of the split.
`finetune validate` re-checks an already-exported file (role ordering, valid
JSON, ends on a real `final` action) — useful after hand-editing one or
combining files from another machine.

Every exported line also carries `schema`/`session_id`/`task`/`model`/
`status`/`step_count`/`started_at`/`ended_at` alongside `messages` — this
project's own trainer (the `finetune/` Colab notebook) only ever reads
`messages`, so those extra keys are harmless there and useful for tracing a
training example back to its session. A hosted fine-tuning API typically
*rejects* records with unknown top-level keys, though, so `--strip-metadata`
writes bare `{"messages": [...]}` lines instead and moves everything else to
a same-order `<file>.meta.jsonl` sidecar.

Unlike `../annotation_suite/`'s per-step review of the browser-trace
corpus, there's no *required* annotation pass here — either the model's
` ```action ` block was valid and got acted on, or the transcript shows
the protocol-error observation it got instead, which is itself useful
negative-example signal, and `validate_trajectory` already catches
anything structurally broken. What that doesn't catch is *quality* — a
session can be perfectly well-formed and still not be something you'd
want to train on. See "Reviewing sessions" below for that.

## Reviewing sessions

Structural validation proves a session is well-formed; it says nothing
about whether it was actually a *good* demonstration. Review is a human
judgment call, at two granularities:

- **Whole-session** — what export actually filters on, since that's
  finetune.py's real unit (a complete trajectory, not individual steps —
  see its module docstring).
- **Per-step** — one reward per message/turn, a finer-grained diagnostic
  layer for catching a bad turn inside an otherwise-fine session, closer
  in spirit to how `annotation_suite` reviews the browser-trace corpus
  (though that reviews independent per-step *training pairs*; this is
  still one whole session at export time regardless of its step rewards).

Both use the same `-2` unusable … `2` exemplary scale `annotation_suite`
uses, for consistency across the two pipelines. Only an `assistant` turn
can be scored per-step — a system/user/tool row is context the model was
*given*, not something it decided, so there's nothing to grade; scoring
one is rejected (`ValueError` / HTTP 400) both from `db.set_message_review`
directly and from the CLI/UI on top of it.

**Review, export and import all live on the dedicated `/annotate` page**
(linked from the main page's header), not the main run/session page —
that one is just for starting and watching runs. Pick a session there and
its transcript renders as a sequence of step cards, one per message; each
`assistant` one gets its own reward buttons + note + save (a
system/user/tool card renders read-only, labeled "context, not scored"),
with a whole-session verdict bar pinned above them, and the same
Export/Import panels described below. The main page's session list still
shows the whole-session review badge (reward, or "unreviewed") so you can
see status at a glance without switching pages.

From the CLI:

```bash
python -m orchestrator sessions review <session_id> --reward 2 --note "clean, minimal, correct"
python -m orchestrator sessions review <session_id> --clear   # back to unreviewed
python -m orchestrator sessions step-review <session_id> <seq> --reward -1 --note "wrong tool choice"
python -m orchestrator sessions step-review <session_id> <seq> --clear
python -m orchestrator sessions list                          # shows reward/unreviewed per row
python -m orchestrator sessions show <session_id>              # shows [step reward=N (...)] per message
```

Per-step reviews travel with the session on export/import too — see
`step_reviews` in the next section — but they're informational, kept
separate from the `messages` array itself; only the whole-session verdict
above gates what `finetune export` includes. Then gate export on it:

```bash
python -m orchestrator finetune export --out export.jsonl --require-review
python -m orchestrator finetune export --out export.jsonl --min-reward 1   # good or exemplary only
```

`--min-reward` implies reviewed on its own — a never-reviewed session's
`reward` is `NULL`, never `0`, so `reward >= min_reward` alone can't
silently let it through (same class of bug `finetune/prepare_data.py`'s
own `--min-reward` comment calls out for the harvested corpus). Sessions
skipped for being unreviewed or under the reward bar are reported
separately from structurally-invalid ones (`skipped_unreviewed` vs.
`skipped_invalid` in the report / CLI output), since "not reviewed yet"
isn't a defect in the session. The `/annotate` page's Export panel exposes
the same options.

## Importing external session data

The reverse of export: bring an `orchestrator-sft/1` record — one you
exported yourself earlier, one from another machine, or someone else's —
back into `sessions`/`messages` so it can be reviewed and re-exported
alongside locally-generated sessions.

```bash
python -m orchestrator sessions import <file.jsonl> [--overwrite] [--no-finetune-flag]
```

or the `/annotate` page's **Import external session data** panel: pick a
`.jsonl` file, optionally check "overwrite" (replace a session already on
disk under the same id — off by default, a collision is skipped and
reported rather than silently clobbered), and Import.

It's not a lossless round-trip — `to_api_messages` already folded every
`tool` observation into a `role: "user"` turn with an `"Observation
(<name>):\n"` prefix before export, and an image became a base64 data
URI. Import reverses exactly that: a `user` turn matching the prefix
becomes a `tool` row again (`tool_name` extracted from the prefix); an
embedded image is decoded back to a real file under
`orchestrator/data/finetune/imported_images/`. A `user` turn that
doesn't match the prefix — a real user turn, or genuinely external data
that never went through `to_api_messages` — is kept as plain `user`
rather than guessed at. Either way, the reconstructed session goes
through the exact same `validate_trajectory` check a native session does
before it's accepted — imported data is never a second-class, unchecked
citizen in the corpus. Bare `{"messages": [...]}` lines (e.g.
`--strip-metadata` output, or hand-written data) work too; a missing
`session_id`/`task` gets one synthesized. If the record carries
`step_reviews` (see "Reviewing sessions" above), those are restored onto
the matching message rows by `seq`; a record with none just imports with
every step unreviewed, same as a freshly-run session.

Every tool observation in an exported transcript is folded into a `role:
"user"` turn (see `db.to_api_messages`) — this schema has no `role: "tool"`
at all. That means the standard "assistant-only loss" trainer setting (TRL's
`assistant_only_loss=True`, axolotl's `train_on_inputs: false`) is exactly
right out of the box: every `user` row is masked, every `assistant` row
(reasoning + the action block) is trained on. Skipping that flag is the most
common way this kind of data goes wrong — the model learns to hallucinate
tool output instead of waiting for the real observation.

## Screenshots (vision)

Off by default (`vision: false`). Turn it on with `--vision`, `"vision": true`
in `config.json`, `ORCHESTRATOR_VISION=true`, or the checkbox in the UI
(per-run — it overrides the server's default for that one run only), and the
model gets one more tool:

```
screenshot: input {"label": "optional-short-name", "full": false}
```

It shells out to `python -m jobagent browse shot` (the same CDP driver a
human or Claude Code uses — see `plugin/references/browser-adapter.md` for
getting that connected), reads the resulting PNG back off disk, and attaches
it to the *next* model turn as a standard OpenAI `image_url` content block —
not a file path in text. The UI's transcript view renders it inline.

**This only does anything useful with a vision-capable local model** —
Qwen2-VL, Llama-3.2-Vision, Pixtral, LLaVA, and similar served through
vLLM's multimodal support. Pointing `vision: true` at a model/endpoint
that doesn't actually accept image input **does** fail, with an HTTP 400
like `messages contain images, but <model> does not support image
inputs` — and unlike most errors here, it doesn't go away on retry or on
switching models, because the image is now baked into the session's
*stored* history and gets resent on every future turn regardless of who's
serving it (LM Studio quietly loading a text-only build of a model that's
multimodal upstream is a real, easy way to hit this — the base
architecture supporting vision doesn't guarantee the specific
GGUF/quantization being served does).

**To recover a session stuck this way**, pass `--no-send-images` on the
CLI (`sessions answer <id> "..." --no-send-images`, or on `run` itself),
or check "resume without images" in the UI's answer box, or set
`send_images: false` on the `/api/sessions/<id>/answer` call. This is
deliberately *not* pinned to the session the way `vision`/`interactive`/
`browse` are — see `agent.drive_session` — because it only ever removes
what gets sent, never grants a capability, so it's safe to flip on a
resume call even for a session that took its screenshot under a different
config. The stripped turn is replaced with a `[screenshot omitted]` text
note rather than silently vanishing, so the transcript still shows a
screenshot happened there.

Two things worth knowing before you turn `vision` on:

- **Context grows with every screenshot.** Every prior screenshot stays in
  the conversation and gets re-sent (as its full base64 payload) on every
  later turn — there's no eviction. For a long session, watch your model's
  context window; a handful of screenshots on a small local model adds up
  fast. The system prompt tells the model not to call `screenshot`
  speculatively, but it's still the model's call.
- **Needs the same connected browser driver `jobagent` itself needs.** If
  `python -m jobagent browse status` doesn't say `connected: true`, the tool
  returns a clear text error instead of an image — it never fabricates one.

## Interactive Q&A

Off by default (`interactive: false`) — the default matches the LinkedIn
agent's own subagent rule ("you cannot ask questions... you record it and
move on"), which the system prompt states verbatim when this is off. Turn
it on with `--interactive`, `"interactive": true` in `config.json`,
`ORCHESTRATOR_INTERACTIVE=true`, or the checkbox in the UI (per-run,
overriding the server's default for that run only), and the model gets a
third response type alongside a tool call and `final`:

```action
{"ask": "<a question only a person can answer>"}
```

What happens next depends on how you're driving it:

- **`orchestrator run --interactive`** blocks on `input()` right there in
  the terminal — the question is printed, you type an answer, and the same
  `run` command continues as if nothing happened. This is the "attended"
  case: you're watching, so waiting is fine.
- **The UI** can't block a request thread on a browser round-trip, so a
  paused session just sits with `status: waiting_for_user` — nothing is
  consuming a thread or a socket — until an answer box (shown automatically
  in the transcript view) is submitted, which resumes it in a new
  background thread.
- **A closed terminal, or a session you want to answer from elsewhere:**
  `python -m orchestrator sessions answer <session_id> "<answer>"` resumes
  any paused session, from either origin, and continues printing the
  transcript the same way `run` does.

A session's `interactive`/`vision`/`browse`/`effort` **and — the one that
actually matters most — `provider`/`model`/`base_url`** are all pinned at
creation time, not re-read from whatever config happens to drive a later
resume. This is the fix for a real bug: `model`/`base_url` have been
columns on `sessions` since the start, but were never actually read back
into the resuming call before, so `sessions answer <id> "..."` (or the
UI's answer box) run with no flags re-specified silently fell back to
`orchestrator run`'s *default* config instead of continuing with whatever
provider/model actually started the session — surfacing as, among other
things, an HTTP 400 from a completely different (and completely
unexpected) endpoint mid-conversation. Answering a paused session from a
different call site than the one that started it (UI vs. CLI, or a future
scheduled resume) now can't silently change *any* of these — not what the
model is and isn't allowed to do, and not which model it even is.
`api_key` is the one deliberate exception — it's a secret, never stored on
the session row at all, so it still comes from whatever config the
resuming call provides ($ANTHROPIC_API_KEY or `--api-key` on the CLI; the
answer box's own "API key" field in the UI) every time, including a
resume — it does *not* carry over from whatever key started the session.
Forget to resupply it for a session on a real remote provider and you'll
get a clear `LLMError` up front (`llm.py`'s `LLMClient`/`AnthropicClient`
refuse to even make the request while still holding the project's
local-server placeholder `"EMPTY"`) rather than a confusing 401 from the
provider itself.

## Stopping a running session

Any `running` session can be interrupted with `request_stop` — it just sets
a flag (`sessions.stop_requested`) on the row; the session's own
`drive_session` loop, running in whatever thread or process actually
started it, notices the flag itself and ends the session with status
`stopped`. This is deliberately cooperative, not a forced kill: there's no
clean way to abort a blocking `urllib.request.urlopen` call or a tool
already in flight without a much bigger rewrite, so the check only happens
once per step — at the top of the loop, right before what would be the
*next* model call (the same boundary `max_steps` and the consecutive-error
limits already treat as safe). A step already in flight when you hit Stop
still finishes; only the one after it never starts.

- **The UI** shows a Stop button next to each `running` session in the
  sidebar list, and in the transcript header while a running session is
  open.
- **The CLI:** `python -m orchestrator sessions stop <session_id>`.

A `stopped` session is resumable exactly like a `failed`/`aborted` one —
see the next section.

## Retrying (or resuming) a failed, aborted, or stopped session

A session ends `failed` when the model call itself blows up (network
timeout, bad/missing `api_key`, the provider unreachable, ...), `aborted`
after too many *consecutive* protocol/tool errors in a row, or `stopped` by
an explicit stop request — none of these are a normal finish like
`completed`/`max_steps_reached`. All three are resumable in place, same
idea as `waiting_for_user`/`answer_question`, and safe for the same reason:
none of them appends anything partial to the transcript for the step that
didn't complete (a `failed` call raises before `db.add_message`; a
`stopped` session never even starts the step it's stopped on) — so the very
next attempt sends the exact same messages, picking up from exactly where
it broke or paused, no replay of earlier steps.

- **The UI** shows a retry bar (API key field + "resume without images" +
  Retry/Resume button) under the transcript automatically whenever the open
  session's status is `failed`, `aborted`, or `stopped` (labeled "Resume"
  instead of "Retry" for the last one, since nothing actually went wrong).
- **The CLI:** `python -m orchestrator sessions retry <session_id>` (same
  `--api-key`/`--no-send-images`/etc. overrides as `run`, via the shared
  `common` flags).

Like `answer_question`, `api_key` is never pinned to the session row, so it
still has to be resupplied on a retry if the failure was key-related — same
placeholder-vs-remote-host fail-fast applies if you forget.

The system prompt tells the model, when this is on, to ask only when
genuinely blocked on something only a person can decide — not for anything
`jobagent`'s own tools can already answer, and not the same thing twice.
That's a norm the model can ignore, though: a chatty local model can still
turn an unattended-feeling `run` into one that stops and waits. If you want
a run to be guaranteed non-blocking (a cron job, an overnight batch), leave
`interactive` off — the model then gets an error instead of a pause if it
tries, and keeps going.

## Safety notes

- `run_shell` only executes commands starting with `python -m jobagent`,
  `python -m orchestrator`, or read-only `git status`/`log`/`diff`/`branch`.
  Everything else is rejected with an explanation the model can act on.
- The agent never approves or sends outreach on its own — it can only run
  `jobagent` commands, and those already refuse to send anything without
  approval unless the user has explicitly disabled that.
- `read_file` is confined to the project root (path traversal is rejected).
- Every session has a hard step cap (`max_steps`) and aborts after
  repeated protocol or tool errors (`max_consecutive_errors`) so a
  confused model can't loop forever or burn an unbounded number of calls.
- `screenshot` (when enabled) only ever reads a PNG that `jobagent` itself
  just wrote to `data/traces/screenshots/`; it never accepts an arbitrary
  path from the model.
- A `waiting_for_user` pause never blocks a thread or holds a connection
  open — it's just a status flag and a stored question, so a forgotten
  paused session costs nothing but a row in the database. And a session
  can only ever pause on `ask` if it was created with `interactive: true`;
  the flag is pinned to the session, not something a later call can grant.
- `sessions delete` (CLI and the UI's per-row Delete button) removes a
  session's own rows and its fine-tune export file, but leaves referenced
  screenshot files alone by default. They live in `jobagent`'s own shared
  trace store (`data/traces/screenshots/`), and the exact same file could
  be cited by `jobagent`'s own harvest corpus independently of this
  session — deleting them is an explicit, separate opt-in
  (`--delete-screenshots`, or the UI's second confirmation), never implied
  by deleting the session alone.

## Testing

```bash
python -m unittest tests/test_orchestrator_smoke.py -v
```

Runs the full agent loop against an in-process stub HTTP server standing in
for the model endpoint (so it needs no real local model to run), plus unit
tests for the database, the tool allowlist/sandboxing, fine-tune export, the
vision path (screenshot tool gating, a mocked `jobagent browse shot` call,
and a full agent-loop run asserting the image reaches the LLM request as an
`image_url` block), and the interactive path (asking disallowed vs. pausing,
answer-then-resume to completion, the step budget spanning a pause, and
`drive_interactively`'s `on_ask` loop) — no real browser, Chrome, or
terminal needed either.

## Layout

```
orchestrator/
  __main__.py        entry point: python -m orchestrator
  cli.py              argparse commands
  config.py           config loading (defaults / config.json / env / flags)
  db.py               sqlite schema: sessions, messages, memory
  llm.py              minimal OpenAI-compatible chat client (urllib only)
  tools.py            the tool surface + run_shell allowlist
  agent.py            the loop: system prompt, action parsing, dispatch
  finetune.py         session transcript -> JSONL export
  ui.py               stdlib http.server UI backend
  ui_page.html         the UI itself (vanilla HTML/CSS/JS, no build step)
  config.example.json  documents the config.json shape (config.json itself is gitignored)
  data/                gitignored: orchestrator.db, finetune/*.jsonl
```
