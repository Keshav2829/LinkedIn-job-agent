# finetune

Turns the `annotation_suite`-reviewed corpus under `data/sft/` into a LoRA
fine-tune of the local model (`google/gemma-4-e4b`, per
`orchestrator/config.json`) that's currently driving the LinkedIn agent
poorly. Standalone folder, like `orchestrator/` and `annotation_suite/` —
doesn't import either, just reads `data/sft/` and shells out to
`annotation_suite`'s own CLI.

## Why Colab, not local

`gemma-4-E4B` needs ~12–16GB VRAM for QLoRA (per
[Unsloth's docs](https://unsloth.ai/docs/models/gemma-4/train)). This
machine's card is a 6GB RTX 3060 — enough to *run* the quantized GGUF in LM
Studio, not enough to fine-tune it. `train_gemma4_e4b_qlora.ipynb` runs on
Colab's free T4 (16GB) instead; you bring the resulting `.gguf` back and
drop it into LM Studio.

Worth knowing: this means uploading `train.messages.jsonl` /
`val.messages.jsonl` to Google's Colab servers. `harvest.py` redacts PII
before these records ever land in `data/sft/`, but the page text still
includes job listings, company names and your search goals, and — see below
— some records now carry an actual screenshot of a LinkedIn page, embedded
inline as base64. Reasonable for a personal Colab session, but know that
it's leaving this machine.

## Why the data is small, and why that's correct

`jobagent harvest` captures *every* step from a Claude Code transcript —
thousands of them per real session. `annotation_suite serve` is where a
human actually reviews a step and assigns it a reward; anything never opened
there defaults to `reward: None`, not a passing score. As of writing:

| file | raw steps | reviewed | kept (reward ≥ 1) |
|---|---|---|---|
| `2026-09-11/01-06-56` | 1500 | 33 | 15 |
| `2026-09-11/21-54-21` | 1900 | 152 | 94 |
| `2026-09-14/00-18-33` | 220 | 219 | 200 |
| `2026-09-14/00-45-34` | 220 | 117 | 102 |
| `2026-09-14/18-11-20` | 2981 | 0 | 0 |

**411 reviewed-and-good examples out of ~6800 raw steps.** `prepare_data.py`
filters on `reward is not None` explicitly — `annotation_suite export
--min-reward` alone isn't enough, since a never-reviewed record's reward is
`None`, not `0`, and `None < min_reward` is never true, so unreviewed junk
silently passes a naive `--min-reward` filter. (Caught this while building
the export — the first pass over this folder produced 6711 "kept" examples,
almost all of them never actually looked at by anyone.)

411 examples, and thin per action-kind (some kinds have 1–4 examples total —
see the `by_kind` breakdown `prepare_data.py` prints), is genuinely small.
Treat this as an **initial pass**: it should teach the model the
`<thinking>...</thinking>` + `<action kind="...">...</action>` output shape
and the dominant, well-represented action kinds (`browse.open`,
`browse.click`, `tool.Read`, `browse.find`). It won't have seen enough of
the rare kinds to generalize there. The fix is more annotation rounds, not a
bigger model or fancier training recipe — run `annotation_suite serve` on
more harvested corpora, then redo the two steps below.

## Images

`gemma-4-E4B` is natively multimodal, and the harvested corpus carries real
screenshots (`observation.screenshot`, logged per the "screenshot on every
step" rule in the root `AGENTS.md`). `prepare_data.py` embeds a kept
record's screenshot — if one survived the annotator's own review, since
`annotation_suite serve` lets a human strip a screenshot per record
(`screenshot_removed`) and that decision is respected, never overridden — as
an OpenAI-style `image_url` base64 data-URI block on the user turn. Same
content-block shape `orchestrator/db.py:to_api_messages` already uses when
showing the live model a screenshot from the `screenshot` tool, so training
data matches what the model actually sees at inference time.

Of the 411 kept examples, **22 have an embedded screenshot** — the
`observation.screenshot.exists` flag is true and the annotator didn't strip
it. That's a thin multimodal slice; the notebook trains with the vision
tower frozen (`finetune_vision_layers=False`) for exactly that reason — 22
images isn't enough to safely adapt the encoder itself without overfitting
on them, but it's still enough for the LoRA-tuned language layers to start
getting signal on "here's a screenshot, here's the action." Screenshot
count is the number most worth growing in the next annotation round.

## Run it

1. Build the train/val split from every reviewed corpus:

   ```bash
   python finetune/prepare_data.py "data/sft/**/*.steps.jsonl"
   ```

   Writes `finetune/data/train.messages.jsonl` and `finetune/data/val.messages.jsonl`
   (gitignored — same `data/` rule as everywhere else in this repo, since the
   underlying page text can carry job/company/message content). Re-run this
   any time a new `annotation_suite` review pass finishes; it rebuilds from
   whatever's currently reviewed rather than accumulating drift.

   Every kept-good record is structurally validated first — has exactly the
   3 (system, user, assistant) turns, and the assistant turn actually
   contains both a `<thinking>` block and a non-empty `<action kind="...">`
   block — before it's written; anything that fails is dropped and printed,
   not silently shipped. The train/val split is by **episode**, not by
   individual row: steps from the same episode are near-duplicate context of
   each other, so splitting at the row level would leak the same trajectory
   across both sides. The summary also prints a per-`kind` breakdown and
   flags any kind with fewer than 5 kept examples as thin coverage — the
   automated version of the by-hand table below.

   Each written line is `{"messages": [...], "meta": {...}}` — `meta` carries
   `episode_id`, `reward`, `kind`, and the rest, for tracing an example back
   to its source. The Colab notebook only ever reads `row["messages"]`, so
   this is harmless there; pass `--strip-metadata` if you're exporting
   somewhere stricter (a hosted fine-tuning API typically rejects unknown
   top-level keys) — it writes bare `{"messages": [...]}` lines instead and
   moves `meta` to a same-order `train.messages.jsonl.meta.jsonl` /
   `val.messages.jsonl.meta.jsonl` sidecar.

2. Open `train_gemma4_e4b_qlora.ipynb` in Colab (upload it, or File → Open
   notebook → GitHub if this repo is pushed there), select a T4 runtime, and
   run the cells in order. It'll ask you to upload the two files from step 1,
   trains a LoRA adapter, sanity-checks a handful of held-out examples
   against the model's own output before committing to the slow part, then
   merges and quantizes to a `.gguf`.

3. Download the `.gguf`, drop it into LM Studio's models folder, point
   `orchestrator/config.json`'s `model` field at whatever LM Studio names it,
   and smoke-test with `python -m orchestrator run "List the available
   skills."` before trusting it with a real cycle.

## Layout

```
finetune/
  prepare_data.py              annotated corpus -> train/val messages.jsonl (stdlib only)
  train_gemma4_e4b_qlora.ipynb Colab notebook: QLoRA train -> sanity check -> GGUF export
  data/                        gitignored: train.messages.jsonl, val.messages.jsonl
```
