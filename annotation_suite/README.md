# annotation_suite

A standalone data-annotation tool for turning a harvested `jobagent-sft/1`
corpus into fine-tuning data. It lives in its own folder deliberately — it
does not import anything from `jobagent/` and knows nothing about LinkedIn;
its only contract is the record shape documented in
`../plugin/references/harvest.md` (`schema`, `episode_id`, `step_index`,
`task`, `observation`, `thinking`, `rationale`, `action`, `result`). Point it
at any `*.steps.jsonl` with that shape.

No dependencies beyond the Python standard library and, optionally, the
`claude` CLI on `PATH` for backfilling reasoning.

## Run it

```powershell
python -m annotation_suite serve   data\sft\2026-09-06\14-32-05-harvest.steps.jsonl
python -m annotation_suite export  data\sft\2026-09-06\14-32-05-harvest.steps.jsonl --format messages
python -m annotation_suite stats   data\sft\2026-09-06\14-32-05-harvest.steps.jsonl
```

(equivalently `python annotation_suite\annotate.py serve <file>` — it also
runs as a plain script)

`serve` opens a browser UI to walk the corpus task by task: edit any field
worth correcting, mark a step kept or dropped, reward it -2..2, classify it
(example type / provenance), and backfill a missing `thinking` block by
shelling out to `claude -p ... --output-format json --restricted` — one
record at a time or in a bulk pass. `export` applies everything and writes
`<file>.annotated.<format>.jsonl` plus a manifest, with no UI needed for a
scripted pull.

Nothing is overwritten in place: every change is appended as one line to
`<file>.annotations.jsonl` next to the input, replayed newest-line-wins on
load — a full audit trail, safe to stop and resume.

Screenshots referenced by a record are resolved relative to a guessed
`data/traces/` folder (derived from the corpus path's `data/sft/<day>/...`
structure); pass `--traces-dir` if the corpus was copied somewhere else.

See `annotate.py`'s module docstring for the full design rationale.
