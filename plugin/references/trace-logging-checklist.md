# Moved → `data-generation-checklist.md`

This file was named wrong, and the name shaped how it was used.

It was never a logging checklist. Its only purpose is to **generate
supervised training data for a small local model** that will take over the
mechanical half of this agent. Read as a logging checklist, it invited a
conformance audit — "does the record have the required fields" — and on
4 Sep 2026 a 119-record day scored **26/26** on exactly that audit while
producing **0 usable training examples**.

The replacement is organised by example type — grounding, judgement,
decision, field fill, stop — and every rule states which of those it
serves and what the model cannot learn without it.

→ **`data-generation-checklist.md`**
→ `trace-schema-v3.md` for the record shape that makes the yield non-zero
→ `tools/audit_training_yield.py` for the measurement that matters

Delete this stub once nothing references it.
