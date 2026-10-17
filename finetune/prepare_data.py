"""Combine every annotated `*.steps.jsonl` corpus into one fine-tune train/val split.

Shells out to `python -m annotation_suite export` per file (the same command
a human would run from that folder's own README) -- once for `--format
messages` (the rendered text turns) and once for the default `--format
steps` (which still carries `observation.screenshot`, after the annotator's
own `screenshot_removed` edits have been applied) -- then joins the two on
(episode_id, step_index) so a kept record's real screenshot, if one survived
review, gets embedded as an OpenAI-style `image_url` data-URI block instead
of collapsing to a text path. Same content-block shape
`orchestrator/db.py:to_api_messages` already uses for the live screenshot
tool, so training data matches what the model sees at inference time.

Concatenates every file's kept rows, then shuffles and splits. Files with no
sidecar `<name>.annotations.jsonl` are skipped -- nothing has been reviewed
in them yet, so there's nothing to train on.

Usage:
    python finetune/prepare_data.py "data/sft/**/*.steps.jsonl"
    python finetune/prepare_data.py data/sft/2026-09-14/00-18-33-harvest.steps.jsonl --min-reward 1

Run again whenever a new annotation_suite pass finishes -- it re-derives
train.messages.jsonl / val.messages.jsonl from whatever *.steps.jsonl +
*.annotations.jsonl pairs exist at that point, so the fine-tune set grows
with the corpus instead of drifting from it.
"""
import argparse
import base64
import glob
import json
import mimetypes
import random
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Minimum kept-good examples for a `kind` before it's flagged as thin
# coverage in the summary — same threshold implied by finetune/README.md's
# by-hand breakdown, just computed instead of eyeballed.
THIN_KIND_THRESHOLD = 5

THINKING_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.DOTALL)
ACTION_RE = re.compile(r'<action kind="([^"]*)">\s*(.*?)\s*</action>', re.DOTALL)


def _assistant_text(row: dict) -> str:
    content = row["messages"][2]["content"]
    if isinstance(content, str):
        return content
    return next((b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"), "")


def _write_jsonl(path: Path, rows: list[dict], strip_metadata: bool) -> Path | None:
    """Write `rows` to `path`. With `strip_metadata`, each line is reduced
    to exactly `{"messages": [...]}` -- the shape a hosted fine-tuning API
    or a strict trainer expects, since it typically rejects unknown
    top-level keys -- and each row's `meta` (episode_id, step_index, kind,
    reward, ...) is written instead to a same-order sidecar,
    `<path>.meta.jsonl`. This project's own trainer (the Colab notebook)
    only ever reads `row["messages"]`, so `meta` is harmless left inline;
    this is for exporting somewhere stricter. Returns the sidecar path, or
    None when `strip_metadata` is False."""
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            line = {"messages": r["messages"]} if strip_metadata else r
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    if not strip_metadata:
        return None
    meta_path = path.with_name(path.name + ".meta.jsonl")
    with meta_path.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            meta = {"line": i, **(r.get("meta") or {})}
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return meta_path


def validate_row(row: dict) -> list[str]:
    """Structural checks for one exported (system, user, assistant) record —
    this pipeline's version of a general agent-dataset validator, adapted to
    the `<thinking>...</thinking>` + `<action kind="...">...</action>`
    output shape the local vision model is trained to produce (see
    finetune/README.md). A row failing this is dropped rather than shipped,
    the same way `export_all` in orchestrator/finetune.py never silently
    includes a structurally broken trajectory."""
    errors = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        return [f"expected exactly 3 messages (system, user, assistant), got "
                f"{len(msgs) if isinstance(msgs, list) else type(msgs).__name__}"]
    roles = [m.get("role") for m in msgs]
    if roles != ["system", "user", "assistant"]:
        errors.append(f"unexpected role order: {roles}")
        return errors  # can't reliably pull the assistant turn below

    text = _assistant_text(row)
    if not THINKING_RE.search(text):
        errors.append("assistant turn has no <thinking>...</thinking> block")
    m = ACTION_RE.search(text)
    if not m:
        errors.append('assistant turn has no <action kind="...">...</action> block')
    else:
        if not m.group(1).strip():
            errors.append("<action> has an empty kind attribute")
        if not m.group(2).strip():
            errors.append("<action> body is empty")
    return errors


def has_annotations(steps_path: Path) -> bool:
    name = steps_path.name
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return steps_path.with_name(name + ".annotations.jsonl").exists()


def export_one(steps_path: Path, tmp_dir: Path, fmt: str) -> Path:
    # No --min-reward here: annotation_suite's own filter only drops a record
    # whose reward is *below* the threshold, and a never-reviewed record has
    # reward=None (not 0), so it slides straight through that filter. We do
    # the reward check ourselves below, where we can also require reward is
    # not None -- i.e. a human actually scored it, not just "wasn't bad enough
    # to drop."
    out_path = tmp_dir / f"{steps_path.stem}.{fmt}.jsonl"
    cmd = [sys.executable, "-m", "annotation_suite", "export", str(steps_path),
           "--format", fmt, "--out", str(out_path)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return out_path


def image_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def embed_image(row: dict, steps_by_key: dict) -> bool:
    """Mutate `row`'s user turn in place to add the record's screenshot, if
    it has one that survived annotation review. Returns whether it did."""
    meta = row["meta"]
    key = (meta.get("episode_id"), meta.get("step_index"))
    steps_rec = steps_by_key.get(key)
    if not steps_rec:
        return False
    shot = (steps_rec.get("observation") or {}).get("screenshot") or {}
    if not shot.get("exists"):
        return False
    abs_path = shot.get("abs_path") or shot.get("path")
    if not abs_path:
        return False
    data_uri = image_data_uri(Path(abs_path))
    if not data_uri:
        return False
    user_msg = row["messages"][1]
    assert user_msg["role"] == "user"
    # annotation_suite's own text rendering already names the screenshot file
    # as a "Screenshot: <local abs path>" line -- with the actual image now
    # attached below, that line is both dead weight and a leaked local path,
    # so swap it for a plain pointer instead of leaving it in the text.
    text = re.sub(r"^Screenshot: .*$", "Screenshot: attached below.", user_msg["content"],
                  count=1, flags=re.MULTILINE)
    user_msg["content"] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="+",
                     help="glob(s) for *.steps.jsonl corpora, e.g. \"data/sft/**/*.steps.jsonl\"")
    ap.add_argument("--min-reward", type=int, default=1,
                     help="drop annotated records below this reward (default 1 -- keeps only kept-good examples)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out-dir", default="finetune/data")
    ap.add_argument("--strip-metadata", action="store_true",
                     help="write bare {'messages': [...]} lines (for a hosted API/strict trainer "
                          "that rejects unknown keys); each row's 'meta' goes to a same-order "
                          "<file>.meta.jsonl sidecar instead")
    args = ap.parse_args()

    files = sorted({Path(p) for pat in args.patterns for p in glob.glob(pat, recursive=True)})
    if not files:
        sys.exit(f"no files matched: {args.patterns}")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_exports"
    tmp_dir.mkdir(exist_ok=True)

    rows = []
    total_images = 0
    for f in files:
        if not has_annotations(f):
            print(f"skip (no annotations yet): {f}")
            continue
        messages_export = export_one(f, tmp_dir, "messages")
        with messages_export.open(encoding="utf-8") as fh:
            all_rows = [json.loads(line) for line in fh if line.strip()]
        # Only records a human actually scored (reward is not None) and that
        # scored at or above the bar -- an unreviewed record's reward is
        # None, never 0, so this is a stricter (and correct) check than the
        # exporter's own --min-reward flag.
        reviewed = [r for r in all_rows if r["meta"].get("reward") is not None]
        file_rows = [r for r in reviewed if r["meta"]["reward"] >= args.min_reward]

        n_images = 0
        if any(r["meta"].get("has_screenshot") for r in file_rows):
            steps_export = export_one(f, tmp_dir, "steps")
            with steps_export.open(encoding="utf-8") as fh:
                steps_by_key = {(r.get("episode_id"), r.get("step_index")): r
                                for r in (json.loads(line) for line in fh if line.strip())}
            for r in file_rows:
                if embed_image(r, steps_by_key):
                    n_images += 1

        print(f"{f}: {len(all_rows)} total, {len(reviewed)} reviewed, "
              f"{len(file_rows)} kept (reward >= {args.min_reward}), "
              f"{n_images} with an embedded screenshot")
        rows.extend(file_rows)
        total_images += n_images

    if not rows:
        sys.exit("no reviewed examples found across the given files -- "
                  "run annotation_suite serve on them first")

    valid_rows = []
    dropped_invalid = []
    for r in rows:
        errs = validate_row(r)
        if errs:
            dropped_invalid.append((r["meta"].get("episode_id"), r["meta"].get("step_index"), errs))
        else:
            valid_rows.append(r)
    if dropped_invalid:
        print(f"\ndropped {len(dropped_invalid)} structurally invalid row(s):")
        for episode_id, step_index, errs in dropped_invalid[:20]:
            print(f"  {episode_id}::{step_index}: {'; '.join(errs)}")
        if len(dropped_invalid) > 20:
            print(f"  ... and {len(dropped_invalid) - 20} more")
    rows = valid_rows
    if not rows:
        sys.exit("every reviewed example failed validation -- nothing to write")

    # Split by episode, not by individual row: steps from the same episode
    # share near-identical prior context (each is one point along the same
    # trajectory), so splitting at the row level would routinely put
    # near-duplicate context on both sides of train/val -- the exact
    # leakage a fine-tuning dataset split is supposed to prevent.
    by_episode: dict = {}
    for r in rows:
        by_episode.setdefault(r["meta"].get("episode_id"), []).append(r)
    episode_keys = sorted(by_episode, key=lambda k: (k is None, k))
    random.Random(args.seed).shuffle(episode_keys)
    n_val_eps = max(1, int(len(episode_keys) * args.val_frac)) if len(episode_keys) > 1 else 0
    val_keys = set(episode_keys[:n_val_eps])
    val_rows = [r for k in episode_keys if k in val_keys for r in by_episode[k]]
    train_rows = [r for k in episode_keys if k not in val_keys for r in by_episode[k]]

    train_path = out_dir / "train.messages.jsonl"
    val_path = out_dir / "val.messages.jsonl"
    _write_jsonl(train_path, train_rows, args.strip_metadata)
    _write_jsonl(val_path, val_rows, args.strip_metadata)

    kind_counts: dict[str, int] = {}
    for r in rows:
        kind = r["meta"].get("kind") or "(unknown)"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    thin_kinds = {k: n for k, n in kind_counts.items() if n < THIN_KIND_THRESHOLD}

    print(f"\ntotal examples: {len(rows)}  (with an embedded screenshot: {total_images})")
    print(f"episodes: {len(episode_keys)}  (val: {len(val_keys)}, train: {len(episode_keys) - len(val_keys)})")
    print(f"train: {len(train_rows)} -> {train_path}")
    print(f"val:   {len(val_rows)} -> {val_path}")
    if args.strip_metadata:
        print(f"  (metadata stripped -> {train_path}.meta.jsonl, {val_path}.meta.jsonl)")
    print("\nby action kind: " + ", ".join(f"{k}={n}" for k, n in sorted(kind_counts.items(), key=lambda kv: -kv[1])))
    if thin_kinds:
        print(f"thin coverage (< {THIN_KIND_THRESHOLD} examples), won't generalize yet: "
              + ", ".join(f"{k}={n}" for k, n in sorted(thin_kinds.items())))
    if len(rows) < 300:
        print("\nsmall corpus -- treat this as an initial/format-adherence pass, "
              "not a final fine-tune. See finetune/README.md.")


if __name__ == "__main__":
    main()
