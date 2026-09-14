"""Score a trace file by TRAINING YIELD, not logging conformance.

The question is not "does this record have the required fields" but "what
supervised example can be constructed from this record, and is its input
actually present?" A record passes only if BOTH sides of the pair exist:
the input the model would see at inference, and the target it must emit.
"""
import json, sys, collections, os, glob

path = sys.argv[1]
shots_dir = sys.argv[2] if len(sys.argv) > 2 else None
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

dims = {}
if shots_dir and os.path.isdir(shots_dir):
    try:
        from PIL import Image
        for f in glob.glob(os.path.join(shots_dir, "*.jpg")):
            dims[os.path.basename(f)] = Image.open(f).size
    except ImportError:
        pass

def shot_dims(r):
    s = r.get("screenshot")
    if not s: return None
    return dims.get(os.path.basename(s))

ACT = ("click", "type", "select")
buckets = collections.Counter()
reasons = collections.Counter()
usable = collections.Counter()

def verdict(r):
    """Return (example_type, usable, reason)."""
    step, at = r["step"], r.get("action_type")

    # ---- 1. GROUNDING: (page state, instruction) -> where to act ----------
    if at in ACT and step != "_task_end":
        if not r.get("screenshot"):
            return "grounding", False, "no screenshot: nothing to ground the coordinate in"
        if not (r.get("coords") or r.get("bbox")):
            return "grounding", False, "no coords/bbox: no target to predict"
        if not r.get("viewport"):
            return "grounding", False, "no viewport: coordinate frame unknown"
        if "scroll" not in r:
            return "grounding", False, "no scroll offset: bbox cannot be placed on the page"
        sd = shot_dims(r)
        if sd and (sd[0] != r["viewport"]["width"] or sd[1] != r["viewport"]["height"]):
            return "grounding", False, (f"frame mismatch: image {sd[0]}x{sd[1]} vs viewport "
                                        f"{r['viewport']['width']}x{r['viewport']['height']}")
        if not (r.get("page_text") or r.get("a11y")):
            return "grounding", False, "no page text / a11y tree: cannot learn ref-based action"
        return "grounding", True, ""

    # ---- 2. JUDGEMENT: (candidate card) -> keep or reject -----------------
    if step == "screen_job_candidate":
        res = r.get("result") or {}
        if not res.get("verdict"):
            return "judgement", False, "no verdict to predict"
        if not (r.get("bbox") or r.get("coords")):
            return "judgement", False, "no bbox: which card on the page was judged is unrecoverable"
        if not (r.get("page_text") or r.get("a11y")):
            return "judgement", False, "card text not stored - only a written summary of it"
        return "judgement", True, ""

    # ---- 3. DECISION: (JD + score + thresholds) -> apply / skip -----------
    if step == "record_score_decision":
        res = r.get("result") or {}
        if not res.get("decision"):
            return "decision", False, "no decision to predict"
        if not r.get("job_description") and not r.get("page_text"):
            return "decision", False, "JD text not in the record (it lives in state.db) - input incomplete"
        return "decision", True, ""

    if step == "score_job":
        return "decision", False, "deterministic CLI output, not a behaviour to clone"

    # ---- 4. FIELD FILL: (question, answer bank) -> value ------------------
    if step in ("fill_screening_field", "select_work_auth"):
        if not r.get("retrieval"):
            return "field_fill", False, "no retrieval: provenance of the value is unknown"
        if not r.get("action_value"):
            return "field_fill", False, "no action_value: nothing to predict"
        return "field_fill", True, ""

    # ---- 5. STOP / ABORT: (page state) -> stop ---------------------------
    if step == "detect_abort_signal" or r.get("outcome") == "blocked":
        if not r.get("screenshot"):
            return "stop", False, "no screenshot: recognising an abort screen by sight is the point"
        if not (r.get("page_text") or r.get("a11y")):
            return "stop", False, "no page state stored"
        return "stop", True, ""

    # ---- 6. TOOL USE: when to call the store -----------------------------
    if step in ("check_job_seen", "open_run", "capture_user_intent"):
        return "tool_use", False, "no action_type for a tool call - the call is not an emittable action"

    # ---- 7. Everything else: annotation, not an example ------------------
    if at in ("verify", "na") or at is None:
        return "annotation", False, "verify/na is a note about what happened, not an action to predict"
    return "other", False, "unclassified"

per_type = collections.defaultdict(lambda: [0, 0])
fail_reasons = collections.defaultdict(collections.Counter)
for r in rows:
    t, ok, why = verdict(r)
    per_type[t][0] += 1
    per_type[t][1] += ok
    if not ok:
        fail_reasons[t][why] += 1

total = len(rows)
tot_ok = sum(v[1] for v in per_type.values())

print(f"{path}  -  {total} records\n")
print(f"{'example type':<12} {'records':>8} {'usable':>8}   why the rest yield nothing")
print("-" * 96)
for t in ("grounding", "judgement", "decision", "field_fill", "stop", "tool_use", "annotation", "other"):
    if t not in per_type: continue
    n, ok = per_type[t]
    print(f"{t:<12} {n:>8} {ok:>8}")
    for why, c in fail_reasons[t].most_common(3):
        print(f"{'':<30}   {c:>4}x  {why}")
print("-" * 96)
print(f"{'TOTAL':<12} {total:>8} {tot_ok:>8}   "
      f"{100*tot_ok/total:.1f}% of records yield a usable training example")

# ---------------------------------------------------------------------------
# PROVENANCE. A usable example needs its input present. A *safe* example also
# needs every claim in its target to be derivable from that input. A verdict
# justified by a fact that is on no page, in no store and in no retrieval
# entry did not come from anywhere the model can reach - it came from the
# annotator's world knowledge, and training on it teaches confident
# fabrication rather than judgement.
# ---------------------------------------------------------------------------
import re
OFF_CARD = {
    "headcount / company size": r"headcount|employees|\bsize\b|small (company|ai startup|startup)"
                                r"|startup|micro-company|~?\d+-person|1-person|under \d",
    "org type / MNC status":    r"\bmnc\b|captive gcc|staffing agenc|consulting intermediar"
                                r"|job board|services firm|mid-size|large mnc",
    "funding / maturity":       r"funded|seed round|series [a-d]\b|scale-?up",
}
judged = [r for r in rows if (r.get("result") or {}).get("reject_reason")]
if judged:
    ung, hits = 0, collections.Counter()
    for r in judged:
        reason = r["result"]["reject_reason"]
        found = [k for k, p in OFF_CARD.items() if re.search(p, reason, re.I)]
        supported = bool(r.get("retrieval") or r.get("page_text") or r.get("a11y"))
        if found and not supported:
            ung += 1
            for k in found: hits[k] += 1
    print(f"\nPROVENANCE  -  {len(judged)} verdicts carry a reject_reason")
    print(f"   {len(judged)-ung:>4} justified only by facts visible on the card")
    print(f"   {ung:>4} justified by facts that are on no page, in no store, and in no")
    print(f"        retrieval entry - i.e. recalled, not derived")
    for k, c in hits.most_common():
        print(f"          {c:>3}x  {k}")
    if ung:
        print("   These are worse than unusable: an SLM trained on them learns to assert")
        print("   company facts it cannot check. The honest verdict when the supporting")
        print("   fact is not available is 'needs_lookup', never 'reject'.")
