"""Audit a day's trace file against plugin/references/trace-logging-checklist.md.

Every check below is a rule literally stated in that file. Reports PASS/FAIL
per rule with the offending records named, so a failure is actionable rather
than a score.
"""
import json, sys, collections
from datetime import datetime

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

VALID_SKILLS = {"linkedin-apply","linkedin-outreach","linkedin-followup",
                "linkedin-inbox","linkedin-setup","linkedin-daily-run"}
VALID_STEPS = {
 "_task_start","_task_end","capture_user_intent","open_run","open_linkedin",
 "verify_session","open_jobs_tab","type_search_query","apply_filter","search_jobs",
 "paginate_results","screen_job_candidate","open_job_card","check_job_seen","open_job",
 "read_job_description","score_job","record_score_decision","detect_external_ats",
 "find_easy_apply_button","verify_no_error","upload_resume","fill_screening_field",
 "select_work_auth","harvest_hiring_team","click_review","submit_application",
 "close_confirmation_dialog","find_connect_button","fill_connect_note","send_invite",
 "open_message_thread","fill_message_body","attach_resume","send_message","read_reply",
 "read_own_profile","detect_abort_signal"}
STEP_OUTCOMES = {"success","error_shown","blocked","na"}
END_BASE = {"applied","sent","not_applied","error","aborted"}
SOURCES = {"user_request","scheduled_daily_run","skill_default"}
ACTIONABLE = ("click","type","select")

results = []
def check(rule, failures, note=""):
    results.append((rule, failures, note))

def rid(r):
    return f"{r.get('task_id')}#{r.get('step_index')}:{r.get('step')}"

tasks = collections.OrderedDict()
for r in rows:
    tasks.setdefault(r.get("task_id"), []).append(r)
for t in tasks:
    tasks[t].sort(key=lambda r: r.get("step_index", 0))

# ---- Golden rules ---------------------------------------------------------
f = []
for t, rs in tasks.items():
    starts = [r for r in rs if r["step"] == "_task_start"]
    ends = [r for r in rs if r["step"] == "_task_end"]
    if len(starts) != 1: f.append(f"{t}: {len(starts)} _task_start")
    if len(ends) != 1: f.append(f"{t}: {len(ends)} _task_end")
    if starts and starts[0].get("step_index") != 0:
        f.append(f"{t}: _task_start at step_index {starts[0].get('step_index')}, not 0")
check("G1  one _task_start and one _task_end per task_id, start at index 0", f)

f = []
for t, rs in tasks.items():
    ts = [datetime.fromisoformat(r["ts"].replace("Z","")) for r in rs]
    if any(ts[i] < ts[i-1] for i in range(1, len(ts))):
        f.append(f"{t}: timestamps not monotonic by step_index")
    # Measuring the WHOLE task span misses the thing it is meant to catch: a
    # batch written after the fact hides inside a task that legitimately ran
    # for minutes. run_17's 9 screening records shared one timestamp inside a
    # 373s task and passed. Look at clusters, not the span.
    clusters = collections.Counter(r["ts"] for r in rs)
    ts_str, n = clusters.most_common(1)[0]
    if n > 2:
        f.append(f"{t}: {n} records share one timestamp ({ts_str}) - written as a batch, "
                 f"not as each step happened")
check("G2  logged as it went (no batch written after the fact)", f)

f = [rid(r) for r in rows
     if r.get("action_type") in ACTIONABLE and r["step"] != "_task_end"
     and not r.get("screenshot")]
check("G3  every click/type/select step carries a screenshot", f)

f = [rid(r) for r in rows
     if r.get("action_type") and r["step"] not in ("_task_start","_task_end")
     and not r.get("driver")]
check("G4  driver set whenever action_type is set", f)

f = []
for r in rows:
    s, o = r["step"], r.get("outcome")
    if s == "_task_start":
        if "outcome" in r: f.append(f"{rid(r)}: _task_start carries outcome")
        if r.get("source") not in SOURCES: f.append(f"{rid(r)}: bad source {r.get('source')!r}")
    elif s == "_task_end":
        ok = isinstance(o,str) and (o in END_BASE or (o.startswith("skipped:") and len(o)>8))
        if not ok: f.append(f"{rid(r)}: bad _task_end outcome {o!r}")
    elif "outcome" in r and o not in STEP_OUTCOMES:
        f.append(f"{rid(r)}: bad step outcome {o!r}")
check("G5  outcome vocabulary closed and correct per record type", f)

# G6: decision-shaping lookups carry retrieval
NEEDS_RETRIEVAL = {"check_job_seen","score_job","open_run"}
f = [rid(r) for r in rows if r["step"] in NEEDS_RETRIEVAL and not r.get("retrieval")]
check("G6  every decision-shaping CLI lookup has its own retrieval entry", f)

f = []
for r in rows:
    for e in (r.get("retrieval") or []):
        if e.get("accepted") is False and not e.get("override_reason"):
            f.append(f"{rid(r)}: accepted:false with no override_reason")
check("G6b every override (accepted:false) carries an override_reason", f)

f = [rid(r) for r in rows if r["step"] not in VALID_STEPS]
check("G9  step names inside the closed vocabulary", f)

f = [rid(r) for r in rows if r.get("skill") not in VALID_SKILLS]
check("G9b skill names inside the closed vocabulary", f)

screened = [r for r in rows if r["step"] == "screen_job_candidate"]
rejects = [r for r in screened if (r.get("result") or {}).get("verdict") == "reject"]
check("G10 rejected candidates logged, not just the ones applied to",
      [] if rejects else ["no reject verdicts logged at all"],
      f"{len(rejects)} rejects / {len(screened)} screened")

# ---- Pre-flight: the session ---------------------------------------------
sess = [t for t in tasks if str(t).startswith("run_")]
f = []
for t in sess:
    steps = [r["step"] for r in tasks[t]]
    for req in ("capture_user_intent","open_run","open_linkedin","verify_session"):
        if req not in steps: f.append(f"{t}: missing {req}")
    if tasks[t][0]["step"] != "_task_start": f.append(f"{t}: does not open with _task_start")
    skills = {r.get("skill") for r in tasks[t]}
    if skills != {"linkedin-daily-run"}: f.append(f"{t}: skill(s) {skills}, expected linkedin-daily-run")
check("P1  session task: run_<id>, linkedin-daily-run, intent+run+linkedin+session logged", f,
      f"{len(sess)} session task(s)")

f = []
for t in sess:
    steps = [r["step"] for r in tasks[t]]
    end = [r for r in tasks[t] if r["step"]=="_task_end"]
    reached_jobs = "open_jobs_tab" in steps
    aborted = end and end[0].get("outcome") == "aborted"
    if not reached_jobs and not aborted:
        f.append(f"{t}: no open_jobs_tab and not closed as aborted")
check("P1b open_jobs_tab logged unless the session aborted before it", f)

# ---- Pre-flight: each search ---------------------------------------------
searches = [t for t in tasks if str(t).startswith("search_")]
f = []
for t in searches:
    steps = [r["step"] for r in tasks[t]]
    if "search_jobs" not in steps: f.append(f"{t}: no search_jobs")
    sj = [r for r in tasks[t] if r["step"]=="search_jobs"]
    for r in sj:
        res = r.get("result") or {}
        if "result_count" not in res: f.append(f"{t}: search_jobs has no result.result_count")
check("P2  every search episode logs search_jobs with a real result_count", f,
      f"{len(searches)} search episode(s)")

f = []
for t in searches:
    steps = [r["step"] for r in tasks[t]]
    if not any(s in steps for s in ("screen_job_candidate",)):
        f.append(f"{t}: results existed but no cards screened")
check("P2b every search screens the cards it looked at", f,
      f"{len(screened)} screen_job_candidate records")

f = [rid(r) for r in rows if r["step"] == "open_job_card" and not r.get("candidates")]
check("P2c open_job_card carries candidates (the cards not clicked)", f)

# ---- Pre-flight: each job ------------------------------------------------
jobs = [t for t in tasks if str(t).startswith("job_")]
f = []
for t in jobs:
    steps = [r["step"] for r in tasks[t]]
    first = [s for s in steps if s != "_task_start"]
    if first and first[0] != "check_job_seen":
        f.append(f"{t}: episode opens at {first[0]}, not check_job_seen")
check("P3  job episode opens at check_job_seen, not the Easy Apply button", f,
      f"{len(jobs)} job episode(s)")

f = []
for t in jobs:
    steps = [r["step"] for r in tasks[t]]
    for req in ("read_job_description","score_job","record_score_decision"):
        if req not in steps: f.append(f"{t}: missing {req}")
check("P3b read_job_description, score_job and record_score_decision on every job", f)

f = []
for t in jobs:
    sj = [r for r in tasks[t] if r["step"]=="score_job"]
    for r in sj:
        srcs = {e.get("source") for e in (r.get("retrieval") or [])}
        if "job.score" not in srcs: f.append(f"{t}: score_job without a job.score retrieval")
check("P3c score_job cites job.score as a retrieval entry", f)

f = []
for t in jobs:
    rd = [r for r in tasks[t] if r["step"]=="record_score_decision"]
    for r in rd:
        res = r.get("result") or {}
        for k in ("decision","next_action"):
            if k not in res: f.append(f"{t}: record_score_decision missing result.{k}")
check("P3d record_score_decision carries decision and next_action", f)

f = []
for t in jobs:
    end = [r for r in tasks[t] if r["step"]=="_task_end"]
    if not end: f.append(f"{t}: no _task_end")
    else:
        sub = [r for r in tasks[t] if r["step"]=="submit_application"]
        o = end[0].get("outcome")
        if o == "applied":
            if not sub: f.append(f"{t}: _task_end applied with no submit_application")
            else:
                res = sub[0].get("result") or {}
                if not res.get("confirmation_text"):
                    f.append(f"{t}: submit_application has no confirmation_text - outcome not seen on screen")
check("G8  _task_end applied only where a confirmation was actually recorded", f)

# ---- Field-level requirements --------------------------------------------
f = [rid(r) for r in rows if not r.get("instruction") and r["step"] != "_task_end"]
check("F1  instruction present on every record except _task_end", f)

f = [rid(r) for r in rows
     if r.get("action_type") in ACTIONABLE and r["step"] != "_task_end" and not r.get("element")]
check("F2  element{role,name} on every click/type/select", f)

f = [rid(r) for r in rows
     if r.get("action_type") in ACTIONABLE and r["step"] != "_task_end"
     and not (r.get("coords") or r.get("bbox"))]
check("F3  coords or bbox on every click/type/select", f)

f = [rid(r) for r in rows if (r.get("coords") or r.get("bbox")) and not r.get("viewport")]
check("F4  viewport present whenever coords/bbox is set", f)

f = [rid(r) for r in rows if r.get("url") and not str(r["url"]).startswith("http")]
check("F5  url is a real URL where present", f)

# PII
import re
PII = re.compile(r"9602654571|\b\d{10}\b")
f = []
for r in rows:
    for field in ("action_value",):
        v = r.get(field)
        if isinstance(v,str) and PII.search(v): f.append(f"{rid(r)}: {field} looks like PII")
    for c in (r.get("candidates") or []):
        if PII.search(json.dumps(c)): f.append(f"{rid(r)}: candidates contain PII")
check("F6  no phone/PII in action_value or candidates", f)

# ---- Report ---------------------------------------------------------------
print(f"Auditing {path} - {len(rows)} records, {len(tasks)} episodes\n")
w = max(len(r[0]) for r in results)
npass = 0
for rule, fails, note in results:
    status = "PASS" if not fails else f"FAIL ({len(fails)})"
    npass += not fails
    extra = f"   [{note}]" if note else ""
    print(f"{status:10s} {rule}{extra}")
    for x in fails[:6]:
        print(f"           - {x}")
    if len(fails) > 6:
        print(f"           - ... and {len(fails)-6} more")
print(f"\n{npass}/{len(results)} checks pass")
