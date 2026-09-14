"""Relevance scoring: is this job worth an application?

Pure-python, no dependencies. The score is deliberately explainable — every
job carries the reasons it scored what it did, so when the agent auto-applies
you can audit why, and when it skips you can see what filter caught it.
"""

from __future__ import annotations

import re

STOP = {
    "a", "an", "and", "the", "for", "with", "of", "to", "in", "on", "at", "or",
    "is", "are", "be", "as", "by", "we", "you", "your", "our", "will", "must",
    "have", "has", "job", "role", "work", "team", "years", "year", "experience",
}

SENIORITY_RANK = {
    "intern": 0, "trainee": 0, "graduate": 0,
    "junior": 1, "associate": 1, "entry": 1,
    "": 2, "mid": 2, "software engineer": 2, "engineer": 2, "developer": 2,
    "senior": 3, "sr": 3, "lead": 4, "staff": 4, "principal": 5,
    "manager": 4, "head": 5, "director": 6, "vp": 7,
}


def tokens(text: str) -> set[str]:
    if not text:
        return set()
    words = re.findall(r"[a-z0-9+#.]+", text.lower())
    return {w for w in words if w not in STOP and len(w) > 1}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def seniority_of(title: str) -> int:
    t = (title or "").lower()
    best = 2
    for word, rank in SENIORITY_RANK.items():
        if word and re.search(rf"\b{re.escape(word)}\b", t):
            best = max(best, rank) if rank > 2 else min(best, rank)
    return best


def score_job(job: dict, profile: dict, prefs: dict) -> tuple[int, list[str], str | None]:
    """Return (score 0-100, reasons, hard_skip_reason)."""
    reasons: list[str] = []
    title = job.get("title", "") or ""
    desc = job.get("description", "") or ""
    company = job.get("company_name", "") or ""
    location = job.get("location", "") or ""
    workplace = (job.get("workplace") or "").lower()

    jt = tokens(title)
    jd = tokens(f"{title} {desc}")

    # ---- hard filters -----------------------------------------------------
    blacklist = {_norm(c) for c in prefs.get("company_blacklist", [])}
    if _norm(company) in blacklist:
        return 0, ["company is blacklisted"], "blacklisted_company"

    for bad in prefs.get("exclude_keywords", []):
        if bad and bad.lower() in f"{title} {desc}".lower():
            return 0, [f"excluded keyword: {bad}"], f"excluded_keyword:{bad}"

    dealbreakers = prefs.get("dealbreakers", [])
    for bad in dealbreakers:
        if bad and bad.lower() in f"{title} {desc}".lower():
            return 0, [f"dealbreaker: {bad}"], f"dealbreaker:{bad}"

    # ---- title fit (35) ---------------------------------------------------
    target_titles = [t.lower() for t in prefs.get("target_titles", [])]
    title_score = 0
    for t in target_titles:
        tt = tokens(t)
        if not tt:
            continue
        overlap = len(tt & jt) / len(tt)
        if overlap >= 0.99:
            title_score = 35
            reasons.append(f"title matches target '{t}'")
            break
        title_score = max(title_score, int(overlap * 30))
    if title_score and title_score < 35:
        reasons.append(f"partial title match ({title_score}/35)")
    if not target_titles:
        title_score = 20

    # ---- skills overlap (25) ---------------------------------------------
    skills = [s.lower() for s in profile.get("skills", [])]
    hits = [s for s in skills if tokens(s) and tokens(s) <= jd]
    skill_score = min(25, int(25 * len(hits) / max(3, min(len(skills), 8)))) if skills else 12
    if hits:
        reasons.append("skills matched: " + ", ".join(hits[:6]))
    else:
        reasons.append("no listed skill appeared in the posting")

    # ---- location / workplace (15) ---------------------------------------
    loc_score = 0
    pref_locs = [l.lower() for l in prefs.get("locations", [])]
    remote_ok = prefs.get("remote_ok", True)
    if workplace == "remote" and remote_ok:
        loc_score = 15
        reasons.append("remote")
    elif any(l and l in location.lower() for l in pref_locs):
        loc_score = 15
        reasons.append(f"preferred location: {location}")
    elif prefs.get("willing_to_relocate"):
        loc_score = 8
        reasons.append("outside preferred locations but open to relocating")
    else:
        reasons.append(f"location mismatch: {location}")
        if prefs.get("location_is_hard_filter"):
            return 0, reasons, "location_mismatch"

    # ---- company shape (10) ----------------------------------------------
    comp_score = 0
    want_kind = (prefs.get("company_kind") or "any").lower()
    kind = (job.get("company_kind") or "unknown").lower()
    if want_kind in ("any", "", kind) or kind == "unknown":
        comp_score += 5
    else:
        reasons.append(f"company kind {kind} != preferred {want_kind}")
    head = job.get("company_headcount")
    lo = prefs.get("min_headcount")
    hi = prefs.get("max_headcount")
    if head is None:
        comp_score += 3
    elif (lo is None or head >= lo) and (hi is None or head <= hi):
        comp_score += 5
        reasons.append(f"headcount {head} within range")
    else:
        reasons.append(f"headcount {head} outside preferred range")

    # ---- relevance gate ---------------------------------------------------
    # Title and skills are the only signals that say "this is your job". If
    # neither fires, no amount of right-city, right-size, Easy-Apply bonus
    # should lift the score — otherwise a frontend internship in your city
    # outranks a genuine role in the wrong one.
    core = title_score + skill_score
    if core < 20:
        reasons.insert(0, "not relevant: neither title nor skills matched")
        return core, reasons, None

    # ---- seniority fit (10) ----------------------------------------------
    my_rank = seniority_of(profile.get("current_title", ""))
    job_rank = seniority_of(title)
    gap = job_rank - my_rank
    if gap <= -2:
        # An intern or junior req when you're senior. Being over-qualified is
        # not a match; it's a different job.
        sen_score = 0
        reasons.append(f"{abs(gap)} levels below your current title")
    elif gap <= 0:
        sen_score = 10 if gap == 0 else 6
    elif gap == 1:
        sen_score = 8
        reasons.append("one level up — a stretch, worth applying")
    else:
        sen_score = 2
        reasons.append(f"{gap} levels above current title")

    # ---- easy apply (5) ---------------------------------------------------
    ea_score = 5 if job.get("easy_apply") else 0
    if not job.get("easy_apply"):
        reasons.append("not Easy Apply — needs an external site")

    total = title_score + skill_score + loc_score + comp_score + sen_score + ea_score
    return max(0, min(100, total)), reasons, None
