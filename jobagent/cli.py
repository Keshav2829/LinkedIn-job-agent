"""jobagent — the state CLI the LinkedIn skills talk to.

Every command prints one compact JSON object on stdout. That is the whole
contract: the agent never has to remember anything across turns, it asks.

    python -m jobagent init
    python -m jobagent job seen --job-id 4021553311
    python -m jobagent job queue --limit 10
    python -m jobagent outreach pending
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from . import answers as A
from . import db as D
from . import matching as M
from . import templates as T

DEFAULT_CONFIG = {
    "daily_applications": 10,
    "daily_invites": 15,
    "daily_messages": 10,
    "weekly_invites": 80,
    "auto_apply_min_score": 70,
    "review_min_score": 55,
    "outreach_requires_approval": True,
    "apply_requires_approval": False,
    "contacts_per_company": 4,
    "min_days_between_touches": 5,
    "invite_followup_days": 7,
    "application_followup_days": 10,
    "pace_seconds_between_actions": [25, 70],
    # Trace logging is a training-data byproduct, not app behaviour — a run
    # is free to turn it off (a quick one-off apply, a debugging pass, a
    # machine whose corpus shouldn't be merged in). `trace log` becomes a
    # no-op that reports why, so a skill can keep calling it unconditionally
    # instead of branching on config at every step.
    "trace_logging": True,
}


def out(obj) -> None:
    try:
        json.dump(obj, sys.stdout, default=str, separators=(",", ":"))
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        # Someone piped us into `head`. The work is already committed; leave
        # quietly rather than dumping a traceback the caller has to read.
        try:
            sys.stdout.close()
        except Exception:
            pass
        os._exit(0)


def rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def one(cur) -> dict | None:
    r = cur.fetchone()
    return dict(r) if r else None


def name_key(name: str) -> str:
    return re.sub(r"\b(inc|llc|ltd|limited|pvt|private|corp|corporation|technologies|technology|labs|india)\b", "",
                  (name or "").lower()).strip()


def jloads(s, default):
    if not s:
        return default
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


def get_config(con) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(D.kv_get(con, "config", {}) or {})
    return cfg


def get_profile(con) -> dict:
    return D.kv_get(con, "profile", {}) or {}


def get_prefs(con) -> dict:
    return D.kv_get(con, "prefs", {}) or {}


def get_templates(con) -> dict:
    t = dict(T.DEFAULTS)
    t.update(D.kv_get(con, "templates", {}) or {})
    return t


# ---------------------------------------------------------------- quotas ---

def quota(con) -> dict:
    cfg = get_config(con)
    day = D.today()
    week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")

    applied = con.execute(
        "SELECT COUNT(*) c FROM applications WHERE substr(applied_at,1,10)=?", (day,)
    ).fetchone()["c"]
    invites = con.execute(
        "SELECT COUNT(*) c FROM outreach WHERE kind='invite' AND status='sent' "
        "AND substr(sent_at,1,10)=?", (day,)
    ).fetchone()["c"]
    msgs = con.execute(
        "SELECT COUNT(*) c FROM outreach WHERE kind!='invite' AND status='sent' "
        "AND substr(sent_at,1,10)=?", (day,)
    ).fetchone()["c"]
    week_invites = con.execute(
        "SELECT COUNT(*) c FROM outreach WHERE kind='invite' AND status='sent' "
        "AND substr(sent_at,1,10)>=?", (week_start,)
    ).fetchone()["c"]

    def band(used, cap):
        return {"used": used, "cap": cap, "left": max(0, cap - used)}

    q = {
        "day": day,
        "applications": band(applied, cfg["daily_applications"]),
        "invites": band(invites, cfg["daily_invites"]),
        "messages": band(msgs, cfg["daily_messages"]),
        "invites_this_week": band(week_invites, cfg["weekly_invites"]),
    }
    q["invites"]["left"] = min(q["invites"]["left"], q["invites_this_week"]["left"])
    return q


# ------------------------------------------------------------ companies ---

def company_upsert(con, name: str, data: dict | None = None) -> int:
    data = data or {}
    nk = name_key(name)
    row = one(con.execute("SELECT id FROM companies WHERE name_key=?", (nk,)))
    now = D.utcnow()
    if row:
        cid = row["id"]
        sets, vals = [], []
        for col in ("linkedin_url", "linkedin_id", "industry", "size", "headcount",
                    "kind", "is_mnc", "hq", "status", "notes"):
            if data.get(col) is not None:
                sets.append(f"{col}=?")
                vals.append(data[col])
        if sets:
            vals += [now, cid]
            con.execute(f"UPDATE companies SET {','.join(sets)}, updated_at=? WHERE id=?", vals)
        return cid
    cur = con.execute(
        "INSERT INTO companies (name,name_key,linkedin_url,linkedin_id,industry,size,headcount,"
        "kind,is_mnc,hq,status,notes,first_seen,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (name, nk, data.get("linkedin_url"), data.get("linkedin_id"), data.get("industry"),
         data.get("size"), data.get("headcount"), data.get("kind"), data.get("is_mnc"),
         data.get("hq"), data.get("status", "seen"), data.get("notes"), now, now),
    )
    return cur.lastrowid


# ------------------------------------------------------------- commands ---

def cmd_init(con, args):
    seeded = 0
    for c in A.CANONICAL:
        exists = one(con.execute("SELECT key FROM answers WHERE key=?", (c["key"],)))
        if exists:
            continue
        con.execute(
            "INSERT INTO answers (key,question,value,kind,category,aliases,confidence,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (c["key"], c["question"], "", c["kind"], c["category"],
             json.dumps(c.get("aliases", [])), 0.0, D.utcnow()),
        )
        seeded += 1
    if D.kv_get(con, "config") is None:
        D.kv_set(con, "config", DEFAULT_CONFIG)
    if D.kv_get(con, "profile") is None:
        D.kv_set(con, "profile", {})
    if D.kv_get(con, "prefs") is None:
        D.kv_set(con, "prefs", {})
    con.commit()
    unanswered = [r["key"] for r in rows(con.execute("SELECT key FROM answers WHERE value=''"))]
    out({"ok": True, "db": str(D.default_db_path()), "seeded_questions": seeded,
         "schema": D.SCHEMA_VERSION, "unanswered": len(unanswered),
         "setup_complete": bool(get_profile(con)) and not unanswered})


def cmd_profile(con, args):
    if args.action == "get":
        out(get_profile(con))
    else:
        p = get_profile(con)
        p.update(json.loads(args.json))
        D.kv_set(con, "profile", p)
        # keep the answer bank in sync with the profile
        mirror = {"full_name": "name", "current_company": "current_company",
                  "current_title": "current_title", "city": "location",
                  "linkedin_url": "linkedin_url", "email": "email", "phone": "phone",
                  "total_experience_years": "experience_years", "summary": "summary"}
        for akey, pkey in mirror.items():
            if p.get(pkey) not in (None, ""):
                _answer_set(con, akey, str(p[pkey]))
        out({"ok": True, "profile": p})


def cmd_prefs(con, args):
    if args.action == "get":
        out(get_prefs(con))
    else:
        p = get_prefs(con)
        p.update(json.loads(args.json))
        D.kv_set(con, "prefs", p)
        out({"ok": True, "prefs": p})


def cmd_config(con, args):
    if args.action == "get":
        out(get_config(con))
    else:
        c = get_config(con)
        c.update(json.loads(args.json))
        D.kv_set(con, "config", c)
        out({"ok": True, "config": c})


def _answer_set(con, key, value, question=None, kind=None, category=None, aliases=None):
    canon = A.CANONICAL_BY_KEY.get(key, {})
    existing = one(con.execute("SELECT * FROM answers WHERE key=?", (key,)))
    q = question or (existing or {}).get("question") or canon.get("question") or key.replace("_", " ")
    k = kind or (existing or {}).get("kind") or canon.get("kind") or "text"
    cat = category or (existing or {}).get("category") or canon.get("category") or "misc"
    al = aliases if aliases is not None else jloads((existing or {}).get("aliases"), canon.get("aliases", []))
    con.execute(
        "INSERT INTO answers (key,question,value,kind,category,aliases,confidence,updated_at) "
        "VALUES (?,?,?,?,?,?,1.0,?) ON CONFLICT(key) DO UPDATE SET "
        "value=excluded.value, question=excluded.question, kind=excluded.kind, "
        "category=excluded.category, aliases=excluded.aliases, confidence=1.0, "
        "updated_at=excluded.updated_at",
        (key, q, value, k, cat, json.dumps(al), D.utcnow()),
    )
    con.commit()


def cmd_answers(con, args):
    if args.action == "set":
        _answer_set(con, args.key, args.value, args.question, args.kind, args.category,
                    args.alias or None)
        out({"ok": True, "key": args.key})
    elif args.action == "get":
        r = one(con.execute("SELECT * FROM answers WHERE key=?", (args.key,)))
        if r:
            r["aliases"] = jloads(r["aliases"], [])
        out(r or {"found": False})
    elif args.action == "list":
        sql = "SELECT key,question,value,kind,category FROM answers"
        params = []
        if args.category:
            sql += " WHERE category=?"
            params.append(args.category)
        sql += " ORDER BY category, key"
        out({"answers": rows(con.execute(sql, params))})
    elif args.action == "missing":
        r = rows(con.execute(
            "SELECT key,question,kind,category FROM answers WHERE value='' ORDER BY category,key"))
        out({"missing": r, "count": len(r)})
    elif args.action == "resolve":
        allrows = rows(con.execute("SELECT * FROM answers WHERE value!=''"))
        for x in allrows:
            x["aliases"] = jloads(x["aliases"], [])
        hit = A.resolve(args.question, allrows, threshold=args.threshold)
        if not hit:
            out({"found": False, "question": args.question,
                 "action": "ask_user_then_store_with: answers set --key <k> --value <v>"})
            return
        opts = [o.strip() for o in args.options.split("|")] if args.options else None
        answer, truncated = A.shape(args.question, hit["value"], hit["kind"], opts,
                                    args.max_chars)
        out({"found": True, "key": hit["key"], "value": hit["value"],
             "answer": answer, "truncated": truncated,
             "kind": hit["kind"], "confidence": hit["confidence"],
             "matched_question": hit["question"]})


def cmd_company(con, args):
    if args.action == "upsert":
        data = json.loads(args.json) if args.json else {}
        cid = company_upsert(con, args.name, data)
        con.commit()
        out({"ok": True, "id": cid, "name": args.name})
    elif args.action == "get":
        r = one(con.execute("SELECT * FROM companies WHERE name_key=?", (name_key(args.name),)))
        out(r or {"found": False})
    elif args.action == "list":
        sql = "SELECT id,name,kind,headcount,status,industry FROM companies"
        params = []
        if args.status:
            sql += " WHERE status=?"
            params.append(args.status)
        sql += " ORDER BY name"
        out({"companies": rows(con.execute(sql, params))})
    elif args.action == "blacklist":
        cid = company_upsert(con, args.name, {"status": "blacklisted"})
        prefs = get_prefs(con)
        bl = set(prefs.get("company_blacklist", []))
        bl.add(args.name)
        prefs["company_blacklist"] = sorted(bl)
        D.kv_set(con, "prefs", prefs)
        con.commit()
        out({"ok": True, "id": cid, "blacklisted": args.name})


def cmd_job(con, args):
    if args.action == "seen":
        r = one(con.execute(
            "SELECT j.id,j.job_id,j.status,j.match_score,j.title,j.company_name, "
            "(SELECT COUNT(*) FROM applications a WHERE a.job_id=j.id) applied "
            "FROM jobs j WHERE j.job_id=?", (args.job_id,)))
        out({"known": bool(r), **(r or {})})
        return

    if args.action == "upsert":
        d = json.loads(args.json)
        jid = str(d["job_id"])
        now = D.utcnow()
        cid = company_upsert(con, d["company_name"], {
            "linkedin_url": d.get("company_url"), "headcount": d.get("company_headcount"),
            "kind": d.get("company_kind"), "industry": d.get("industry")}) if d.get("company_name") else None
        existing = one(con.execute("SELECT id FROM jobs WHERE job_id=?", (jid,)))
        if existing:
            con.execute(
                "UPDATE jobs SET title=COALESCE(?,title), location=COALESCE(?,location), "
                "url=COALESCE(?,url), description=COALESCE(?,description), "
                "easy_apply=COALESCE(?,easy_apply), workplace=COALESCE(?,workplace), "
                "posted=COALESCE(?,posted), updated_at=? WHERE id=?",
                (d.get("title"), d.get("location"), d.get("url"), d.get("description"),
                 int(d["easy_apply"]) if "easy_apply" in d else None, d.get("workplace"),
                 d.get("posted"), now, existing["id"]))
            con.commit()
            out({"ok": True, "id": existing["id"], "is_new": False})
            return
        cur = con.execute(
            "INSERT INTO jobs (job_id,title,company_id,company_name,location,workplace,url,posted,"
            "easy_apply,seniority,description,status,discovered_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,'new',?,?)",
            (jid, d.get("title", ""), cid, d.get("company_name"), d.get("location"),
             d.get("workplace"), d.get("url"), d.get("posted"),
             int(d.get("easy_apply", 0)), d.get("seniority"), d.get("description"), now, now))
        con.commit()
        D.log_event(con, "job_discovered", jid, {"title": d.get("title"), "company": d.get("company_name")})
        out({"ok": True, "id": cur.lastrowid, "is_new": True})
        return

    if args.action == "score":
        where, param = ("job_id=?", args.job_id) if args.job_id else ("id=?", args.id)
        job = one(con.execute(f"SELECT * FROM jobs WHERE {where}", (param,)))
        if not job:
            out({"found": False}); return
        comp = one(con.execute("SELECT * FROM companies WHERE id=?", (job["company_id"],))) or {}
        job = {**job, "company_kind": comp.get("kind"), "company_headcount": comp.get("headcount")}
        cfg = get_config(con)
        score, reasons, skip = M.score_job(job, get_profile(con), get_prefs(con))
        status = "skipped" if skip else ("queued" if score >= cfg["review_min_score"] else "scored")
        con.execute("UPDATE jobs SET match_score=?, match_reasons=?, status=?, skip_reason=?, "
                    "updated_at=? WHERE id=?",
                    (score, json.dumps(reasons), status, skip, D.utcnow(), job["id"]))
        con.commit()
        out({"id": job["id"], "job_id": job["job_id"], "score": score, "reasons": reasons,
             "status": status, "skip_reason": skip,
             "auto_apply": (not skip) and score >= cfg["auto_apply_min_score"]})
        return

    if args.action == "queue":
        cfg = get_config(con)
        q = quota(con)
        limit = min(args.limit or 99, q["applications"]["left"])
        minscore = args.min_score if args.min_score is not None else cfg["auto_apply_min_score"]
        r = rows(con.execute(
            "SELECT j.id,j.job_id,j.title,j.company_name,j.location,j.url,j.easy_apply,"
            "j.match_score,j.match_reasons FROM jobs j "
            "LEFT JOIN applications a ON a.job_id=j.id "
            "WHERE a.id IS NULL AND j.status='queued' AND j.match_score>=? "
            "ORDER BY j.match_score DESC, j.discovered_at ASC LIMIT ?", (minscore, max(0, limit))))
        for x in r:
            x["match_reasons"] = jloads(x["match_reasons"], [])
        out({"quota_left": q["applications"]["left"], "count": len(r), "jobs": r})
        return

    if args.action == "applied":
        job = one(con.execute("SELECT * FROM jobs WHERE job_id=?", (args.job_id,)))
        if not job:
            out({"ok": False, "error": "unknown job_id"}); return
        dup = one(con.execute("SELECT id FROM applications WHERE job_id=?", (job["id"],)))
        if dup:
            out({"ok": True, "duplicate": True, "application_id": dup["id"]}); return
        cur = con.execute(
            "INSERT INTO applications (job_id,applied_at,method,resume_used,answers_used,screening,notes) "
            "VALUES (?,?,?,?,?,?,?)",
            (job["id"], D.utcnow(), args.method, args.resume,
             args.answers or "{}", args.screening or "[]", args.notes))
        con.execute("UPDATE jobs SET status='applied', updated_at=? WHERE id=?", (D.utcnow(), job["id"]))
        con.execute("UPDATE companies SET status='applied', updated_at=? WHERE id=?",
                    (D.utcnow(), job["company_id"]))
        con.commit()
        D.log_event(con, "applied", args.job_id, {"title": job["title"], "company": job["company_name"]})
        out({"ok": True, "application_id": cur.lastrowid, "quota": quota(con)["applications"]})
        return

    if args.action == "skip":
        con.execute("UPDATE jobs SET status='skipped', skip_reason=?, updated_at=? WHERE job_id=?",
                    (args.reason, D.utcnow(), args.job_id))
        con.commit()
        out({"ok": True})
        return

    if args.action == "get":
        j = one(con.execute("SELECT * FROM jobs WHERE job_id=?", (args.job_id,)))
        if j:
            j["match_reasons"] = jloads(j["match_reasons"], [])
        out(j or {"found": False})
        return

    if args.action == "list":
        sql = "SELECT job_id,title,company_name,location,match_score,status FROM jobs"
        params = []
        if args.status:
            sql += " WHERE status=?"; params.append(args.status)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(args.limit or 50)
        out({"jobs": rows(con.execute(sql, params))})


def norm_url(url: str) -> str:
    """LinkedIn hands out the same profile with and without a trailing slash and
    with tracking query strings. Insert and lookup must agree, or a contact we
    already know looks new — and gets invited twice."""
    return (url or "").split("?")[0].rstrip("/")


def _contact_by_url(con, url):
    return one(con.execute("SELECT * FROM contacts WHERE profile_url=?", (norm_url(url),)))


def cmd_contact(con, args):
    if args.action == "seen":
        r = _contact_by_url(con, args.url)
        if not r:
            out({"known": False}); return
        cfg = get_config(con)
        cooldown_ok = True
        if r["last_touch_at"]:
            last = datetime.fromisoformat(r["last_touch_at"])
            cooldown_ok = (datetime.now(last.tzinfo) - last).days >= cfg["min_days_between_touches"]
        out({"known": True, "id": r["id"], "name": r["name"], "relation": r["relation"],
             "role_type": r["role_type"], "touch_count": r["touch_count"],
             "last_touch_at": r["last_touch_at"], "do_not_contact": bool(r["do_not_contact"]),
             "cooldown_ok": cooldown_ok,
             "recommend": "skip" if r["do_not_contact"] or not cooldown_ok else "ok"})
        return

    if args.action == "upsert":
        d = json.loads(args.json)
        if not d.get("profile_url"):
            out({"ok": False, "error": "contact upsert requires a non-empty 'profile_url' in --json "
                                        "— it's the unique key contacts are matched on (see db.py: "
                                        "contacts.profile_url). If a 'Meet the hiring team' panel entry "
                                        "has no clickable profile link, it can't be recorded as a "
                                        "contact — skip it."})
            return
        url = norm_url(d["profile_url"])
        cid = company_upsert(con, d["company_name"], {}) if d.get("company_name") else None
        now = D.utcnow()
        ex = _contact_by_url(con, url)
        if ex:
            con.execute(
                "UPDATE contacts SET name=COALESCE(?,name), headline=COALESCE(?,headline), "
                "company_id=COALESCE(?,company_id), company_name=COALESCE(?,company_name), "
                "title=COALESCE(?,title), role_type=COALESCE(?,role_type), degree=COALESCE(?,degree), "
                "mutual_count=COALESCE(?,mutual_count), is_alumni=COALESCE(?,is_alumni), "
                "alumni_school=COALESCE(?,alumni_school), location=COALESCE(?,location), "
                "updated_at=? WHERE id=?",
                (d.get("name"), d.get("headline"), cid, d.get("company_name"), d.get("title"),
                 d.get("role_type"), d.get("degree"), d.get("mutual_count"),
                 int(d["is_alumni"]) if "is_alumni" in d else None, d.get("alumni_school"),
                 d.get("location"), now, ex["id"]))
            con.commit()
            out({"ok": True, "id": ex["id"], "is_new": False, "relation": ex["relation"]})
            return
        relation = d.get("relation") or ("connected" if d.get("degree") == "1st" else "none")
        cur = con.execute(
            "INSERT INTO contacts (profile_url,name,headline,company_id,company_name,title,location,"
            "role_type,degree,mutual_count,is_alumni,alumni_school,relation,connected_at,notes,"
            "first_seen,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (url, d.get("name", ""), d.get("headline"), cid, d.get("company_name"), d.get("title"),
             d.get("location"), d.get("role_type", "employee"), d.get("degree"),
             d.get("mutual_count", 0), int(d.get("is_alumni", 0)), d.get("alumni_school"),
             relation, now if relation == "connected" else None, d.get("notes"), now, now))
        con.commit()
        out({"ok": True, "id": cur.lastrowid, "is_new": True, "relation": relation})
        return

    if args.action == "mark":
        r = _contact_by_url(con, args.url)
        if not r:
            out({"ok": False, "error": "unknown contact"}); return
        now = D.utcnow()
        if args.relation == "connected":
            con.execute("UPDATE contacts SET relation='connected', connected_at=? WHERE id=?", (now, r["id"]))
            D.log_event(con, "connection_accepted", args.url, {"name": r["name"]})
        else:
            con.execute("UPDATE contacts SET relation=? WHERE id=?", (args.relation, r["id"]))
        con.commit()
        out({"ok": True, "id": r["id"], "relation": args.relation})
        return

    if args.action == "pending-invites":
        days = args.older_than_days or 0
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        r = rows(con.execute(
            "SELECT id,profile_url,name,company_name,role_type,invited_at FROM contacts "
            "WHERE relation='invited' AND (invited_at IS NULL OR invited_at<=?) "
            "ORDER BY invited_at ASC LIMIT ?", (cutoff, args.limit or 50)))
        out({"count": len(r), "contacts": r})
        return

    if args.action == "awaiting-followup":
        # accepted the invite but we have not sent the real message yet
        r = rows(con.execute(
            "SELECT c.id,c.profile_url,c.name,c.company_name,c.role_type,c.connected_at,"
            "(SELECT COUNT(*) FROM outreach o WHERE o.contact_id=c.id AND o.kind!='invite' "
            " AND o.status IN ('sent','queued','approved')) followed "
            "FROM contacts c WHERE c.relation='connected' AND c.do_not_contact=0 "
            "AND c.connected_at IS NOT NULL "
            "HAVING followed=0 ORDER BY c.connected_at ASC LIMIT ?", (args.limit or 25,)))
        out({"count": len(r), "contacts": r})
        return

    if args.action == "list":
        sql = ("SELECT id,profile_url,name,company_name,role_type,relation,degree,mutual_count,"
               "is_alumni,touch_count,last_touch_at FROM contacts WHERE 1=1")
        params = []
        if args.relation:
            sql += " AND relation=?"; params.append(args.relation)
        if args.company:
            sql += " AND company_id=(SELECT id FROM companies WHERE name_key=?)"
            params.append(name_key(args.company))
        if args.role_type:
            sql += " AND role_type=?"; params.append(args.role_type)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(args.limit or 50)
        out({"contacts": rows(con.execute(sql, params))})


def _experience_years(con, p: dict) -> str:
    """Best-effort years-of-experience, since profile.experience_years is
    frequently left unset. Falls back to the answer bank (setup asks this
    directly as total_experience_years), then to "" so callers can decide
    how to phrase an unknown rather than rendering a bare 'years'."""
    years = p.get("experience_years")
    if years not in (None, ""):
        return str(years)
    row = one(con.execute("SELECT value FROM answers WHERE key='total_experience_years'"))
    if row and row["value"]:
        return str(row["value"])
    return ""


def _job_url(job: dict | None) -> str:
    """job.url is often blank (jobs are upserted from scraped listings that
    don't always carry it) — reconstruct the canonical LinkedIn URL from the
    job_id we always have rather than leaving a dangling '()' in messages."""
    if not job:
        return ""
    url = job.get("url")
    if url:
        return url
    job_id = job.get("job_id")
    return f"https://www.linkedin.com/jobs/view/{job_id}/" if job_id else ""


def _ctx_for(con, contact: dict, job: dict | None, extra: dict) -> dict:
    p = get_profile(con)
    skills = p.get("skills", [])
    name = contact.get("name", "") or ""
    years = _experience_years(con, p)
    ctx = {
        "first_name": name.split(" ")[0] if name else "there",
        "full_name": name,
        "company": contact.get("company_name") or (job or {}).get("company_name") or "your team",
        "school": contact.get("alumni_school") or (p.get("education") or [{}])[0].get("school", ""),
        "mutual_count": contact.get("mutual_count") or "",
        "my_name": p.get("name", ""),
        "my_title": p.get("current_title", ""),
        "years": years,
        # "{years} years" reads as "4 years"; when years is unknown, phrase it
        # so the sentence still scans ("with a few years across ..." instead
        # of the bare, blank-looking "with  years across ...").
        "years_phrase": f"{years} years" if years else "a few years",
        "top_skill": skills[0] if skills else "",
        "top_skills": ", ".join(skills[:3]),
        "job_title": (job or {}).get("title") or extra.get("job_title") or "engineering",
        "job_url": _job_url(job),
        # "(job_url)" reads as a dangling "()" once job_url is empty — give
        # templates a pre-wrapped, already-optional parenthetical instead.
        "job_url_paren": f" ({_job_url(job)})" if _job_url(job) else "",
        "one_specific_reason": extra.get("one_specific_reason", ""),
        # Pre-wrapped ", and <reason>" clause — bare {one_specific_reason} used
        # directly after "and" leaves a dangling "and" when it's unset (the
        # common case, since it's only ever filled via --extra).
        "reason_clause": (
            f", and {extra['one_specific_reason']}" if extra.get("one_specific_reason") else ""
        ),
        "next_step": extra.get("next_step", ""),
        "resume_link": p.get("resume_url", ""),
        # Browser MCP cannot attach a file, so a message sent on that driver
        # carries a link instead. Never promise an attachment we can't add.
        "resume_note": (
            "I've attached my resume"
            if extra.get("can_attach")
            else (f"My resume: {p['resume_url']}" if p.get("resume_url") else "")
        ),
        # A soft, honest offer rather than a claim of already having attached
        # something — the browser driver cannot actually attach a file to a
        # LinkedIn message, so templates that assumed it could were promising
        # something the send step never delivers.
        "resume_offer": (
            f"Happy to share my resume ({p['resume_url']}) if useful. "
            if p.get("resume_url")
            else "Happy to share my resume or any other details if useful. "
        ),
    }
    ctx.update({k: v for k, v in extra.items() if v not in (None, "")})
    return ctx


def cmd_outreach(con, args):
    if args.action == "draft":
        c = _contact_by_url(con, args.contact_url)
        if not c:
            out({"ok": False, "error": "unknown contact — upsert it first"}); return
        cfg = get_config(con)
        if c["do_not_contact"]:
            out({"ok": False, "error": "contact marked do_not_contact"}); return
        dupe = one(con.execute(
            "SELECT id,status FROM outreach WHERE contact_id=? AND kind=? AND status!='rejected'",
            (c["id"], args.kind)))
        if dupe and not args.force:
            out({"ok": False, "duplicate": True, "existing_id": dupe["id"],
                 "status": dupe["status"], "hint": "pass --force to draft anyway"}); return
        job = one(con.execute("SELECT * FROM jobs WHERE job_id=?", (args.job_id,))) if args.job_id else None
        extra = json.loads(args.extra) if args.extra else {}
        if args.attachment:
            extra.setdefault("can_attach", True)
        # A resume-bearing message with no way to carry the resume is worse than
        # no message: it promises the recipient something they never receive.
        if args.kind in ("hr_pitch", "referral_ask", "referral_ask_existing"):
            if not extra.get("can_attach") and not get_profile(con).get("resume_url"):
                out({"ok": False, "error": "no_resume_channel",
                     "hint": "This driver cannot attach files and profile.resume_url is unset. "
                             "Set one with: profile set --json '{\"resume_url\":\"https://...\"}' "
                             "or pass --attachment <path> on a driver that supports upload."})
                return
        tpl_name = args.template or args.kind
        # A message-length template (hr_pitch, referral_ask, ...) squeezed
        # into an invite's 280-char cap doesn't fit — trim() used to chop it
        # wherever the count landed, which reliably cut off mid-sentence and
        # dropped the sign-off entirely. Long templates belong to a message
        # kind (fits() gives those an 8000-char budget); invites need their
        # own short invite_* copy instead of a truncated message.
        _MESSAGE_ONLY_TEMPLATES = {
            "hr_pitch", "referral_ask", "referral_ask_existing",
            "intro_peer", "followup_nudge", "thanks_reply",
        }
        if args.kind.startswith("invite") and tpl_name in _MESSAGE_ONLY_TEMPLATES:
            out({"ok": False, "error": "template_too_long_for_invite",
                 "hint": f"'{tpl_name}' is written for a post-connection message (8000-char "
                         f"budget), not a {T.INVITE_LIMIT}-char invite note — squeezing it in "
                         "truncates mid-sentence. Use an invite_* template (invite_hr, "
                         "invite_employee, invite_alumni, invite_mutual) for --kind invite, "
                         f"and draft '{tpl_name}' as its own kind (e.g. --kind {tpl_name}) "
                         "once the connection is accepted."})
            return
        body = args.body or T.render(get_templates(con).get(tpl_name, ""), _ctx_for(con, c, job, extra))
        body = T.trim(args.kind, body)
        ok, length, limit = T.fits(args.kind, body)
        status = "queued" if cfg["outreach_requires_approval"] else "approved"
        cur = con.execute(
            "INSERT INTO outreach (contact_id,job_id,kind,body,body_original,status,queued_at,"
            "attachment,followup_due) VALUES (?,?,?,?,?,?,?,?,?)",
            (c["id"], (job or {}).get("id"), args.kind, body, body, status, D.utcnow(), args.attachment,
             (datetime.now() + timedelta(days=cfg["invite_followup_days"])).isoformat(timespec="seconds")))
        con.commit()
        out({"ok": True, "id": cur.lastrowid, "status": status, "kind": args.kind,
             "contact": c["name"], "chars": length, "limit": limit, "within_limit": ok, "body": body})
        return

    if args.action == "pending":
        r = rows(con.execute(
            "SELECT o.id,o.kind,o.body,o.queued_at,c.name,c.profile_url,c.company_name,c.role_type,"
            "j.title job_title FROM outreach o JOIN contacts c ON c.id=o.contact_id "
            "LEFT JOIN jobs j ON j.id=o.job_id WHERE o.status='queued' "
            "ORDER BY o.queued_at ASC LIMIT ?", (args.limit or 25,)))
        out({"count": len(r), "pending": r})
        return

    if args.action in ("approve", "reject"):
        status = "approved" if args.action == "approve" else "rejected"
        if args.all:
            con.execute("UPDATE outreach SET status=?, decided_at=? WHERE status='queued'",
                        (status, D.utcnow()))
            con.commit()
            out({"ok": True, "status": status, "applied_to": "all queued"}); return
        if args.body:
            con.execute("UPDATE outreach SET body=? WHERE id=?", (args.body, args.id))
        con.execute("UPDATE outreach SET status=?, decided_at=?, error=? WHERE id=?",
                    (status, D.utcnow(), args.reason, args.id))
        con.commit()
        out({"ok": True, "id": args.id, "status": status})
        return

    if args.action == "next":
        q = quota(con)
        sql = ("SELECT o.id,o.kind,o.body,o.attachment,c.profile_url,c.name,c.company_name,"
               "c.relation,j.url job_url,j.title job_title FROM outreach o "
               "JOIN contacts c ON c.id=o.contact_id LEFT JOIN jobs j ON j.id=o.job_id "
               "WHERE o.status='approved'")
        params = []
        if args.kind:
            sql += " AND o.kind=?"; params.append(args.kind)
        sql += " ORDER BY o.decided_at ASC"
        items = rows(con.execute(sql, params))
        inv_left, msg_left = q["invites"]["left"], q["messages"]["left"]
        picked = []
        for it in items:
            if it["kind"] == "invite":
                if inv_left <= 0:
                    continue
                inv_left -= 1
            else:
                if msg_left <= 0:
                    continue
                msg_left -= 1
            picked.append(it)
            if args.limit and len(picked) >= args.limit:
                break
        out({"count": len(picked), "quota": q, "send": picked,
             "pacing_seconds": get_config(con)["pace_seconds_between_actions"]})
        return

    if args.action == "sent":
        o = one(con.execute("SELECT * FROM outreach WHERE id=?", (args.id,)))
        if not o:
            out({"ok": False, "error": "unknown outreach id"}); return
        now = D.utcnow()
        con.execute("UPDATE outreach SET status='sent', sent_at=? WHERE id=?", (now, args.id))
        con.execute("UPDATE contacts SET touch_count=touch_count+1, last_touch_at=? WHERE id=?",
                    (now, o["contact_id"]))
        if o["kind"] == "invite":
            con.execute("UPDATE contacts SET relation='invited', invited_at=? WHERE id=? "
                        "AND relation IN ('none','declined')", (now, o["contact_id"]))
        con.commit()
        D.log_event(con, f"sent_{o['kind']}", str(args.id))
        out({"ok": True, "id": args.id, "quota": quota(con)})
        return

    if args.action == "fail":
        con.execute("UPDATE outreach SET status='failed', error=? WHERE id=?", (args.error, args.id))
        con.commit()
        out({"ok": True, "id": args.id})
        return

    if args.action == "list":
        # body_original + body together are the (draft, sent) pair a
        # preference-tuned writer model trains on — keep both visible here
        # rather than making a caller reconstruct the diff from `approve`
        # history that doesn't exist.
        sql = ("SELECT o.id,o.kind,o.status,o.queued_at,o.sent_at,o.body,o.body_original,"
               "c.name,c.company_name FROM outreach o JOIN contacts c ON c.id=o.contact_id WHERE 1=1")
        params = []
        if args.status:
            sql += " AND o.status=?"; params.append(args.status)
        if args.kind:
            sql += " AND o.kind=?"; params.append(args.kind)
        sql += " ORDER BY o.id DESC LIMIT ?"
        params.append(args.limit or 50)
        out({"outreach": rows(con.execute(sql, params))})


def cmd_reply(con, args):
    if args.action == "add":
        c = _contact_by_url(con, args.contact_url) if args.contact_url else None
        try:
            cur = con.execute(
                "INSERT INTO replies (contact_id,outreach_id,thread_url,snippet,received_at,sentiment) "
                "VALUES (?,?,?,?,?,?)",
                ((c or {}).get("id"), args.outreach_id, args.thread_url, args.snippet,
                 args.received_at or D.utcnow(), args.sentiment))
            con.commit()
            D.log_event(con, "reply_received", args.contact_url or "", {"snippet": args.snippet[:120]})
            out({"ok": True, "id": cur.lastrowid, "is_new": True,
                 "contact": (c or {}).get("name")})
        except Exception:
            out({"ok": True, "is_new": False, "duplicate": True})
        return

    if args.action == "unnotified":
        r = rows(con.execute(
            "SELECT r.id,r.snippet,r.received_at,r.thread_url,r.sentiment,c.name,c.profile_url,"
            "c.company_name,c.role_type FROM replies r LEFT JOIN contacts c ON c.id=r.contact_id "
            "WHERE r.notified=0 ORDER BY r.received_at DESC"))
        out({"count": len(r), "replies": r})
        return

    if args.action == "mark-notified":
        if args.all:
            con.execute("UPDATE replies SET notified=1 WHERE notified=0")
        else:
            con.execute("UPDATE replies SET notified=1 WHERE id=?", (args.id,))
        con.commit()
        out({"ok": True})
        return

    if args.action == "handled":
        con.execute("UPDATE replies SET handled=1 WHERE id=?", (args.id,))
        con.commit()
        out({"ok": True})
        return

    if args.action == "list":
        out({"replies": rows(con.execute(
            "SELECT r.id,r.snippet,r.received_at,r.handled,c.name FROM replies r "
            "LEFT JOIN contacts c ON c.id=r.contact_id ORDER BY r.received_at DESC LIMIT ?",
            (args.limit or 30,)))})


def cmd_run(con, args):
    if args.action == "start":
        cur = con.execute("INSERT INTO runs (day,started_at) VALUES (?,?)", (D.today(), D.utcnow()))
        con.commit()
        D.kv_set(con, "current_run", cur.lastrowid)
        p, pr = get_profile(con), get_prefs(con)
        unanswered = con.execute("SELECT COUNT(*) c FROM answers WHERE value=''").fetchone()["c"]
        out({"run_id": cur.lastrowid, "quota": quota(con),
             "setup_complete": bool(p.get("name")) and bool(pr.get("target_titles")),
             "unanswered_questions": unanswered,
             "config": get_config(con)})
        return

    if args.action == "quota":
        out(quota(con)); return

    if args.action == "end":
        rid = D.kv_get(con, "current_run")
        q = quota(con)
        con.execute("UPDATE runs SET ended_at=?, applied=?, invites=?, messages=?, notes=? WHERE id=?",
                    (D.utcnow(), q["applications"]["used"], q["invites"]["used"],
                     q["messages"]["used"], args.notes, rid))
        con.commit()
        out({"ok": True, "run_id": rid, "totals": q})
        return

    if args.action == "history":
        out({"runs": rows(con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (args.limit or 10,)))})


def cmd_report(con, args):
    if args.action == "daily":
        day = D.today()
        out({
            "day": day,
            "quota": quota(con),
            "applied_today": rows(con.execute(
                "SELECT j.title,j.company_name,j.match_score,a.applied_at FROM applications a "
                "JOIN jobs j ON j.id=a.job_id WHERE substr(a.applied_at,1,10)=?", (day,))),
            "sent_today": rows(con.execute(
                "SELECT o.kind,c.name,c.company_name FROM outreach o JOIN contacts c ON c.id=o.contact_id "
                "WHERE o.status='sent' AND substr(o.sent_at,1,10)=?", (day,))),
            "new_replies": con.execute(
                "SELECT COUNT(*) c FROM replies WHERE substr(received_at,1,10)=?", (day,)).fetchone()["c"],
            "awaiting_approval": con.execute(
                "SELECT COUNT(*) c FROM outreach WHERE status='queued'").fetchone()["c"],
        })
        return

    if args.action == "pipeline":
        out({
            "jobs": {r["status"]: r["c"] for r in rows(con.execute(
                "SELECT status, COUNT(*) c FROM jobs GROUP BY status"))},
            "applications": con.execute("SELECT COUNT(*) c FROM applications").fetchone()["c"],
            "companies": con.execute("SELECT COUNT(*) c FROM companies").fetchone()["c"],
            "contacts": {r["relation"]: r["c"] for r in rows(con.execute(
                "SELECT relation, COUNT(*) c FROM contacts GROUP BY relation"))},
            "outreach": {r["status"]: r["c"] for r in rows(con.execute(
                "SELECT status, COUNT(*) c FROM outreach GROUP BY status"))},
            "replies": con.execute("SELECT COUNT(*) c FROM replies").fetchone()["c"],
        })
        return

    if args.action == "followups":
        cfg = get_config(con)
        inv_cut = (datetime.now() - timedelta(days=cfg["invite_followup_days"])).isoformat()
        app_cut = (datetime.now() - timedelta(days=cfg["application_followup_days"])).isoformat()
        out({
            "stale_invites": rows(con.execute(
                "SELECT profile_url,name,company_name,invited_at FROM contacts "
                "WHERE relation='invited' AND invited_at<=? ORDER BY invited_at LIMIT 25", (inv_cut,))),
            "accepted_awaiting_message": rows(con.execute(
                "SELECT c.profile_url,c.name,c.company_name,c.role_type,c.connected_at FROM contacts c "
                "WHERE c.relation='connected' AND c.connected_at IS NOT NULL AND NOT EXISTS "
                "(SELECT 1 FROM outreach o WHERE o.contact_id=c.id AND o.kind!='invite' "
                " AND o.status IN ('sent','queued','approved')) ORDER BY c.connected_at LIMIT 25")),
            "silent_applications": rows(con.execute(
                "SELECT j.title,j.company_name,a.applied_at FROM applications a JOIN jobs j ON j.id=a.job_id "
                "WHERE a.applied_at<=? AND a.outcome='submitted' ORDER BY a.applied_at LIMIT 25", (app_cut,))),
        })


def cmd_template(con, args):
    t = get_templates(con)
    if args.action == "list":
        out({"templates": sorted(t.keys())})
    elif args.action == "get":
        out({"name": args.name, "body": t.get(args.name, "")})
    elif args.action == "set":
        overrides = D.kv_get(con, "templates", {}) or {}
        overrides[args.name] = args.body
        D.kv_set(con, "templates", overrides)
        out({"ok": True, "name": args.name})
    elif args.action == "render":
        c = _contact_by_url(con, args.contact_url) if args.contact_url else {}
        job = one(con.execute("SELECT * FROM jobs WHERE job_id=?", (args.job_id,))) if args.job_id else None
        extra = json.loads(args.extra) if args.extra else {}
        body = T.render(t.get(args.name, ""), _ctx_for(con, c or {}, job, extra))
        ok, length, limit = T.fits(args.name, body)
        out({"body": body, "chars": length, "limit": limit, "within_limit": ok})


START_STEP = "_task_start"  # reserved step name: the one record naming the instruction that opened this task_id
END_STEP = "_task_end"  # reserved step name: the one episode-level summary record per task_id

# Bumped whenever the trace record shape changes in a way that would make a
# mixed corpus misleading to train on. Every record written from here on
# carries it; anything without a `trace_schema` field predates this fix and
# should be treated as v1 — thinner, less validated, safe to exclude from a
# real training pull rather than silently merged in.
#
# v3 (this cut-over, per plugin/references/trace-schema-v3.md): v2 made
# records *valid*; it did not make them *learnable* — a Sep 5 measurement of
# a clean, fully-conformant 55-record day still scored 0.0% usable training
# examples (tools/audit_training_yield.py). The record now carries an
# `observation` block (page state: url, screenshot, viewport, scroll,
# page_dims, a11y, page_text) and an `action` block (the real action space,
# including `tool_call` for a jobagent CLI lookup and `read`/`scroll` for
# the interactions that used to leave no trace) instead of the old flat
# `action_type`/`coords`/`element`/`url` fields. v2 records already on disk
# are not rewritten — `by_schema` in `trace stats` keeps the two corpora
# separable, and v2's grounding is dead for coordinate supervision anyway
# (recorded coords didn't match the stored image size), so there is nothing
# to migrate forward.
TRACE_SCHEMA_VERSION = 3

# The real action space (trace-schema-v3.md §2). `verify`/`na` (v2) are
# retired — a look is a `read`, and a record with nothing to emit shouldn't
# exist.
VALID_ACTION_KINDS = {
    "click", "type", "select", "key", "scroll", "navigate",
    "read", "tool_call", "ask_human", "stop",
}

# claude_browser_pane added alongside the v3 cut-over: trace-schema-v3.md's
# own "what v3 still does not give you" flagged this gap explicitly — a
# third browser surface (the Claude desktop app's in-app browser pane,
# `mcp__remote-devices__Claude_Browser__*`) has different coordinate
# semantics from both existing drivers and was logging as unset or silently
# mislabeled as one of them, which corrupts coordinate training the same
# way an unlabeled claude_in_chrome/browsermcp mix would.
VALID_DRIVERS = {"claude_in_chrome", "browsermcp", "claude_browser_pane", "cdp"}
_DRIVER_ALIASES = {
    "claude_in_chrome": "claude_in_chrome", "claude-in-chrome": "claude_in_chrome",
    "claudeinchrome": "claude_in_chrome", "claude in chrome": "claude_in_chrome",
    "browsermcp": "browsermcp", "browser_mcp": "browsermcp",
    "browser-mcp": "browsermcp", "browser mcp": "browsermcp",
    "claude_browser_pane": "claude_browser_pane", "claude-browser-pane": "claude_browser_pane",
    "claude browser pane": "claude_browser_pane", "browser_pane": "claude_browser_pane",
    "in_app_browser": "claude_browser_pane", "in-app-browser": "claude_browser_pane",
    "claude_browser": "claude_browser_pane",
    # The CDP driver (`jobagent browse`). Without this entry a CDP step has
    # no honest value to record and lands mislabelled as one of the MCP
    # drivers — which then makes the corpus claim a capability the step
    # didn't have.
    "cdp": "cdp", "jobagent_browse": "cdp", "browse": "cdp", "playwright": "cdp",
    "playwright_cdp": "cdp", "chrome_devtools_protocol": "cdp",
}

# Playbook §9's own documented vocabulary for a *step* outcome. Before this
# was enforced, `outcome` accumulated 17 distinct values across 59 records —
# "success", raw result dicts, full English sentences — none of it usable as
# a training label without a lossy cleaning pass. Anything richer than one
# of these four words belongs in the new `result` field instead.
VALID_STEP_OUTCOMES = {"success", "error_shown", "blocked", "na"}

# _task_start's governing field is `source`, not `outcome` — the two were
# landing in the same slot depending on which call wrote the record.
VALID_TASK_SOURCES = {"user_request", "scheduled_daily_run", "skill_default"}

# _task_end is an episode summary. The base vocabulary is closed; `skipped:`
# is an open family since the reason varies (needs_human:<question>,
# external_ats, ...) and playbook §9 already documents it that way.
TASK_END_BASE_OUTCOMES = {"applied", "sent", "not_applied", "error", "aborted"}

# Playbook §9 has always said `step` is "a closed, reusable vocabulary, not
# one-off free text" — but nothing enforced it, and an audit of the first
# three days' corpus (154 records) found five invented names that each
# duplicate a canonical one: `select_resume`, `view_job_posting`,
# `click_easy_apply`, `fill_screening_questions`, `review_application`. Each
# splits examples off a step that already existed, which is exactly the
# fragmentation §9 warned about. Same treatment as `driver` and `outcome`:
# a closed set, aliases for the near-misses that already happened, and a
# loud rejection for anything genuinely new so the vocabulary is extended
# in this file deliberately rather than per-run.
VALID_SKILLS = {
    "linkedin-apply", "linkedin-outreach", "linkedin-followup",
    "linkedin-inbox", "linkedin-setup", "linkedin-daily-run",
}

VALID_STEPS = {
    START_STEP, END_STEP,
    # --- session / run phase (task_id: run_<run_id>, skill linkedin-daily-run)
    "capture_user_intent", "open_run", "open_linkedin", "verify_session",
    "open_jobs_tab",
    # --- search & discovery phase (task_id: search_<slug>)
    "type_search_query", "apply_filter", "search_jobs", "paginate_results",
    "screen_job_candidate", "open_job_card",
    # --- one job's evaluation + application (task_id: job_<job_id>)
    "check_job_seen", "open_job", "read_job_description", "score_job",
    "record_score_decision", "detect_external_ats", "find_easy_apply_button",
    "verify_no_error", "upload_resume", "fill_screening_field",
    "select_work_auth", "harvest_hiring_team", "click_review",
    "submit_application", "close_confirmation_dialog",
    # --- outreach / follow-up (task_id: profile_url)
    "find_connect_button", "fill_connect_note", "send_invite",
    "open_message_thread", "fill_message_body", "attach_resume",
    "send_message",
    # --- inbox / setup
    "read_reply", "read_own_profile",
    # --- any skill, any task
    "detect_abort_signal",
}

# Names that already leaked into the corpus, mapped to the canonical step
# they duplicate. Rewritten rather than rejected so a run that reaches for
# the old name still logs a usable record instead of losing the step.
_STEP_ALIASES = {
    "select_resume": "upload_resume",
    "view_job_posting": "open_job",
    "click_easy_apply": "find_easy_apply_button",
    "fill_screening_questions": "fill_screening_field",
    "review_application": "click_review",
    "click_submit": "submit_application",
    "check_for_errors": "verify_no_error",
    "read_job_posting": "read_job_description",
    "search_for_jobs": "search_jobs",
    "score_the_job": "score_job",
}


def _normalize_step(v):
    if not isinstance(v, str):
        return v
    s = v.strip()
    return _STEP_ALIASES.get(s, s)


def _normalize_task_id(rec):
    """Job task ids arrived three different ways in the first corpus —
    `job_4455089718`, a bare `4455863129`, and a full
    `https://www.linkedin.com/jobs/view/4445891402/` — which means the same
    kind of episode sits under three id shapes and `trace task` can't group
    them. The two unambiguous shapes are rewritten to the canonical
    `job_<id>`; anything else is left alone (outreach/inbox ids are URLs by
    design). Returns the note to report, or None."""
    tid = rec.get("task_id")
    if not isinstance(tid, str) or rec.get("skill") != "linkedin-apply":
        return None
    t = tid.strip()
    if t.startswith(("job_", "search_")):
        return None
    if t.isdigit():
        rec["task_id"] = f"job_{t}"
        return f"task_id {t!r} normalized to {rec['task_id']!r} — bare job ids fragment `trace task`"
    m = re.search(r"/jobs/view/(\d+)", t)
    if m:
        rec["task_id"] = f"job_{m.group(1)}"
        return f"task_id {t!r} normalized to {rec['task_id']!r} — use job_<job_id>, not the job URL"
    return None


def _normalize_driver(v):
    if not isinstance(v, str):
        return v
    return _DRIVER_ALIASES.get(v.strip().lower(), v)


def _valid_task_end_outcome(v) -> bool:
    return isinstance(v, str) and (
        v in TASK_END_BASE_OUTCOMES or (v.startswith("skipped:") and len(v) > len("skipped:"))
    )


def _persona_id(con) -> str:
    """Stable per-machine/per-account tag on every trace record. Without it,
    a corpus that ever merges more than one account's runs has no way to
    separate them — and a click-prediction or field-filling model trained
    across personas will happily memorise one person's CTC and phone number
    as if they were facts about the world instead of an answer-bank lookup.
    Generated once, then cached in kv so it's stable across runs."""
    pid = D.kv_get(con, "trace_persona_id")
    if not pid:
        pid = f"persona_{uuid.uuid4().hex[:10]}"
        D.kv_set(con, "trace_persona_id", pid)
    return pid


def _image_dims(img_bytes: bytes, ext: str) -> dict | None:
    """Best-effort {width,height} straight from the file header — no Pillow
    dependency, since it isn't guaranteed installed wherever this CLI runs.
    v3 requires `observation.screenshot_dims` alongside any screenshot (§8);
    this is what makes that automatic for the two formats the two drivers
    actually emit (Claude in Chrome: JPEG, Browser MCP/browsermcp: PNG).
    Returns None for gif/webp/bin — the caller then has to supply
    `observation.screenshot_dims` itself, or the record is rejected rather
    than silently missing the dims v3 needs for coordinate supervision."""
    try:
        if ext == "png" and len(img_bytes) >= 24 and img_bytes[12:16] == b"IHDR":
            w, h = struct.unpack(">II", img_bytes[16:24])
            return {"width": w, "height": h}
        if ext == "jpg":
            i = 2
            n = len(img_bytes)
            while i + 9 < n:
                if img_bytes[i] != 0xFF:
                    i += 1
                    continue
                marker = img_bytes[i + 1]
                # SOF0..SOF15, excluding the DHT/JPG/DAC markers that share
                # the range but aren't start-of-frame.
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", img_bytes[i + 5:i + 9])
                    return {"width": w, "height": h}
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg_len = struct.unpack(">H", img_bytes[i + 2:i + 4])[0]
                i += 2 + seg_len
            return None
    except (struct.error, IndexError):
        return None
    return None


def _save_screenshot(tdir: Path, image_b64_value: str, day: str) -> dict:
    """Decode and save one screenshot, sniffing its real format. Shared by
    a fresh `trace log` call and a backfill merge, so both paths save and
    name screenshots identically."""
    shots_dir = tdir / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    try:
        val = image_b64_value
        if val.startswith("@"):
            # Large base64 strings can exceed the OS's per-argument exec
            # limit (Linux MAX_ARG_STRLEN, ~128KB) even nowhere near ARG_MAX.
            # @<path> reads it from a file instead of the command line.
            val = Path(val[1:]).read_text().strip()
        img_bytes = base64.b64decode(val)
        # Sniff the real format instead of assuming PNG — Claude in Chrome's
        # screenshot tool returns JPEG, and a .png file that's actually JPEG
        # bytes is a papercut every downstream image library trips on.
        if img_bytes[:3] == b"\xff\xd8\xff":
            ext = "jpg"
        elif img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            ext = "png"
        elif img_bytes[:6] in (b"GIF87a", b"GIF89a"):
            ext = "gif"
        elif img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
            ext = "webp"
        else:
            ext = "bin"  # unrecognized — saved as-is, flagged below
        img_name = f"{day}-{uuid.uuid4().hex[:12]}.{ext}"
        (shots_dir / img_name).write_bytes(img_bytes)
        result = {"screenshot": f"screenshots/{img_name}"}
        if ext == "bin":
            result["screenshot_error"] = "unrecognized image format, saved raw"
        dims = _image_dims(img_bytes, ext)
        if dims:
            result["screenshot_dims"] = dims
        return result
    except Exception as e:  # noqa: BLE001 — bad b64 shouldn't kill the record
        return {"screenshot_error": f"{type(e).__name__}: {e}"}


def _trace_files(tdir: Path) -> list[Path]:
    """Every trace record file under `tdir`, oldest layout first: legacy
    flat `<day>.jsonl` files (written before this folder-per-day scheme),
    then the current `<day>/<session>.jsonl` layout — one folder per
    calendar day, one file per run inside it, so two sessions on the same
    day don't commingle into one ever-growing file and a stats/backfill
    scan doesn't have to guess which layout it's looking at."""
    files = sorted(tdir.glob("*.jsonl"))
    for day_dir in sorted(p for p in tdir.glob("*")
                           if p.is_dir() and p.name not in ("screenshots", "pages")):
        files.extend(sorted(day_dir.glob("*.jsonl")))
    return files


def _session_stamp(con, rec: dict) -> str:
    """File name (inside the day folder) for this record — one file per run
    (`jobagent run start` .. `run end`), keyed off the run's own started_at
    so every step of one run lands in the same file no matter when each
    step is actually logged. Records logged with no active run (manual
    testing, one-off calls made before `run start`) share a single 'adhoc'
    file per day instead of fragmenting into one file per call."""
    run_id = rec.get("run_id")
    if run_id is not None:
        row = con.execute("SELECT started_at FROM runs WHERE id=?", (run_id,)).fetchone()
        if row and row["started_at"]:
            try:
                return datetime.fromisoformat(row["started_at"]).astimezone().strftime("%H-%M-%S")
            except ValueError:
                pass
    return "adhoc"


def _merge_backfill(tdir: Path, rec: dict, image_b64_arg: str | None):
    """Update the existing (task_id, step) record in place instead of
    appending a duplicate.

    `trace log --json '{"backfill":true,...}'` used to always append — at
    59 records that already produced 8 "(retrieval backfill)" rows sitting
    beside their originals: same task, same step, near-duplicate content,
    invisible by hand and silently inflating any count taken at scale.
    Returns {"log": path} on success, None if no matching record exists
    (the caller then falls back to logging it as a fresh step).

    (task_id, step) alone is ambiguous whenever a step legitimately repeats
    inside one episode — `fill_screening_field` once per question,
    `apply_filter` once per filter, and every retry chain. Matching the last
    such record silently edited the wrong one: a 4 Sep audit tried to add the
    missing coords to the attempt-1 stale-ref `apply_filter` and would have
    rewritten the attempt-2 success four records later. So `step_index`, when
    the caller supplies it, narrows the match to exactly one record. It stays
    in `protected` — it selects the target, it is never written into it.
    """
    target_task, target_step = rec.get("task_id"), rec.get("step")
    target_index = rec.get("step_index")
    protected = {"ts", "day", "run_id", "step_index", "task_id", "step", "trace_schema", "total_steps"}
    for fp in reversed(_trace_files(tdir)):
        try:
            lines = fp.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
        match_i = None
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if not line:
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if existing.get("task_id") != target_task or existing.get("step") != target_step:
                continue
            if target_index is not None and existing.get("step_index") != target_index:
                continue
            match_i = i
            break
        if match_i is None:
            continue
        existing = json.loads(lines[match_i])
        for k, v in rec.items():
            if k in protected or v is None:
                continue
            # observation/action are v3's nested blocks — merge their keys
            # instead of replacing the whole block, so a backfill that only
            # adds e.g. `action.tool.result` doesn't have to repeat coords,
            # element, and everything else already on the target record.
            if k in ("observation", "action") and isinstance(v, dict) and isinstance(existing.get(k), dict):
                existing[k].update(v)
            else:
                existing[k] = v
        existing["backfilled_at"] = D.utcnow()
        if image_b64_arg and not _has_screenshot(existing):
            _target_dict_for_screenshot(existing).update(
                _save_screenshot(tdir, image_b64_arg, existing.get("day", D.today())))
        _apply_norms(existing)
        lines[match_i] = json.dumps(existing, default=str, separators=(",", ":"))
        fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"log": str(fp)}
    return None


def _normalize_point(pt, viewport):
    """{x,y} in raw pixels -> {x,y} in 0..1, given {width,height}. None if either is missing/invalid."""
    if not isinstance(pt, dict) or not isinstance(viewport, dict):
        return None
    w, h = viewport.get("width"), viewport.get("height")
    x, y = pt.get("x"), pt.get("y")
    if not (isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0):
        return None
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        return None
    return {"x": round(x / w, 4), "y": round(y / h, 4)}


def _target_dict_for_screenshot(rec: dict) -> dict:
    """Where a screenshot (and its dims) belongs on this record: v3's
    `observation` block, or the record's own top level for a legacy
    v2-shaped record being backfilled — so an old-format backfill doesn't
    suddenly grow a nested block it never had."""
    if isinstance(rec.get("observation"), dict) or isinstance(rec.get("action"), dict) \
            or (rec.get("trace_schema") or TRACE_SCHEMA_VERSION) >= 3:
        return rec.setdefault("observation", {})
    return rec


def _has_screenshot(rec: dict) -> bool:
    obs = rec.get("observation")
    if isinstance(obs, dict) and obs.get("screenshot"):
        return True
    return bool(rec.get("screenshot"))


def _apply_norms(rec: dict) -> None:
    """Fill coords_norm/bbox_norm in place. Reads `action{}`/`observation{}`
    on a v3 record, or the legacy top-level coords/bbox/viewport on a v2
    one — so this one function serves both the fresh-log path and a
    backfill onto either schema."""
    action = rec.get("action") if isinstance(rec.get("action"), dict) else None
    observation = rec.get("observation") if isinstance(rec.get("observation"), dict) else None
    target = action if action is not None else rec
    viewport = (observation or rec).get("viewport")
    norm = _normalize_point(target.get("coords"), viewport)
    if norm and "coords_norm" not in target:
        target["coords_norm"] = norm
    bbox = target.get("bbox")
    if bbox and viewport and "bbox_norm" not in target:
        w, h = viewport.get("width"), viewport.get("height")
        if isinstance(bbox, dict) and isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
            try:
                target["bbox_norm"] = {
                    "x": round(bbox["x"] / w, 4), "y": round(bbox["y"] / h, 4),
                    "w": round(bbox["w"] / w, 4), "h": round(bbox["h"] / h, 4),
                }
            except (KeyError, TypeError, ZeroDivisionError):
                pass


def cmd_trace(con, args):
    """Append step-by-step run traces to data/traces/<day>/<session>.jsonl.

    This is *not* app state — nothing here feeds scoring, quotas, or
    dedupe. It exists so real runs leave behind (page state, instruction,
    action) triples that a future local model could be fine-tuned on,
    without changing anything about how the agent behaves today. Skipping
    it costs nothing; calling it costs one extra CLI call per meaningful
    step.
    """
    tdir = D.traces_dir(args.db)

    if args.action == "steps":
        out({"trace_logging": get_config(con).get("trace_logging", True),
             "valid_skills": sorted(VALID_SKILLS),
             "valid_steps": sorted(VALID_STEPS),
             "aliases": _STEP_ALIASES})
        return

    if args.action in ("enable", "disable"):
        c = get_config(con)
        c["trace_logging"] = (args.action == "enable")
        D.kv_set(con, "config", c)
        out({"ok": True, "trace_logging": c["trace_logging"]})
        return

    if args.action == "log":
        # The on/off switch. Checked before anything else so a disabled run
        # costs one config read per step and nothing more — no validation,
        # no screenshot decode, no file write. Skills call `trace log`
        # unconditionally; this is where "don't log this run" is honoured.
        if not get_config(con).get("trace_logging", True):
            out({"ok": True, "skipped": "trace_logging disabled in config — "
                                        "`jobagent trace enable` to turn it back on"})
            return

        rec = jloads(args.json, {})
        rec.setdefault("ts", D.utcnow())
        rec.setdefault("day", D.today())
        rec.setdefault("attempt", 1)
        rec.setdefault("trace_schema", TRACE_SCHEMA_VERSION)
        if not rec.get("persona_id"):
            rec["persona_id"] = _persona_id(con)
        if "run_id" not in rec:
            rec["run_id"] = D.kv_get(con, "current_run")

        if rec.get("step"):
            rec["step"] = _normalize_step(rec["step"])
        step = rec.get("step")
        is_backfill = bool(rec.pop("backfill", False))
        if is_backfill and not (rec.get("task_id") and step):
            out({"ok": False, "error": "a backfill record needs both task_id and step to find its target"})
            return

        # --- closed-vocabulary validation, rejected loudly rather than -----
        # --- silently accepted and cleaned up later -------------------------
        if rec.get("skill") and rec["skill"] not in VALID_SKILLS:
            out({"ok": False, "error": f"unknown skill {rec['skill']!r} — "
                                        f"expected one of {sorted(VALID_SKILLS)}"})
            return
        if not step:
            out({"ok": False, "error": "`step` is required"})
            return
        if step not in VALID_STEPS:
            near = sorted(s for s in VALID_STEPS
                          if s.split("_")[0] == step.split("_")[0] or step in s or s in step)
            out({"ok": False, "error": f"unknown step {step!r} — `step` is a closed vocabulary so the "
                                        f"corpus doesn't fragment into near-duplicate labels. "
                                        + (f"Did you mean one of {near}? " if near else "")
                                        + "If this is a genuinely new kind of step, add it to "
                                          "VALID_STEPS in cli.py and to the step catalog in "
                                          "plugin/references/trace-logging-checklist.md — deliberately, "
                                          "not per-run.",
                 "valid_steps": sorted(VALID_STEPS)})
            return

        task_note = _normalize_task_id(rec)

        if rec.get("driver"):
            rec["driver"] = _normalize_driver(rec["driver"])
            if rec["driver"] not in VALID_DRIVERS:
                out({"ok": False, "error": f"unknown driver {rec['driver']!r} — "
                                            f"expected one of {sorted(VALID_DRIVERS)}"})
                return

        if step == START_STEP:
            if "outcome" in rec:
                out({"ok": False, "error": "_task_start records carry `source`, not `outcome` — "
                                            "move this value to the `source` field"})
                return
            if rec.get("source") not in VALID_TASK_SOURCES:
                out({"ok": False, "error": f"_task_start requires `source` in {sorted(VALID_TASK_SOURCES)}"})
                return
        elif step == END_STEP:
            if "outcome" in rec:
                if not isinstance(rec["outcome"], str):
                    out({"ok": False, "error": "_task_end `outcome` must be a string"})
                    return
                if not _valid_task_end_outcome(rec["outcome"]):
                    out({"ok": False, "error": f"_task_end `outcome` must be one of "
                                                f"{sorted(TASK_END_BASE_OUTCOMES)} or 'skipped:<reason>' — "
                                                "put richer detail (score, company, reason text) in the "
                                                "`result` field instead"})
                    return
        else:
            if "outcome" in rec:
                if not isinstance(rec["outcome"], str) or rec["outcome"] not in VALID_STEP_OUTCOMES:
                    out({"ok": False, "error": f"`outcome` must be a string, one of "
                                                f"{sorted(VALID_STEP_OUTCOMES)} — put a richer payload "
                                                "(ids, scores, confirmation text) in the `result` field "
                                                "instead"})
                    return

        # --- backfill: merge into the existing record instead of ------------
        # --- appending a near-duplicate --------------------------------------
        if is_backfill:
            merged = _merge_backfill(tdir, rec, args.image_b64)
            if merged is not None:
                out({"ok": True, "backfilled": True, "log": merged["log"], "warnings": []})
                return
            # Nothing matched (task_id/step typo, or genuinely the first
            # record for this step) — log it fresh below, but say so; a
            # backfill call that can't find its target is more often a
            # mistake than a new step. From here on this *is* a fresh
            # record — `is_backfill` is cleared so every v3 requirement
            # below applies exactly as it would to a first-time call,
            # instead of silently accepting a partial patch as a whole
            # episode step because it happened to arrive via `backfill:true`.
            rec["backfill_target_not_found"] = True
            is_backfill = False

        # --- v3 record shape (trace-schema-v3.md §2/§8): action_type/coords/ -
        # --- element/etc at the top level are retired in favour of nested ---
        # --- `action`{kind,...} and `observation`{url,screenshot,a11y,...} --
        action = rec.get("action") if isinstance(rec.get("action"), dict) else None
        observation = rec.get("observation") if isinstance(rec.get("observation"), dict) else None
        is_reserved_step = step in (START_STEP, END_STEP)

        if rec.get("action_type") and not is_reserved_step:
            out({"ok": False, "error": "`action_type` (and top-level coords/element/bbox) are retired "
                                        "as of trace_schema 3 — use action: {kind: \"click\"|\"type\"|"
                                        "\"select\"|\"key\"|\"scroll\"|\"navigate\"|\"read\"|\"tool_call\"|"
                                        "\"ask_human\"|\"stop\", ...} instead (trace-schema-v3.md §2). "
                                        "v2's `verify`/`na` are gone — a look that doesn't act is a `read`."})
            return
        # driver only means something for an action that actually touches the
        # browser — tool_call is a jobagent CLI lookup and ask_human/stop are
        # meta-actions, none tied to a driver's coordinate semantics.
        if action is not None and not is_reserved_step and not rec.get("driver") \
                and action.get("kind") not in ("tool_call", "ask_human", "stop"):
            out({"ok": False, "error": "driver is required on any step with a browser-facing `action` — "
                                        "cdp, claude_in_chrome, browsermcp and claude_browser_pane use "
                                        "different coordinate semantics, and an unlabeled mix corrupts "
                                        "training"})
            return

        if step == END_STEP:
            # The last transition into the episode's close has to be closed
            # too — v3 has no `observation_after`, so _task_end's own
            # observation *is* how the final state gets recorded at all.
            if observation is None:
                out({"ok": False, "error": "_task_end requires a final `observation` block (url/title at "
                                            "minimum) — trace-schema-v3.md §3/§8, so the last state before "
                                            "the episode closes isn't a silent gap"})
                return
        elif not is_reserved_step:
            if action is None:
                out({"ok": False, "error": "every non-reserved step needs an `action` block — "
                                            "{kind: ..., ...}, see trace-schema-v3.md §2"})
                return
            kind = action.get("kind")
            if kind not in VALID_ACTION_KINDS:
                out({"ok": False, "error": f"unknown action.kind {kind!r} — expected one of "
                                            f"{sorted(VALID_ACTION_KINDS)}"})
                return
            if not rec.get("plan_step"):
                out({"ok": False, "error": "`plan_step` is required on every non-reserved step "
                                            "(trace-schema-v3.md §6/§8) — the current sub-goal this "
                                            "action serves; without it only an executor is trainable, "
                                            "never the planner above it"})
                return
            if kind == "tool_call":
                tool = action.get("tool")
                if not isinstance(tool, dict) or not tool.get("name") or "args" not in tool:
                    out({"ok": False, "error": "action.kind 'tool_call' requires action.tool.name and "
                                                "action.tool.args (trace-schema-v3.md §5)"})
                    return
            else:
                # Every non-tool_call action needs the page state it acted on —
                # this is the block that was 0/199 in the pre-v3 corpus.
                if observation is None:
                    out({"ok": False, "error": "an `observation` block is required on every step except "
                                                "tool_call (trace-schema-v3.md §1/§8) — url/title at "
                                                "minimum"})
                    return
                if not observation.get("a11y") or not observation.get("page_text"):
                    out({"ok": False, "error": "observation.a11y and observation.page_text are required — "
                                                "paths to the read_page/get_page_text output for this "
                                                "state, saved to their own files (see "
                                                "plugin/references/browser-adapter.md for why these can't "
                                                "be inlined) — trace-schema-v3.md §1/§8"})
                    return
                if (action.get("coords") or action.get("bbox")) and not (
                        observation.get("viewport") and observation.get("scroll") is not None
                        and observation.get("page_dims")):
                    out({"ok": False, "error": "observation.viewport, observation.scroll and "
                                                "observation.page_dims are all required whenever "
                                                "action.coords or action.bbox is set — without `scroll` a "
                                                "bbox names a place on the screen, not on the page "
                                                "(trace-schema-v3.md §4/§8)"})
                    return

        # A coordinate or bounding box with no screenshot behind it can't
        # train anything — this used to be a warning that shipped in the
        # response and was never read; 53 of 59 real records had no
        # screenshot despite the playbook saying to attach one every time.
        if action is not None and action.get("kind") in ("click", "type", "select") and step != END_STEP:
            if not _has_screenshot(rec) and not args.image_b64:
                out({"ok": False, "error": "click/type/select steps require a screenshot — pass "
                                            "--image-b64 (or --image-b64 @<path> for a large one)"})
                return

        # task_id groups the steps of one episode (a single application, a
        # single outreach send, a single reply) so trajectories can be
        # replayed in order later — not just isolated (state, action) pairs.
        # step_index is auto-assigned: count of prior records with the same
        # task_id, so callers never have to track it themselves or risk two
        # concurrent tasks colliding on a number. Scanned across *every* day
        # file, not just today's — a task started on one day (e.g. an invite
        # sent, followed up days later, or a backfilled record added after
        # the fact like this one) still gets a correct running count instead
        # of restarting at 0 whenever the day rolls over.
        if rec.get("task_id") and "step_index" not in rec:
            idx = 0
            for fp in _trace_files(tdir):
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if json.loads(line).get("task_id") == rec["task_id"]:
                            idx += 1
                    except json.JSONDecodeError:
                        pass
            rec["step_index"] = idx
            # step_index at this point is exactly "how many steps this task
            # already had" — which is also its total step count, so the
            # episode-summary record can report it for free.
            if rec.get("step") == END_STEP and "total_steps" not in rec:
                rec["total_steps"] = idx

        # Attach the screenshot before validating it — screenshot_dims below
        # is largely auto-derived from the actual decoded image, so the dims
        # check has to run after the image exists, not before.
        if args.image_b64:
            _target_dict_for_screenshot(rec).update(_save_screenshot(tdir, args.image_b64, rec["day"]))
            observation = rec.get("observation") if isinstance(rec.get("observation"), dict) else observation

        # One coordinate frame, stated explicitly (trace-schema-v3.md §4):
        # a screenshot with no dims, or dims that silently disagree with the
        # viewport, is exactly how v2 ended up with coords recorded in a
        # 1568×744 frame against 784×372 images — unusable, and undetectable
        # without this check.
        if observation and observation.get("screenshot"):
            dims = observation.get("screenshot_dims")
            if not dims:
                out({"ok": False, "error": "observation.screenshot_dims is required alongside "
                                            "observation.screenshot — auto-computed for PNG/JPEG; pass it "
                                            "explicitly for gif/webp (trace-schema-v3.md §4/§8)"})
                return
            vp = observation.get("viewport")
            if vp and dims != vp and not observation.get("scale"):
                out({"ok": False, "error": f"observation.screenshot_dims {dims} doesn't match "
                                            f"observation.viewport {vp} — set `scale`, or capture the "
                                            "screenshot at native resolution (trace-schema-v3.md §4)"})
                return

        # coords_norm/bbox_norm stay auto-derived so examples are comparable
        # across runs with different window sizes — now reading action{}/
        # observation{} (or the legacy top-level fields on a backfilled v2
        # record).
        _apply_norms(rec)

        # Best-effort quality nudges, returned but never blocking — a run
        # should never fail or slow down because a trace record is thin.
        # The hard requirements above cover what the corpus measurably needs;
        # what's left here is genuinely optional context that improves a
        # record without making it unusable.
        warnings = []
        if task_note:
            warnings.append(task_note)
        if action is not None and action.get("kind") in ("click", "type", "select") and step != END_STEP:
            if not action.get("coords") and not action.get("bbox"):
                warnings.append("no action.coords/bbox — unusable for click-prediction training")
            if not action.get("element"):
                warnings.append("no action.element{role,name,ref} — consider adding for grounding label quality")
            if not action.get("candidates"):
                warnings.append("no action.candidates — the other elements seen and not picked teach a "
                                 "grounding model to discriminate, not just imitate")
        if not is_reserved_step and not rec.get("expected_effect") and not (action and action.get("expected_effect")):
            warnings.append("no expected_effect — comparing it to the next record's observation is how "
                             "a failure gets auto-labelled instead of requiring a second annotation pass")
        if rec.get("step") == START_STEP and rec.get("step_index") != 0:
            warnings.append(f"_task_start has step_index {rec.get('step_index')}, not 0 — "
                             "log it before any other step for this task_id")
        if rec.get("step") not in (START_STEP, END_STEP) and rec.get("step_index") == 0 \
                and not rec.get("instruction"):
            warnings.append("first step of this task has no instruction and isn't _task_start — "
                             "consider a _task_start record naming what authorized this episode")

        day_dir = tdir / rec["day"]
        day_dir.mkdir(parents=True, exist_ok=True)
        log_path = day_dir / f"{_session_stamp(con, rec)}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
        shot = observation.get("screenshot") if observation else rec.get("screenshot")
        out({"ok": True, "log": str(log_path), "screenshot": shot, "warnings": warnings})
        return

    if args.action == "stats":
        by_day, by_skill, by_step, by_driver = {}, {}, {}, {}
        total = 0
        task_ids, started_task_ids, ended_task_ids, untasked = set(), set(), set(), 0
        actionable, actionable_with_coords, actionable_with_element = 0, 0, 0
        retried_steps = 0
        steps_with_retrieval, retrieval_overrides = 0, 0
        # Records written before this fix carry no `trace_schema` at all —
        # call that "v1_legacy" so a training pull can exclude it by default
        # instead of silently mixing thinner, unvalidated rows in with the
        # rest.
        by_schema = {}
        for fp in _trace_files(tdir):
            n = 0
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sk = rec.get("skill", "unknown")
                by_skill[sk] = by_skill.get(sk, 0) + 1
                st = rec.get("step", "unknown")
                by_step[st] = by_step.get(st, 0) + 1
                drv = rec.get("driver", "unset")
                by_driver[drv] = by_driver.get(drv, 0) + 1
                sv = str(rec.get("trace_schema") or "v1_legacy")
                by_schema[sv] = by_schema.get(sv, 0) + 1
                tid = rec.get("task_id")
                if tid:
                    task_ids.add(tid)
                    if st == START_STEP:
                        started_task_ids.add(tid)
                    if st == END_STEP:
                        ended_task_ids.add(tid)
                else:
                    untasked += 1
                # v3 records carry these nested under `action`; v2 (and a
                # backfilled v2 record) still has them at the top level —
                # check both so grounding_coverage doesn't silently zero out
                # the moment the corpus is all schema 3.
                act = rec.get("action") if isinstance(rec.get("action"), dict) else {}
                a_kind = act.get("kind") or rec.get("action_type")
                a_coords = act.get("coords") or rec.get("coords")
                a_bbox = act.get("bbox") or rec.get("bbox")
                a_element = act.get("element") or rec.get("element")
                if a_kind in ("click", "type", "select") and st != END_STEP:
                    actionable += 1
                    if a_coords or a_bbox:
                        actionable_with_coords += 1
                    if a_element:
                        actionable_with_element += 1
                if (rec.get("attempt") or 1) > 1:
                    retried_steps += 1
                retrieval = rec.get("retrieval")
                if isinstance(retrieval, list) and retrieval:
                    steps_with_retrieval += 1
                    if any(isinstance(r, dict) and r.get("accepted") is False for r in retrieval):
                        retrieval_overrides += 1
            day_key = fp.parent.name if fp.parent != tdir else fp.stem
            by_day[day_key] = by_day.get(day_key, 0) + n
            total += n
        top_steps = dict(sorted(by_step.items(), key=lambda kv: -kv[1])[:15])
        out({
            "total": total, "by_day": by_day, "by_skill": by_skill,
            "top_steps": top_steps, "distinct_step_names": len(by_step),
            "by_driver": by_driver, "by_schema": by_schema,
            "tasks": len(task_ids),
            "tasks_with_start_record": len(started_task_ids), "tasks_with_end_record": len(ended_task_ids),
            "avg_steps_per_task": round(total / len(task_ids), 1) if task_ids else None,
            "steps_without_task_id": untasked,
            "retried_steps": retried_steps,
            "grounding_coverage": {
                "actionable_steps": actionable,
                "with_coords_or_bbox": actionable_with_coords,
                "with_element_info": actionable_with_element,
                "pct_with_coords": round(100 * actionable_with_coords / actionable, 1) if actionable else None,
            },
            "retrieval_coverage": {
                "steps_with_retrieval": steps_with_retrieval,
                "retrieval_overrides": retrieval_overrides,
            },
            # The gap this whole checklist exists to close was invisible in
            # these stats: the first three days logged 154 records and not
            # one of them came from the search/discovery phase or from
            # `score_job` — the decision that gates every application. Record
            # count and grounding coverage both looked healthy. Naming the
            # canonical steps that have *zero* records makes a missing phase
            # visible without anyone re-auditing the corpus by hand.
            "never_logged_steps": sorted(VALID_STEPS - set(by_step)),
            "off_vocabulary_steps": sorted(set(by_step) - VALID_STEPS),
            "dir": str(tdir),
        })
        return

    if args.action == "list":
        entries = [fp.stem for fp in sorted(tdir.glob("*.jsonl"))]  # legacy flat <day>.jsonl files
        for day_dir in sorted(p for p in tdir.glob("*")
                              if p.is_dir() and p.name not in ("screenshots", "pages")):
            sessions = sorted(fp.stem for fp in day_dir.glob("*.jsonl"))
            if sessions:
                entries.append({"day": day_dir.name, "sessions": sessions})
        out({"days": entries, "dir": str(tdir)})
        return

    if args.action == "task":
        if not args.task_id:
            out({"ok": False, "error": "--task-id is required"})
            return
        steps = []
        for fp in _trace_files(tdir):
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("task_id") == args.task_id:
                    steps.append(rec)
        steps.sort(key=lambda r: (r.get("step_index", 0), r.get("ts", "")))
        out({"task_id": args.task_id, "steps": steps})
        return


def cmd_harvest(con, args):
    """Build a fine-tuning corpus from Claude Code's own session transcripts.

    `trace log` records what the agent remembered to write down; this records
    what actually happened. Measured on two real runs, that is 123 browser
    actions against 11 hand-written records.

    Reads `~/.claude/projects/**` directly — no copying files around — plus
    `data/traces/` for the skill/step/task_id labels and the screenshots.
    """
    from . import harvest as H

    m = H.run(
        claude_dir=args.claude_dir,
        project_root=args.project_root,
        transcript=args.transcript,
        session=args.session,
        db_path=args.db,
        out_dir=args.out,
        fmt=args.format,
        kinds=args.kind or None,
        max_chars=args.max_chars,
        do_redact=not args.no_redact,
        copy_images=args.copy_images,
        include_parent=not args.subagents_only,
    )
    out(m)


def cmd_browse(con, args):
    """Drive the user's own Chrome over CDP — see `jobagent/browser.py`.

    Deliberately does not touch `con`. This group reads and acts on a live
    page; anything worth remembering goes through the other groups (`job`,
    `answers`, `trace`) so there is still exactly one place state lives.

    The whole reason this exists: `snapshot`-style calls return the entire
    accessibility tree and blow past the host's tool-output cap on almost
    every LinkedIn page. Here the page is searched in-process and only the
    answer is printed.
    """
    from . import browser as B

    B.set_timeout(args.timeout)
    a = args.action
    kw = {"cdp_url": args.cdp, "tab": args.tab}

    if a == "status":
        return out(B.status(**kw))
    if a == "open":
        if not args.url:
            raise ValueError("browse open needs --url")
        return out(B.open_url(args.url, wait=args.wait, new_tab=args.new_tab, **kw))
    if a == "find":
        if not args.query:
            raise ValueError("browse find needs --query")
        return out(B.find(args.query, limit=args.limit or 5, scoped=not args.whole_page,
                          db_path=args.db, **kw))
    if a == "outline":
        return out(B.outline(limit=args.limit or 30, chars=args.chars or 120,
                             db_path=args.db, **kw))
    if a == "form":
        return out(B.form(limit=args.limit or 60, db_path=args.db, **kw))
    if a == "click":
        return out(B.click(ref=args.ref, text=args.query, db_path=args.db, **kw))
    if a == "fill":
        if not (args.ref and args.value is not None):
            raise ValueError("browse fill needs --ref and --value")
        return out(B.fill(args.ref, args.value, blur=not args.no_blur, db_path=args.db, **kw))
    if a == "select":
        if not (args.ref and args.value):
            raise ValueError("browse select needs --ref and --value")
        return out(B.select(args.ref, args.value, db_path=args.db, **kw))
    if a == "check":
        if not args.ref:
            raise ValueError("browse check needs --ref")
        return out(B.check(args.ref, on=not args.off, db_path=args.db, **kw))
    if a == "upload":
        if not (args.ref and args.path):
            raise ValueError("browse upload needs --ref and --path")
        return out(B.upload(args.ref, args.path, db_path=args.db, **kw))
    if a == "press":
        if not args.key:
            raise ValueError("browse press needs --key")
        return out(B.press(args.key, **kw))
    if a == "scroll":
        return out(B.scroll(dy=args.dy, **kw))
    if a == "text":
        return out(B.text(selector=args.selector, limit=args.limit or 4000, **kw))
    if a == "shot":
        return out(B.shot(label=args.label or "", db_path=args.db, full=args.full, **kw))
    if a == "dump":
        if not args.label:
            raise ValueError("browse dump needs --label")
        return out(B.dump(args.label, db_path=args.db, **kw))
    raise ValueError(f"unknown browse action: {a}")


# ------------------------------------------------------------------ main ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jobagent", description="LinkedIn job agent state store")
    p.add_argument("--db", help="path to state.db")
    sub = p.add_subparsers(dest="group", required=True)

    sub.add_parser("init", help="create/upgrade the database and seed questions")

    g = sub.add_parser("profile"); g.add_argument("action", choices=["get", "set"])
    g.add_argument("--json", default="{}")

    g = sub.add_parser("prefs"); g.add_argument("action", choices=["get", "set"])
    g.add_argument("--json", default="{}")

    g = sub.add_parser("config"); g.add_argument("action", choices=["get", "set"])
    g.add_argument("--json", default="{}")

    g = sub.add_parser("answers")
    g.add_argument("action", choices=["set", "get", "list", "missing", "resolve"])
    g.add_argument("--key"); g.add_argument("--value"); g.add_argument("--question")
    g.add_argument("--kind"); g.add_argument("--category"); g.add_argument("--alias", action="append")
    g.add_argument("--options", help="pipe-separated options the form offers")
    g.add_argument("--threshold", type=float, default=0.42)
    g.add_argument("--max-chars", type=int, help="field's character limit, if it has one")

    g = sub.add_parser("company")
    g.add_argument("action", choices=["upsert", "get", "list", "blacklist"])
    g.add_argument("--name"); g.add_argument("--json"); g.add_argument("--status")

    g = sub.add_parser("job")
    g.add_argument("action", choices=["seen", "upsert", "score", "queue", "applied", "skip", "get", "list"])
    g.add_argument("--job-id"); g.add_argument("--id", type=int); g.add_argument("--json")
    g.add_argument("--min-score", type=int); g.add_argument("--limit", type=int)
    g.add_argument("--method", default="easy_apply"); g.add_argument("--resume")
    g.add_argument("--answers"); g.add_argument("--screening"); g.add_argument("--notes")
    g.add_argument("--reason"); g.add_argument("--status")

    g = sub.add_parser("contact")
    g.add_argument("action", choices=["seen", "upsert", "mark", "pending-invites",
                                      "awaiting-followup", "list"])
    g.add_argument("--url"); g.add_argument("--json"); g.add_argument("--relation")
    g.add_argument("--company"); g.add_argument("--role-type"); g.add_argument("--limit", type=int)
    g.add_argument("--older-than-days", type=int)

    g = sub.add_parser("outreach")
    g.add_argument("action", choices=["draft", "pending", "approve", "reject", "next",
                                      "sent", "fail", "list"])
    g.add_argument("--contact-url"); g.add_argument("--kind"); g.add_argument("--job-id")
    g.add_argument("--body"); g.add_argument("--template"); g.add_argument("--extra")
    g.add_argument("--attachment"); g.add_argument("--id", type=int)
    g.add_argument("--all", action="store_true"); g.add_argument("--force", action="store_true")
    g.add_argument("--reason"); g.add_argument("--error"); g.add_argument("--status")
    g.add_argument("--limit", type=int)

    g = sub.add_parser("reply")
    g.add_argument("action", choices=["add", "unnotified", "mark-notified", "handled", "list"])
    g.add_argument("--contact-url"); g.add_argument("--outreach-id", type=int)
    g.add_argument("--thread-url"); g.add_argument("--snippet"); g.add_argument("--sentiment")
    g.add_argument("--received-at"); g.add_argument("--id", type=int)
    g.add_argument("--all", action="store_true"); g.add_argument("--limit", type=int)

    g = sub.add_parser("run")
    g.add_argument("action", choices=["start", "quota", "end", "history"])
    g.add_argument("--notes"); g.add_argument("--limit", type=int)

    g = sub.add_parser("report")
    g.add_argument("action", choices=["daily", "pipeline", "followups"])

    g = sub.add_parser("template")
    g.add_argument("action", choices=["list", "get", "set", "render"])
    g.add_argument("--name"); g.add_argument("--body"); g.add_argument("--contact-url")
    g.add_argument("--job-id"); g.add_argument("--extra")

    g = sub.add_parser("trace", help="append (page state, instruction, action) records for a future local-model fine-tune")
    g.add_argument("action", choices=["log", "stats", "list", "task", "enable", "disable", "steps"])
    g.add_argument("--json", default="{}",
                    help="v3 record shape (trace-schema-v3.md) — top-level: skill, step, task_id, "
                         "attempt, goal, plan_step (required on every non-reserved step), instruction, "
                         "rationale, outcome, driver, confidence, result{...}, "
                         "retrieval[{source,query,result,accepted,override_reason}]. "
                         "observation{url,title,viewport{width,height},scroll{x,y},page_dims{width,height},"
                         "a11y,page_text,screenshot,screenshot_dims,scale,content_hash} — required on "
                         "every non-reserved step except action.kind='tool_call', and as the closing "
                         "state on _task_end; a11y/page_text are paths to saved read_page/get_page_text "
                         "output, not inline text. "
                         "action{kind,ref,element{role,name,ref},coords{x,y},bbox{x,y,w,h},"
                         "candidates[{role,name,bbox}],value,tool{name,args,result},question,reason,"
                         "expected_effect} — kind is one of "
                         "click|type|select|key|scroll|navigate|read|tool_call|ask_human|stop "
                         "(v2's action_type/verify/na are retired). "
                         "See browser-playbook.md §9 and plugin/references/trace-schema-v3.md. Reserved "
                         "steps: _task_start (first record, carries `source` not `outcome`, names the "
                         "instruction that opened this task_id, no action/observation), _task_end (last, "
                         "`outcome` is applied|sent|not_applied|error|aborted|skipped:<reason>, carries a "
                         "final `observation`, no `action`). `outcome` on any other step is restricted to "
                         "success|error_shown|blocked|na — put richer detail in `result`. `driver` is "
                         "required on any step with an `action` block and normalized to claude_in_chrome"
                         "|browsermcp|claude_browser_pane|cdp. `trace_schema`/`persona_id` are auto-filled. "
                         "Pass `backfill:true` to update an existing (task_id, step) record instead of "
                         "appending a duplicate — a partial patch is fine there (e.g. just `action.tool."
                         "result` or `retrieval`); if no matching record is found it falls back to a "
                         "fresh insert and the full v3 requirements apply. click/type/select steps "
                         "require a screenshot with screenshot_dims that agrees with viewport (or a "
                         "`scale`). `skill` and `step` are closed vocabularies — run `jobagent trace "
                         "steps` to see them; an unknown step is rejected rather than fragmenting the "
                         "corpus. Set config.trace_logging false (or `jobagent trace disable`) to make "
                         "`trace log` a reported no-op for a run.")
    g.add_argument("--image-b64", dest="image_b64",
                    help="base64 screenshot to save alongside the record (PNG/JPEG/GIF/WEBP all fine — "
                         "actual format is sniffed, not assumed). Attach one on every step now, not just "
                         "verification steps — see browser-playbook.md §9.")
    g.add_argument("--task-id", dest="task_id", help="for `trace task`: reconstruct one episode's steps in order")

    g = sub.add_parser("harvest",
                       help="build a fine-tuning corpus from Claude Code's session transcripts "
                            "plus data/traces — what actually happened, not what got logged")
    g.add_argument("--claude-dir", dest="claude_dir",
                   help="Claude Code config dir (default ~/.claude, or $CLAUDE_CONFIG_DIR). "
                        "Under WSL this is the Linux home, not C:\\Users")
    g.add_argument("--project-root", dest="project_root",
                   help="only sessions whose cwd is this folder (default: the current one). "
                        "Matched against each transcript's own `cwd`, so the slug rule never "
                        "has to be guessed")
    g.add_argument("--transcript", help="a specific .jsonl, or a folder of them, instead of "
                                        "discovering from ~/.claude")
    g.add_argument("--session", help="only sessions whose id contains this")
    g.add_argument("--out", help="output folder (default data/sft/)")
    g.add_argument("--format", default="steps", choices=["steps", "messages"],
                   help="`steps` = the full record per action; `messages` = chat-shaped "
                        "system/user/assistant rows ready for SFT")
    g.add_argument("--kind", action="append",
                   help="keep only these action kinds — `browse` for every browser action, "
                        "or an exact kind like browse.click. Repeatable")
    g.add_argument("--max-chars", dest="max_chars", type=int, default=8000,
                   help="cap on observation/result text per record (default 8000); the full "
                        "text stays on disk and `chars` records its real length")
    g.add_argument("--no-redact", dest="no_redact", action="store_true",
                   help="keep phone numbers, emails and salary figures verbatim. Off by "
                        "default — AGENTS.md forbids putting them in the corpus")
    g.add_argument("--copy-images", dest="copy_images", action="store_true",
                   help="copy referenced screenshots into <out>/images/ so the dataset is "
                        "self-contained")
    g.add_argument("--subagents-only", dest="subagents_only", action="store_true",
                   help="skip parent sessions; the real work happens in the sub-agent")

    g = sub.add_parser("browse",
                       help="drive your own already-logged-in Chrome over CDP, returning small answers "
                            "instead of whole-page dumps")
    g.add_argument("action", choices=["status", "open", "outline", "find", "form", "click", "fill",
                                      "select", "check", "upload", "press", "scroll", "text",
                                      "shot", "dump"])
    g.add_argument("--url", help="for `open`")
    g.add_argument("--query", "-q", help="text to look for (`find`), or what to click (`click --query`)")
    g.add_argument("--ref", help="a ref from a previous `find`/`form` (e3, f2, b1)")
    g.add_argument("--value", help="text to type (`fill`) or option to choose (`select`)")
    g.add_argument("--path", help="file to attach, for `upload`")
    g.add_argument("--label", help="a name for the saved screenshot/page dump, e.g. ibm-attach_resume")
    g.add_argument("--selector", help="CSS selector to scope `text` to")
    g.add_argument("--key", help="key to press, e.g. Escape or Enter")
    g.add_argument("--limit", type=int, help="max matches (`find`, default 5), rows (`outline`, 30), "
                                             "fields (`form`, 60), or characters (`text`, 4000)")
    g.add_argument("--chars", type=int, help="for `outline`: characters per row (default 120)")
    g.add_argument("--dy", type=int, default=600, help="pixels to scroll, negative for up")
    g.add_argument("--tab", help="substring of the tab URL to drive (default: a linkedin.com tab)")
    g.add_argument("--cdp", help="Chrome debugging endpoint (default http://127.0.0.1:9222, "
                                "or $JOBAGENT_CDP)")
    g.add_argument("--wait", default="domcontentloaded",
                   choices=["load", "domcontentloaded", "networkidle", "commit"])
    g.add_argument("--timeout", type=int,
                   help="per-call timeout in ms (default 15000, or "
                        "$JOBAGENT_BROWSE_TIMEOUT_MS). Raise it on a slow "
                        "connection; a click already retries once on timeout")
    g.add_argument("--new-tab", dest="new_tab", action="store_true",
                   help="for `open`: open a new tab instead of navigating the current one")
    g.add_argument("--whole-page", dest="whole_page", action="store_true",
                   help="search the whole document instead of the open dialog")
    g.add_argument("--no-blur", dest="no_blur", action="store_true",
                   help="skip the Tab that blurs the field — LinkedIn only validates on blur, so "
                        "you almost never want this")
    g.add_argument("--off", action="store_true", help="for `check`: uncheck instead")
    g.add_argument("--full", action="store_true",
                   help="for `shot`: capture the full page. Off by default because click coordinates "
                        "are viewport-relative and a full-page image puts them in a different frame")

    return p


DISPATCH = {
    "init": cmd_init, "profile": cmd_profile, "prefs": cmd_prefs, "config": cmd_config,
    "answers": cmd_answers, "company": cmd_company, "job": cmd_job, "contact": cmd_contact,
    "outreach": cmd_outreach, "reply": cmd_reply, "run": cmd_run, "report": cmd_report,
    "template": cmd_template, "trace": cmd_trace, "browse": cmd_browse,
    "harvest": cmd_harvest,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    con = D.connect(args.db)
    try:
        DISPATCH[args.group](con, args)
    except Exception as e:  # noqa: BLE001 — the caller is an LLM; give it the message
        out({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
