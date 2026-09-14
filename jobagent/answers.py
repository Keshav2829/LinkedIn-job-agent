"""The answer bank.

LinkedIn Easy Apply asks the same forty-odd questions in a hundred different
phrasings. This module stores one canonical answer per question and resolves
any phrasing back to it, so the agent fills forms without asking you again —
and knows, honestly, when it does not know.
"""

from __future__ import annotations

import re

from .matching import tokens

# Canonical question registry. `setup` seeds these into the answers table and
# asks you for the values it doesn't already have from your profile/resume.
CANONICAL: list[dict] = [
    # -- identity ---------------------------------------------------------
    dict(key="full_name", category="identity", kind="text",
         question="Full name",
         aliases=["first and last name", "your name", "candidate name"]),
    dict(key="first_name", category="identity", kind="text", question="First name", aliases=["given name"]),
    dict(key="last_name", category="identity", kind="text", question="Last name", aliases=["surname", "family name"]),
    dict(key="email", category="identity", kind="text", question="Email address",
         aliases=["email", "e-mail", "contact email"]),
    dict(key="phone", category="identity", kind="text", question="Mobile phone number",
         aliases=["phone", "mobile", "contact number", "cell phone"]),
    dict(key="phone_country_code", category="identity", kind="choice",
         question="Phone country code", aliases=["country code"]),
    dict(key="city", category="identity", kind="text", question="Current city",
         aliases=["location", "where are you based", "current location", "city and state"]),
    dict(key="linkedin_url", category="identity", kind="text", question="LinkedIn profile URL",
         aliases=["linkedin profile", "linkedin"]),
    dict(key="github_url", category="identity", kind="text", question="GitHub / portfolio URL",
         aliases=["github", "portfolio", "personal website", "portfolio link"]),

    # -- work authorisation ------------------------------------------------
    dict(key="authorized_to_work", category="work_auth", kind="bool",
         question="Are you legally authorized to work in the country of this job?",
         aliases=["legally authorized to work", "authorised to work", "right to work",
                  "eligible to work", "work authorization"]),
    dict(key="requires_sponsorship", category="work_auth", kind="bool",
         question="Will you now or in the future require visa sponsorship?",
         aliases=["require sponsorship", "need sponsorship", "visa sponsorship",
                  "sponsorship for employment visa status"]),
    dict(key="legal_age", category="work_auth", kind="bool",
         question="Are you of legal working age (18 or older)?",
         aliases=["are you 18", "of legal age", "at least 18 years"]),
    dict(key="work_permit_country", category="work_auth", kind="text",
         question="Which countries are you authorized to work in?",
         aliases=["countries authorized", "citizenship", "nationality"]),

    # -- current employment -------------------------------------------------
    dict(key="current_company", category="employment", kind="text",
         question="Current employer", aliases=["present company", "current organization", "employer name"]),
    dict(key="current_title", category="employment", kind="text",
         question="Current job title", aliases=["present designation", "current role", "job title"]),
    dict(key="total_experience_years", category="employment", kind="number",
         question="Total years of professional experience",
         aliases=["years of experience", "total experience", "how many years of work experience"]),
    dict(key="notice_period", category="employment", kind="text",
         question="Notice period", aliases=["notice period in days", "how soon can you join",
                                            "availability to start", "earliest start date",
                                            "when can you start"]),
    dict(key="current_ctc", category="employment", kind="text",
         question="Current annual compensation (CTC)",
         aliases=["current salary", "current ctc", "present ctc", "current compensation"]),
    dict(key="expected_ctc", category="employment", kind="text",
         question="Expected annual compensation (CTC)",
         aliases=["expected salary", "expected ctc", "salary expectation",
                  "desired salary", "compensation expectation"]),
    dict(key="willing_to_relocate", category="employment", kind="bool",
         question="Are you willing to relocate?", aliases=["open to relocation", "relocate"]),
    dict(key="preferred_work_mode", category="employment", kind="choice",
         question="Preferred work mode (onsite / hybrid / remote)",
         aliases=["work from office", "hybrid", "remote preference", "comfortable working onsite"]),
    dict(key="willing_to_travel", category="employment", kind="text",
         question="Willingness to travel (%)", aliases=["travel requirement", "able to travel"]),

    # -- legal / compliance -------------------------------------------------
    dict(key="non_compete", category="legal", kind="bool",
         question="Are you subject to a non-compete agreement?",
         aliases=["non compete", "noncompete", "restrictive covenant"]),
    dict(key="confidentiality_agreement", category="legal", kind="bool",
         question="Are you subject to a confidentiality or non-disclosure agreement that would affect this role?",
         aliases=["confidentiality agreement", "nda", "non disclosure"]),
    dict(key="family_at_company", category="legal", kind="bool",
         question="Do you have a family member or relative employed at this company?",
         aliases=["relative working", "family member employed", "any relatives at"]),
    dict(key="previously_employed", category="legal", kind="bool",
         question="Have you previously been employed by this company?",
         aliases=["worked here before", "former employee", "previously worked for"]),
    dict(key="background_check_consent", category="legal", kind="bool",
         question="Do you consent to a background verification check?",
         aliases=["background check", "background verification"]),
    dict(key="criminal_record", category="legal", kind="bool",
         question="Have you ever been convicted of a criminal offence?",
         aliases=["criminal conviction", "convicted"]),
    dict(key="security_clearance", category="legal", kind="text",
         question="Do you hold a security clearance?", aliases=["clearance"]),
    dict(key="drivers_license", category="legal", kind="bool",
         question="Do you hold a valid driver's license?", aliases=["driving license", "driver licence"]),

    # -- EEO / voluntary ----------------------------------------------------
    dict(key="gender", category="eeo", kind="choice", question="Gender (voluntary)", aliases=["gender identity"]),
    dict(key="ethnicity", category="eeo", kind="choice", question="Race / ethnicity (voluntary)",
         aliases=["race", "ethnicity"]),
    dict(key="veteran_status", category="eeo", kind="choice", question="Veteran status (voluntary)",
         aliases=["protected veteran"]),
    dict(key="disability_status", category="eeo", kind="choice", question="Disability status (voluntary)",
         aliases=["disability", "differently abled"]),

    # -- free text ----------------------------------------------------------
    dict(key="why_this_role", category="narrative", kind="text",
         question="Why are you interested in this role? (template — the agent adapts it per job)",
         aliases=["why do you want to work", "cover letter", "tell us why", "motivation"]),
    dict(key="heard_about_us", category="narrative", kind="choice",
         question="How did you hear about this job?", aliases=["how did you hear", "source"]),
    dict(key="summary", category="narrative", kind="text",
         question="Professional summary / elevator pitch",
         aliases=["about you", "brief introduction", "tell us about yourself"]),
    dict(key="publications", category="narrative", kind="text",
         question="Do you have any publications?",
         aliases=["publications", "research papers", "published papers",
                  "list your publications", "any papers published"]),
]

CANONICAL_BY_KEY = {c["key"]: c for c in CANONICAL}

YES = {"yes", "y", "true", "1"}
NO = {"no", "n", "false", "0"}


# Forms say the same thing a dozen ways. Collapsing the vocabulary before
# comparing is worth far more than lowering the match threshold, which would
# start returning confident wrong answers.
SYNONYMS: list[tuple[str, str]] = [
    # Indian recruiting shorthand — these appear on a large share of postings
    # and match nothing in plain English. They must expand before the generic
    # ctc/salary rule below collapses the vocabulary.
    (r"\bcctc\b|\bc\.c\.t\.c\b", "current compensation"),
    (r"\bectc\b|\be\.c\.t\.c\b", "expected compensation"),
    (r"\bper annum\b|\bp\.?a\.?\b|\byearly\b|\bper year\b|\bannually\b", "annual"),
    (r"\blpa\b|\blakhs? per annum\b", "annual compensation"),
    (r"\bctc\b|\bsalary\b|\bremuneration\b|\bpackage\b|\bpay\b", "compensation"),
    (r"\bnotice\b", "notice period"),
    (r"\bjoin\b|\bjoining\b|\bonboard\b", "start"),
    (r"\bmobile\b|\bcontact number\b|\bcell\b", "phone"),
    (r"\byrs?\b", "years"),
    (r"\bexp\b", "experience"),
    (r"\bauthoris", "authoriz"),
    (r"\bwilling to\b|\bopen to\b|\bcomfortable\b", ""),
]


def normalize_question(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"\*|\(required\)|\(optional\)", " ", q)
    for pattern, repl in SYNONYMS:
        q = re.sub(pattern, repl, q)
    q = re.sub(r"\s+", " ", q)
    return q.strip(" ?:.")


def _phrase_score(question: str, phrase: str) -> float:
    """0..1 similarity between an asked question and a stored phrasing."""
    if not phrase:
        return 0.0
    q, p = normalize_question(question), normalize_question(phrase)
    if not q or not p:
        return 0.0
    if p in q or q in p:
        return 0.95 if len(p) > 8 else 0.75
    qt, pt = tokens(q), tokens(p)
    if not qt or not pt:
        return 0.0
    return len(qt & pt) / len(qt | pt)


def resolve(question: str, rows: list[dict], threshold: float = 0.42) -> dict | None:
    """Find the best stored answer for an arbitrary form question.

    `rows` are dicts from the answers table (key, question, value, aliases...).
    Returns the row with an added `confidence`, or None when nothing is close
    enough — in which case the caller must ask the human rather than guess.
    """
    best, best_score = None, 0.0
    for row in rows:
        phrases = [row.get("question", ""), row.get("key", "").replace("_", " ")]
        phrases += row.get("aliases", []) or []
        s = max((_phrase_score(question, p) for p in phrases), default=0.0)
        if s > best_score:
            best, best_score = row, s
    if best is None or best_score < threshold:
        return None
    out = dict(best)
    out["confidence"] = round(best_score, 3)
    return out


# "Notice period (In Days)" wants 60, not "60 days". Forms that name their
# unit want the bare number — and often cap the field at 20 characters.
NUMERIC_HINT = re.compile(r"\bin days\b|\bin years\b|\bin months\b|\bnumber of\b")


def shape(question: str, value: str, kind: str, options: list[str] | None = None,
          max_chars: int | None = None) -> tuple[str, bool]:
    """Fit a stored answer to one specific field. Returns (answer, truncated)."""
    v = coerce(value, kind, options)
    if not options and NUMERIC_HINT.search(normalize_question(question)):
        m = re.search(r"\d+(\.\d+)?", v)
        if m:
            v = m.group(0)
    if max_chars and len(v) > max_chars:
        # Drop parenthetical asides first — they're the least load-bearing part
        # of an answer like "45-50 LPA (negotiable)".
        stripped = re.sub(r"\s*\([^)]*\)", "", v).strip()
        if len(stripped) <= max_chars and stripped:
            return stripped, True
        return v[:max_chars].strip(), True
    return v, False


def coerce(value: str, kind: str, options: list[str] | None = None) -> str:
    """Shape a stored answer to what a given form control expects."""
    v = (value or "").strip()
    if kind == "bool":
        lowered = v.lower()
        truthy = lowered in YES
        if options:
            for o in options:
                if (o.lower() in YES) == truthy and o.lower() in YES | NO:
                    return o
            for o in options:
                if truthy and o.lower().startswith("y"):
                    return o
                if not truthy and o.lower().startswith("n"):
                    return o
        return "Yes" if truthy else "No"
    if kind == "number":
        m = re.search(r"\d+(\.\d+)?", v)
        return m.group(0) if m else v
    if options:
        for o in options:
            if o.strip().lower() == v.lower():
                return o
        for o in options:
            if v.lower() and v.lower() in o.lower():
                return o
    return v
