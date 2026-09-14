"""Outreach copy.

Short, specific, and never the same paragraph twice in a row — the fastest way
to get ignored (or reported) on LinkedIn is a message that reads like a mail
merge. Placeholders in {braces} are filled from the contact/job/profile rows;
the skill is instructed to rewrite one detail per message by hand.

Hard limits LinkedIn enforces:
  * connection request note: 300 characters (we cap at 280 for safety)
  * InMail / message: 8000 characters
"""

from __future__ import annotations

import re

INVITE_LIMIT = 280
MESSAGE_LIMIT = 8000

DEFAULTS: dict[str, str] = {
    # --- connection notes (<= 280 chars) ---------------------------------
    "invite_hr": (
        "Hi {first_name} — I saw the {job_title} opening at {company}. "
        "I'm a {my_title} with {years_phrase} in {top_skill}. "
        "Would love to connect and share my profile if you're open to it."
    ),
    "invite_employee": (
        "Hi {first_name} — I'm exploring {job_title} roles and {company} keeps "
        "coming up. I work on {top_skill} as a {my_title}. "
        "Would be great to connect and hear what the team's like."
    ),
    "invite_alumni": (
        "Hi {first_name} — I'm a {my_title} working on {top_skill}, and I'm "
        "looking at {job_title} roles at {company}. Would be great to connect."
    ),
    "invite_mutual": (
        "Hi {first_name} — I've been following {company} and wanted to reach "
        "out. I'm a {my_title} in {top_skill}, currently exploring {job_title} "
        "roles. Would be glad to connect."
    ),

    # --- messages after a connection is accepted -------------------------
    "hr_pitch": (
        "Hi {first_name}, thanks for connecting.\n\n"
        "I'm reaching out about the {job_title} role at {company}{job_url_paren}. "
        "I'm a {my_title} with {years_phrase} across {top_skills}, "
        "and {one_specific_reason}.\n\n"
        "{resume_note} — happy to share anything else that helps. "
        "Is the role still open, and would it make sense to talk this week?\n\n"
        "Thanks,\n{my_name}"
    ),
    "referral_ask": (
        "Hi {first_name}, thanks so much for connecting.\n\n"
        "I recently applied for the {job_title} role at {company}{job_url_paren}. "
        "A little about me: {my_title}, {years_phrase}, mostly working with "
        "{top_skills}{reason_clause}.\n\n"
        "If you'd be willing to refer me, I'd genuinely appreciate it — and no "
        "worries at all if that's not something you're able to do. {resume_offer}"
        "Thanks again for connecting, and please don't hesitate to reach out if "
        "there's ever anything I can help you with.\n\n{my_name}"
    ),
    "referral_ask_existing": (
        "Hi {first_name}, hope you're doing well!\n\n"
        "I saw that {company} has an opening for {job_title}{job_url_paren} and "
        "wanted to reach out. I've spent {years_phrase} working on "
        "{top_skills}{reason_clause}.\n\n"
        "If you'd be willing to refer me, I'd be really grateful — though I "
        "completely understand if it's not something you're able to help with. "
        "{resume_offer}Either way, please don't hesitate to reach out if there's "
        "ever anything I can help you with too.\n\n{my_name}"
    ),
    "intro_peer": (
        "Hi {first_name}, thanks for connecting.\n\n"
        "I'm looking at {job_title} roles and {company} is high on my list. "
        "If you have five minutes sometime, I'd love to hear what the team "
        "actually works on day to day — and if there's someone on the hiring "
        "side worth speaking to, I'd appreciate a pointer.\n\n"
        "Thanks,\n{my_name}"
    ),
    "followup_nudge": (
        "Hi {first_name} — just floating this back up in case it got buried. "
        "Still very interested in {job_title} at {company}. "
        "Happy to step back if the timing isn't right.\n\nThanks,\n{my_name}"
    ),
    "thanks_reply": (
        "Thanks so much, {first_name} — I really appreciate it. "
        "{next_step}\n\n{my_name}"
    ),
}

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def render(template: str, ctx: dict) -> str:
    """Fill placeholders; unknown ones are dropped along with dangling spaces."""
    def sub(m: re.Match) -> str:
        return str(ctx.get(m.group(1), "") or "")

    out = _PLACEHOLDER.sub(sub, template)
    # Safety net: any placeholder that resolved empty can still leave a
    # dangling "()" or "( )" behind (e.g. a template using {job_url} directly
    # instead of the pre-wrapped {job_url_paren}) — drop those rather than
    # ship a message with empty parens in it.
    out = re.sub(r"\s?\(\s*\)", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r" +([,.!?])", r"\1", out)
    return out.strip()


def fits(kind: str, body: str) -> tuple[bool, int, int]:
    limit = INVITE_LIMIT if kind.startswith("invite") else MESSAGE_LIMIT
    return len(body) <= limit, len(body), limit


def trim(kind: str, body: str) -> str:
    ok, length, limit = fits(kind, body)
    if ok:
        return body
    cut = body[: limit - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-") + "."
