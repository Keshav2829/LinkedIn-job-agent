---
name: linkedin-followup
description: Checks which LinkedIn connection requests were accepted and sends the real message — a resume and role enquiry to recruiters, a referral ask to everyone else — plus nudges on stale invites and silent applications. Use when the user asks to follow up, check if anyone accepted, send referral requests, or chase applications.
---

# Follow-up

The invite gets you in the door; this is the part that actually asks for
something. It runs the loop the user would otherwise have to remember: who
accepted, who went quiet, what's owed a nudge.

## 1. See what's due

```bash
python -m jobagent report followups
```

Three buckets come back:

- **`accepted_awaiting_message`** — connected, never messaged. **Highest value
  work in the whole agent.** Do these first.
- **`stale_invites`** — sent over `config.invite_followup_days` (default 7) ago,
  still pending.
- **`silent_applications`** — applied over `config.application_followup_days`
  (default 10) ago with no reply.

## 2. Detect new acceptances

The store only knows what you tell it, so reconcile against LinkedIn:

```bash
python -m jobagent contact pending-invites
```

Open `https://www.linkedin.com/mynetwork/invitation-manager/sent/`. Anyone in
that list is still pending. Anyone the store thinks is pending but who has
*dropped off* the list has accepted, withdrawn, or expired — open their profile
to check for the "1st" badge and a **Message** button.

```bash
python -m jobagent contact mark --url https://www.linkedin.com/in/priya-sharma --relation connected
python -m jobagent contact mark --url https://www.linkedin.com/in/other-person --relation declined
```

Faster alternative when there are many: check `/mynetwork/invite-connect/connections/`
sorted by "Recently added" and match names against pending invites.

## 3. Message the people who accepted

Branch on `role_type`:

| Who they are | Kind | The ask |
|---|---|---|
| `hr` or `hiring_manager` | `hr_pitch` | Is the role open, here's my resume, can we talk |
| `employee` / `alumni` / `peer` **at a company we applied to** | `referral_ask` | Would you refer me — resume attached |
| `employee` / `alumni` / `peer`, no application yet | `intro_peer` | What's the team like, who should I speak to |

```bash
python -m jobagent outreach draft --contact-url <url> --kind hr_pitch --job-id 4021553311 \
  --attachment "data/resume.pdf" \
  --extra '{"one_specific_reason":"your posting mentions idempotent webhooks — I rebuilt exactly that at Acme last year"}'
```

Then approval (`outreach pending` → user) and send (`outreach next` →
browser-playbook §5). **Attach the resume** on `hr_pitch` and `referral_ask` —
verify the attachment chip shows the right filename before clicking Send.

A referral ask carries the other person's name and reputation. Say so in one
short line — "no pressure at all if you'd rather not" — and mean it. Never send
a second referral ask to someone who didn't answer the first.

## 4. Nudge, sparingly

**Stale invites**: do nothing automatic. Report them, and offer to withdraw
invites over three weeks old so the quota is freed. Withdrawal is a
confirm-dialog action — hand it to the user rather than clicking it.

**Silent applications**: only worth a nudge if there's a connected contact at
that company. Draft `--kind followup --template followup_nudge`. One nudge per
application, ever. If there's no contact there, the better move is outreach, not
a nudge — say that.

**Someone who accepted but never replied to the message**: leave them alone.
One message, one nudge after ten days at most, then stop. Mark
`contact mark --relation connected` and move on — persistence past that point
costs the user their reputation.

## 5. Report

Lead with the acceptances — those are the wins. Then: messages queued for
approval, nudges suggested, and anything that needs a human decision (a
withdrawal, an odd reply, a contact who only accepts follows).

## Stop conditions

This skill drives the browser too — the same five abort signals in
browser-playbook §0 apply (captcha, invite limit, restriction banner,
logged-out state). Stop immediately, log it (§0/§9), and tell the user.
