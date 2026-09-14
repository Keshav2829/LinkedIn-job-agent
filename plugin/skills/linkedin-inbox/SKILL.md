---
name: linkedin-inbox
description: Scans LinkedIn messages and notifications for replies to the agent's outreach, notifies the user, and drafts responses for approval. Use when the user asks whether anyone replied, to check LinkedIn messages, or to see job-search responses.
---

# Inbox watch

Outreach is worthless if a reply sits unread for four days. This skill scans,
triages, notifies — and never answers on the user's behalf without approval.

## 1. Scan

Open `https://www.linkedin.com/messaging/?filter=unread`. For each unread
thread, read only the newest inbound message.

Also check `https://www.linkedin.com/notifications/` for application-status
updates ("your application was viewed", "your application was sent to").

## 2. Record

```bash
python -m jobagent reply add \
  --contact-url https://www.linkedin.com/in/priya-sharma \
  --thread-url "https://www.linkedin.com/messaging/thread/..." \
  --snippet "Thanks for reaching out — can you share your resume and notice period?" \
  --sentiment positive
```

Snippets are deduped on (contact, text), so re-scanning the inbox is free and
safe. Cap the snippet at ~300 characters.

Sentiment:

| Value | Means |
|---|---|
| `positive` | Interest, a question, a request for something, a call offer |
| `neutral` | Acknowledgement, "will keep you in mind" |
| `negative` | No, position filled, not a fit |
| `auto` | Out-of-office, "we received your application" — **not a real reply** |

Mark `auto` accurately. Counting autoresponders as replies makes the whole
report a lie.

If the sender isn't in the store, `contact upsert` them first — inbound
recruiters are the best leads the agent gets, and they should be tracked.

## 3. Notify

```bash
python -m jobagent reply unnotified
```

Tell the user, newest and most positive first: who, which company, what they
want, and what you suggest. Keep it to a line each. Then:

```bash
python -m jobagent reply mark-notified --all
```

If the session is unattended and a push notification tool is available, send
one for `positive` replies only. Nobody wants a phone buzz for an
autoresponder.

## 4. Draft responses — don't send them

For each positive reply, queue a response:

```bash
python -m jobagent outreach draft --contact-url <url> --kind thanks \
  --template thanks_reply --extra '{"next_step":"I have attached my resume — I am on a 60-day notice and could start sooner if needed. Free Tuesday or Thursday afternoon for a call."}'
```

Pull the concrete facts (notice period, CTC, availability) from the answer bank
rather than inventing them:

```bash
python -m jobagent answers get --key notice_period
python -m jobagent answers get --key expected_ctc
```

**Compensation is the exception to everything.** Never state a number to a
recruiter without the user explicitly confirming it in this conversation, even
though it's in the store. Draft it with the number, flag it in the approval
list, and let them decide.

A negative reply gets a short, warm acknowledgement or nothing at all. Never
argue, never re-pitch.

## 5. Close the loop

```bash
python -m jobagent reply handled --id 7
```

Once a reply has a response queued or the user has decided to ignore it.

Then report: how many replies, how many are real, what's queued for approval,
and anything that needs the user personally — an interview slot to pick, a
salary number to confirm, a question only they can answer.

## Stop conditions

This skill drives the browser too — the same five abort signals in
browser-playbook §0 apply (captcha, invite limit, restriction banner,
logged-out state). Stop immediately, log it (§0/§9), and tell the user.
