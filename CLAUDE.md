# LinkedIn Job Agent

**All instructions live in [`AGENTS.md`](AGENTS.md) — read it.** This file is a
pointer, deliberately. Cursor reads both files, so duplicating content here
would just spend context twice and give two things to keep in step.

Claude Code specifics only:

- Skills auto-discover from `.claude/skills/` — say `run linkedin-daily-run`.
- The subagent in `.claude/agents/linkedin-agent.md` runs the whole cycle in its
  own context window and returns a short summary: "use the linkedin-agent
  subagent to run today's job search". Worth it — a full run scrapes a lot of
  page text you don't want in your main conversation.
- Both are **copies of `plugin/`**. Edit `plugin/`, then re-copy (see
  AGENTS.md, "Keeping copies in sync").
