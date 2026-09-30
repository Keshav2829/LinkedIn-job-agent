"""orchestrator — a small local agent harness for the LinkedIn Job Agent.

Standalone on purpose: this package knows nothing about `jobagent` and
imports nothing from it. It only ever *reads* `plugin/skills/*/SKILL.md`
(the same files Claude Code, Cursor and the `linkedin-agent` subagent read)
and shells out to `python -m jobagent ...` the same way a human would. See
`orchestrator/README.md` for the design.
"""

__version__ = "0.1.0"
