# Installing Browser MCP alongside Claude in Chrome

Browser MCP (https://github.com/browsermcp/mcp) is a third way to drive Chrome,
and the last resort in the preference order (CDP, then Claude in Chrome, then
Browser MCP — see `plugin/references/browser-adapter.md`). It matters because
it works in Cursor, VS Code, Windsurf and Claude Desktop — places CDP and
Claude in Chrome don't reach — so the agent becomes runnable outside Cowork.

All three can be installed at once. **They must not drive the same tab in
the same run** — the agent picks one driver at the start and keeps it.

## 1. Prerequisites

- Node.js — https://nodejs.org/en/download (`node --version` to check)
- Google Chrome

## 2. Install the extension

Install the **Browser MCP** extension from the Chrome Web Store, then pin it.
To use it you click the extension and **Connect** it to the tab you want
automated — unlike Claude in Chrome, it attaches to one tab you choose rather
than creating its own.

## 3. Add the server to your MCP client

Same JSON everywhere:

```json
{
  "mcpServers": {
    "browsermcp": {
      "command": "npx",
      "args": ["@browsermcp/mcp@latest"]
    }
  }
}
```

**Claude Desktop** — the config file's location is OS-specific:

| OS | Path |
|---|---|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

Create the file if it doesn't exist; merge the `browsermcp` key into any
existing `mcpServers` object rather than replacing it. Restart Claude Desktop.
A known issue starts the server twice and may show an error on launch; the
server still works.

**Cursor** — Settings → Tools → New MCP server → paste the above → click the
refresh icon next to `browsermcp`.

**VS Code / Windsurf** — follow their MCP server docs with the same JSON.

## 4. Verify

In a new session, ask for the tool list and look for the Browser MCP tools:
`snapshot`, `click`, `type`, `selectOption`, `hover`, `drag`, plus navigation
and `screenshot`. Namespacing depends on the client —
`mcp__browsermcp__snapshot` locally, or `mcp__remote-devices__browsermcp__…`
when a Cowork cloud session proxies them from the desktop app.

If the names differ from the adapter table, correct
`plugin/references/browser-adapter.md` — that table is the only place the tool
names are written down.

## 5. One thing to set before using it

Browser MCP has no file-upload tool, so the agent cannot attach your resume to a
LinkedIn message on this driver. It sends a link instead. Store one:

```bash
python -m jobagent profile set --json '{"resume_url":"https://<your-resume-link>"}'
```

Without it the agent will refuse to send resume-bearing messages on Browser MCP
rather than promise an attachment it can't add.

## What still works on Browser MCP

Job search, scoring, Easy Apply end to end (including screening questions),
connection requests with notes, and unattached messages. What you give up is
resume attachment, natural-language element finding, tab management, and
network inspection.
