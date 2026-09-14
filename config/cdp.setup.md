# Setting up the CDP driver

The CDP driver is the agent driving **your own Chrome directly** through
Python, instead of asking an MCP browser extension to describe pages to it.
Same Chrome, same LinkedIn login, no extension in the loop.

It exists because of one hard wall: `snapshot` / `read_page` return the entire
accessibility tree, which on a LinkedIn job page, profile, search result or
the feed is 100,000+ characters — past the per-tool-result cap every host
enforces, so the call **fails outright** rather than returning something
short. See `plugin/references/browser-adapter.md`, "`snapshot` output size".

On the CDP driver the page is searched inside the Python process and only the
answer is printed. A `find` costs about 600 characters instead of 100,000.

## 1. Install Playwright

```bash
pip install playwright
```

That is the whole install. **Do not run `playwright install`** — we attach to
the Chrome you already have, so there is no browser to download.

## 2. Start Chrome with debugging on

Pick a profile directory anywhere you like; the examples below use a folder
under your home directory so the command needs no editing per machine.

**Windows (PowerShell):**
```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
    --remote-debugging-port=9222 `
    --user-data-dir="$env:USERPROFILE\chrome-agent-profile"
```

**macOS:**
```bash
open -a "Google Chrome" --args \
    --remote-debugging-port=9222 \
    --user-data-dir="$HOME/chrome-agent-profile"
```
(`open -a` on a running Chrome ignores new flags — quit Chrome fully first,
or launch the binary directly: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`
with the same two flags.)

**Linux:**
```bash
google-chrome \
    --remote-debugging-port=9222 \
    --user-data-dir="$HOME/chrome-agent-profile"
```
(binary may be `google-chrome-stable` or `chromium` depending on distro/package)

The separate `--user-data-dir` is **not optional**. Chrome 136 and later
refuse `--remote-debugging-port` on your default profile — a deliberate
security change, not a bug to work around.

Practically that means this is a second Chrome profile: a clean window with
none of your normal extensions or history. **Log into LinkedIn once in it.**
The profile folder persists, so you only do this once; every later launch of
that same command comes up already signed in.

Keep the command somewhere you can re-run it: a `.lnk` shortcut (Windows,
flags in the Target field), a saved `.command` file or shell alias (macOS), or
a shell alias / `.desktop` launcher (Linux) are all the same idea — a
one-click or one-word way to start Chrome with the two flags already on it.

## 3. Verify

```bash
python -m jobagent browse status
```

Expected: `"connected": true` and your open tabs. If instead you get
`no Chrome listening on http://127.0.0.1:9222`, Chrome is either not running
with the flag or is running on your normal profile — check that the window
you started with the command above is still open.

## 4. Use it

Everything speaks the verbs in `plugin/references/browser-adapter.md`. The raw
commands, if you want to drive it by hand:

```bash
python -m jobagent browse open --url "https://www.linkedin.com/jobs/search/?keywords=AI+Engineer"
python -m jobagent browse outline                    # → where am I, refs i1, i2, … + job URLs
python -m jobagent browse open --url "https://www.linkedin.com/jobs/view/4021553311/"
python -m jobagent browse find -q "Easy Apply"       # → refs e1, e2, …
python -m jobagent browse click --ref e1
python -m jobagent browse form                       # → the whole Easy Apply panel
python -m jobagent browse fill --ref f1 --value "9876543210"
python -m jobagent browse upload --ref f4 --path data/resume.pdf
python -m jobagent browse shot --label ibm-attach_resume
```

Rough rule for which reader to reach for: **`outline`** when you don't know
what's on the page, **`form`** once a dialog is open, **`find`** when you know
the words on the thing you want, **`text`** when you want prose to read.

`browse status` and `browse find` are safe to run any time; they only read.

## What this driver gives you that Browser MCP does not

- **Resume attachment.** `browse upload` sets a real file on a real file
  input. This closes the one capability gap that forced resume-as-a-link.
- **Reads that cannot blow the token cap.** `outline`, `find`, `form` and
  `text` are all bounded by construction, not by luck about which page you
  are on. The full page still gets captured for the corpus — `browse dump`
  writes it to `data/traces/pages/` and hands back only the path.
- **Click coordinates that match their screenshots.** `browse click` reports
  the box *after* scrolling, alongside the scroll offset and viewport, and
  `browse shot` captures that same viewport — so a logged coordinate and its
  image finally describe the same pixel. The 4 Sep corpus audit found these
  in different frames.
- **The same behaviour in every client.** Cursor, Claude Code and Cowork all
  run the same Python, because the driver is not the client's browser.

## What it does not do

- No natural-language `find` — you pass text, and matching is exact-then-fuzzy
  on the element's accessible name. In practice LinkedIn's buttons say what
  they say ("Easy Apply", "Review", "Submit application"), so this is rarely
  the limitation it sounds like.
- No second tab management beyond `--tab <url-substring>`; it drives the tab
  you point it at.
- It cannot see a page Chrome is not showing. Same as any driver.

## Turning it off

Nothing to turn off. Skip the flag when you start Chrome and the agent falls
back to whichever MCP driver is present, exactly as before — the `browse`
group simply reports that no Chrome is listening.
