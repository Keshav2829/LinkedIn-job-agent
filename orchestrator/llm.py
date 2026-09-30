"""Minimal clients for the model that drives one orchestrator session.

Three providers, one `.chat(messages: list[dict]) -> str` interface:

- `LLMClient` — any OpenAI-compatible `/chat/completions` endpoint (vLLM,
  llama.cpp's server, LM Studio, Ollama's `/v1` shim, ...). This is the
  default and the whole reason this project has no dependency manifest: it
  only needs one POST that accepts `{model, messages, temperature,
  max_tokens}` and returns `choices[0].message.content`.
- `AnthropicClient` — Claude's native Messages API, for an Anthropic API key.
- `ClaudeCLIClient` — shells out to the `claude` CLI instead, for a Claude
  Code subscription with no separate API key. Same pattern
  `annotation_suite/annotate.py`'s `call_claude` already uses in this repo
  (`claude -p ... --output-format json --restricted`) — this is just that
  pattern reused to drive a full ```action loop turn-by-turn instead of a
  single backfill prompt.

All three exist so a *live* orchestrator session (the real loop in
agent.py, against real `jobagent` output) can be driven by Claude instead
of a local model — the point being to fine-tune a small local model on
Claude's own trajectories through this exact protocol, rather than on a
weaker local model's own attempts at it. `build_client` picks between them
from `cfg.provider`; nothing else in this project (agent.py, the protocol,
the tool surface) changes based on which one is in use — Claude gets the
same text `` ```action `` instructions and produces the same wire format,
since native tool-calling is deliberately not used here (see agent.py's
docstring for why).

The two network clients are stdlib-only (`urllib`, not the
`anthropic`/`openai` packages) for the same reason the OpenAI one always
was; `ClaudeCLIClient` shells out instead of calling any API at all.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


class LLMError(Exception):
    pass


def _looks_local(url: str) -> bool:
    """Best-effort guess at whether `url`'s host is a machine-local or LAN
    server (localhost, a loopback/private IP) vs. a real remote host --
    purely to pick the right troubleshooting hint on a connection failure.
    "Is your local server running?" is useless advice when the host is
    `api.deepseek.com`; "check your firewall/proxy/VPN" is useless advice
    when it's `127.0.0.1`. Never used to gate behavior, only wording.
    """
    host = urlsplit(url).hostname or ""
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


@dataclass
class LLMClient:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 120

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        if (not self.api_key or self.api_key == "EMPTY") and not _looks_local(url):
            raise LLMError(
                f"No API key configured for remote endpoint {url} -- still using "
                "this project's local-server placeholder (\"EMPTY\"), which a real "
                "provider will reject. api_key is deliberately never pinned to a "
                "session (it's a secret), so it has to be supplied again on every "
                "call that talks to this endpoint, including a resume: pass "
                "--api-key, set $ORCHESTRATOR_API_KEY, or (resuming from the UI's "
                "answer box) fill in its API key field."
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            if _looks_local(url):
                hint = "Is the local model server running (e.g. `vllm serve ...`)?"
            else:
                hint = (
                    "This is a remote host, not a local server -- if it's genuinely "
                    "reachable (try it in a browser or `curl`), a bare connection "
                    "timeout/failure like this on an otherwise-working endpoint usually "
                    "means something on this network is blocking or intercepting it: a "
                    "firewall, antivirus HTTPS inspection, VPN, or proxy -- not a code "
                    "or API problem."
                )
            raise LLMError(f"Could not reach {url}: {e.reason}. {hint}") from e
        except TimeoutError as e:
            raise LLMError(f"Request to {url} timed out after {self.timeout}s") from e

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"Non-JSON response from {url}: {raw[:2000]}") from e

        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape from {url}: {json.dumps(body)[:2000]}") from e


# --- Anthropic (Claude) ------------------------------------------------

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"


def _data_uri_to_anthropic_image(url: str) -> dict[str, Any] | None:
    """`data:image/png;base64,AAAA...` (this project's own image_url shape,
    see db.to_api_messages) -> Anthropic's `image` content block. None for
    anything that isn't a base64 data URI (Anthropic's Messages API doesn't
    take a bare remote image URL the way some OpenAI-compatible servers do)."""
    if not url.startswith("data:") or "," not in url:
        return None
    header, _, data = url.partition(",")
    media_type = header[len("data:"):].split(";")[0] or "application/octet-stream"
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _content_to_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks: list[dict[str, Any]] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            blocks.append({"type": "text", "text": b.get("text", "")})
        elif b.get("type") == "image_url":
            img = _data_uri_to_anthropic_image((b.get("image_url") or {}).get("url", ""))
            if img:
                blocks.append(img)
    return blocks


def to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """This project's OpenAI-shaped messages (`db.to_api_messages`'s output:
    system/user/assistant roles, tool observations already folded into
    `user`) -> Anthropic's Messages API shape.

    Two adjustments Anthropic's API requires that OpenAI-compatible servers
    don't:

    - `system` is a separate top-level field, not a role inside `messages`
      — every `role: "system"` turn (there's normally exactly one, the
      protocol/tool-docs prompt from agent.build_system_prompt) is pulled
      out and joined.
    - Anthropic rejects two consecutive same-role turns. This project's own
      protocol produces exactly that in one place: when a session pauses on
      `{"ask": ...}`, the question is stored as a `tool` row (mapped to
      `role: "user"` by to_api_messages) and the human's answer, appended
      later by `answer_question`, is *also* `role: "user"` with no
      assistant turn in between. Consecutive same-role turns are merged
      into one, content blocks concatenated, instead of sent as two.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"] if isinstance(m["content"], str) else str(m["content"]))
            continue
        blocks = _content_to_blocks(m["content"])
        if converted and converted[-1]["role"] == m["role"]:
            converted[-1]["content"].extend(blocks)
        else:
            converted.append({"role": m["role"], "content": blocks})
    return "\n\n".join(system_parts), converted


@dataclass
class AnthropicClient:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 120

    def chat(self, messages: list[dict[str, Any]]) -> str:
        system_text, converted = to_anthropic_messages(messages)
        url = self.base_url.rstrip("/") + "/v1/messages"
        if not self.api_key and not _looks_local(url):
            raise LLMError(
                f"No API key configured for {url}. api_key is deliberately never "
                "pinned to a session (it's a secret), so it has to be supplied "
                "again on every call that talks to this endpoint, including a "
                "resume: pass --api-key, set $ANTHROPIC_API_KEY, or (resuming "
                "from the UI's answer box) fill in its API key field."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_text:
            payload["system"] = system_text
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            if _looks_local(url):
                hint = " Is a local proxy for this endpoint running?"
            else:
                hint = (
                    " If it's genuinely reachable (try it in a browser or `curl`), a bare "
                    "connection timeout/failure like this on an otherwise-working endpoint "
                    "usually means something on this network is blocking or intercepting "
                    "it: a firewall, antivirus HTTPS inspection, VPN, or proxy."
                )
            raise LLMError(f"Could not reach {url}: {e.reason}.{hint}") from e
        except TimeoutError as e:
            raise LLMError(f"Request to {url} timed out after {self.timeout}s") from e

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"Non-JSON response from {url}: {raw[:2000]}") from e

        try:
            blocks = body["content"]
        except (KeyError, TypeError) as e:
            raise LLMError(f"Unexpected response shape from {url}: {json.dumps(body)[:2000]}") from e
        return "".join(
            b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
        )


# --- Claude Code CLI (no API key needed) --------------------------------

CLAUDE_CLI_PROMPT_TEMPLATE = """{system}

You are mid-conversation as the ASSISTANT in the exchange transcribed \
below. Read it, then produce ONLY your next ASSISTANT reply -- continuing \
naturally from where it left off, following the response protocol given \
above exactly. Do not repeat earlier turns, do not prefix your reply with \
a role label, and do not add any commentary about this being a simulated \
exchange -- output nothing but the reply itself.

--- conversation so far ---
{conversation}
--- end of conversation, your reply starts now ---
"""


def _flatten_content_for_cli(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "image_url":
            # `claude -p --restricted` has no file/tool access, so it can't
            # be handed an image the way a real vision API call can. Rather
            # than silently drop the turn or crash, say so in-line -- loud
            # and debuggable beats a model confidently reasoning about a
            # screenshot it never saw.
            parts.append(
                "[screenshot omitted: the claude-cli provider runs with "
                "--restricted and cannot receive image attachments; use "
                "--provider anthropic for a vision-enabled session]"
            )
    return "\n".join(parts)


def render_transcript_for_cli(messages: list[dict[str, Any]]) -> str:
    """This project's OpenAI-shaped messages -> one flattened prompt for a
    single-shot `claude -p` call.

    No multi-turn CLI session to manage here: agent.py's loop is already
    fully stateless per call (`drive_session` rebuilds and resends the
    *entire* transcript from the database every turn, see
    db.to_api_messages) -- so flattening the whole thing into one prompt
    reproduces exactly the same input a real chat-completions call would
    get, just rendered as text instead of a JSON message array.
    """
    system_parts: list[str] = []
    turns: list[str] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"] if isinstance(m["content"], str) else str(m["content"]))
            continue
        label = "USER" if m["role"] == "user" else "ASSISTANT"
        turns.append(f"[{label}]\n{_flatten_content_for_cli(m['content'])}")
    return CLAUDE_CLI_PROMPT_TEMPLATE.format(
        system="\n\n".join(system_parts), conversation="\n\n".join(turns),
    )


@dataclass
class ClaudeCLIClient:
    model: str = "sonnet"
    timeout: int = 120
    claude_bin: str = "claude"
    # A bare `subprocess.run(["claude", ...])` spawn can also fail
    # transiently even when `claude` genuinely is on PATH and works from an
    # interactive prompt -- an antivirus real-time scan briefly locking a
    # freshly touched exe, Claude Code self-updating mid-run, or contention
    # with another concurrently running `claude` process have all been
    # observed to surface as a bare FileNotFoundError from one specific
    # call in the middle of an otherwise-working session. Retrying a
    # couple of times before giving up papers over exactly that class of
    # blip without masking a genuinely-not-installed binary (see the
    # `shutil.which` check in the final error below, which -- together
    # with the winerror check -- tells the different cases apart).
    max_retries: int = 2
    retry_delay: float = 1.0
    # None means "don't pass --effort at all" -- the CLI then falls back to
    # whatever ~/.claude/settings.json's modelSettings.<model>.effortLevel
    # (or the CLI's own built-in default) says. That's an ambient,
    # machine-dependent default, which is a real gap for a data-generation
    # pipeline: the "same" session could get a different effort level on a
    # different machine, or after a config change, with nothing recording
    # which. Pass a real value here to make it explicit and reproducible.
    effort: str | None = None

    def chat(self, messages: list[dict[str, Any]]) -> str:
        prompt = render_transcript_for_cli(messages)
        # The prompt is agent.py's *entire* running transcript, resent in
        # full every turn (see render_transcript_for_cli) -- it has no
        # upper bound and only grows across a session. Passing it as a
        # `-p <prompt>` argv element used to hit Windows' ~32K-character
        # command-line limit partway through a real multi-turn session:
        # CreateProcess then fails with WinError 206 ("the filename or
        # extension is too long" -- a misleading name for "your command
        # line is too long"), surfaced by Python as a plain
        # FileNotFoundError with no hint of the real cause. `claude -p`
        # with no positional prompt argument reads it from stdin instead
        # (verified against the real CLI), which has no such limit.
        cmd = [
            self.claude_bin, "-p", "--output-format", "json", "--model", self.model,
            "--restricted", "--strict-mcp-config", "--permission-prompts", "none",
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        proc = None
        last_error: FileNotFoundError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                proc = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout,
                    encoding="utf-8", errors="replace",
                )
                break
            except FileNotFoundError as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
            except subprocess.TimeoutExpired as e:
                raise LLMError(f"claude CLI timed out after {self.timeout}s") from e

        if proc is None:
            if getattr(last_error, "winerror", None) == 206:
                # Should no longer happen now the prompt goes over stdin --
                # if it still does, something *else* in `cmd` (an
                # unexpectedly long --model value, etc.) is the culprit,
                # not the prompt, so say that plainly instead of repeating
                # the now-inapplicable "transient lock" guess below.
                raise LLMError(
                    f"'{self.claude_bin}' could not be launched: the command line was too long "
                    f"(WinError 206), even with the prompt on stdin. Something else in the "
                    f"argv is unexpectedly large (model={self.model!r}): {last_error}"
                ) from last_error
            resolved = shutil.which(self.claude_bin)
            if resolved:
                detail = (
                    f"shutil.which('{self.claude_bin}') resolves to {resolved!r} right now -- it "
                    "exists, but the spawn still failed every time. Likely a transient lock "
                    "(antivirus scanning the exe, Claude Code self-updating, or another `claude` "
                    "process running at the same time), not a missing install."
                )
            else:
                detail = f"shutil.which('{self.claude_bin}') finds nothing on PATH right now."
            raise LLMError(
                f"'{self.claude_bin}' could not be launched after "
                f"{self.max_retries + 1} attempt(s): {last_error}. {detail}"
            ) from last_error

        if proc.returncode != 0:
            # `claude -p --output-format json` still writes a JSON result
            # object to stdout even when the call itself failed (a rate
            # limit, an upstream API error, ...) -- it's not just for the
            # success path. Try to pull the actual human-readable reason
            # (and HTTP status, if there is one) out of it instead of
            # dumping the raw JSON blob, which is all a plain "exited 1:
            # <stdout>" used to show here -- e.g. a 429 came through as an
            # unreadable wall of JSON instead of "You've hit your session
            # limit - resets 12:20am (Asia/Kolkata)".
            data = None
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                pass
            if isinstance(data, dict) and data.get("result"):
                status = data.get("api_error_status")
                status_note = f" (HTTP {status})" if status else ""
                raise LLMError(f"claude CLI reported an error{status_note}: {data['result']}")
            detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:2000]
            raise LLMError(f"claude CLI exited {proc.returncode}: {detail}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude CLI did not return valid JSON: {proc.stdout[:2000]}") from e
        if data.get("is_error"):
            status = data.get("api_error_status")
            status_note = f" (HTTP {status})" if status else ""
            raise LLMError(f"claude CLI reported an error{status_note}: {str(data.get('result'))[:2000]}")
        text = (data.get("result") or "").strip()
        if not text:
            raise LLMError("claude CLI returned an empty result")
        return text


def build_client(cfg) -> LLMClient | AnthropicClient | ClaudeCLIClient:
    """Pick the client for `cfg.provider` ("openai", the default;
    "anthropic"; or "claude-cli"). All three expose the same
    `.chat(messages) -> str`, so agent.py never needs to know which one
    it's talking to.

    `cfg.base_url` is always already resolved to the right provider's
    default by `config.load_config` (see `PROVIDER_DEFAULT_BASE_URLS`
    there) when nothing set it explicitly, so this only has to decide the
    API key fallback: `AnthropicClient` needs `x-api-key`, and this
    project's own default api_key ("EMPTY", meant for a local server that
    doesn't check one) obviously isn't a real Anthropic key, so fall back
    to $ANTHROPIC_API_KEY when nothing else was configured. `claude-cli`
    needs no key at all -- it authenticates however `claude` itself is
    already logged in.
    """
    if cfg.provider == "claude-cli":
        return ClaudeCLIClient(model=cfg.model, timeout=cfg.request_timeout, effort=cfg.effort)
    if cfg.provider == "anthropic":
        api_key = cfg.api_key if cfg.api_key and cfg.api_key != "EMPTY" else os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicClient(
            base_url=cfg.base_url, api_key=api_key, model=cfg.model,
            temperature=cfg.temperature, max_tokens=cfg.max_tokens, timeout=cfg.request_timeout,
        )
    return LLMClient(
        base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens, timeout=cfg.request_timeout,
    )
