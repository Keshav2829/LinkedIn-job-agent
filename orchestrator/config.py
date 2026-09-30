"""Configuration for the orchestrator.

Precedence, low to high: built-in defaults -> `orchestrator/config.json`
(gitignored; your local override) -> environment variables -> explicit
keyword overrides (CLI flags). Nothing here talks to the network — this
module only decides *what* to talk to.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ORCHESTRATOR_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = ORCHESTRATOR_DIR.parent
CONFIG_JSON_PATH = ORCHESTRATOR_DIR / "config.json"

# No `base_url` here on purpose -- its default depends on `provider` (an
# OpenAI-compatible local server vs. Anthropic's own API have different
# defaults), and that resolution has to happen *after* provider is known
# from the file/env/override layers below, not baked into the starting
# values a provider override could be layered on top of.
PROVIDER_DEFAULT_BASE_URLS = {
    "openai": "http://localhost:8000/v1",
    "anthropic": "https://api.anthropic.com",
}

DEFAULTS: dict[str, Any] = {
    "provider": "openai",
    "api_key": "EMPTY",
    "model": "local-model",
    "temperature": 0.2,
    "max_tokens": 1024,
    "max_steps": 25,
    "max_consecutive_errors": 3,
    "request_timeout": 120,
    "shell_timeout": 300,
    "vision": False,
    "interactive": False,
    "browse": False,
    "send_images": True,
}

_ENV_KEYS = {
    "provider": "ORCHESTRATOR_PROVIDER",
    "base_url": "ORCHESTRATOR_BASE_URL",
    "api_key": "ORCHESTRATOR_API_KEY",
    "model": "ORCHESTRATOR_MODEL",
    "temperature": "ORCHESTRATOR_TEMPERATURE",
    "max_tokens": "ORCHESTRATOR_MAX_TOKENS",
    "max_steps": "ORCHESTRATOR_MAX_STEPS",
    "max_consecutive_errors": "ORCHESTRATOR_MAX_CONSECUTIVE_ERRORS",
    "request_timeout": "ORCHESTRATOR_REQUEST_TIMEOUT",
    "shell_timeout": "ORCHESTRATOR_SHELL_TIMEOUT",
    "vision": "ORCHESTRATOR_VISION",
    "interactive": "ORCHESTRATOR_INTERACTIVE",
    "browse": "ORCHESTRATOR_BROWSE",
    "send_images": "ORCHESTRATOR_SEND_IMAGES",
    "effort": "ORCHESTRATOR_EFFORT",
}
_NUMERIC_KEYS = {
    "temperature": float,
    "max_tokens": int,
    "max_steps": int,
    "max_consecutive_errors": int,
    "request_timeout": int,
    "shell_timeout": int,
}
_BOOL_KEYS = {"vision", "interactive", "browse", "send_images"}
_TRUE_STRINGS = {"1", "true", "yes", "on"}


@dataclass
class Config:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    max_steps: int
    max_consecutive_errors: int
    request_timeout: int
    shell_timeout: int
    repo_root: Path
    provider: str = "openai"
    vision: bool = False
    interactive: bool = False
    browse: bool = False
    send_images: bool = True
    effort: str | None = None
    db_path: Path | None = None
    finetune_dir: Path | None = None

    def masked(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        if d.get("api_key"):
            d["api_key"] = d["api_key"][:2] + "…" + d["api_key"][-2:] if len(d["api_key"]) > 4 else "…"
        d["repo_root"] = str(self.repo_root)
        d["db_path"] = str(self.db_path) if self.db_path else None
        d["finetune_dir"] = str(self.finetune_dir) if self.finetune_dir else None
        return d


def _load_json_file() -> dict[str, Any]:
    if not CONFIG_JSON_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{CONFIG_JSON_PATH} is not valid JSON: {e}") from e


def load_config(overrides: dict[str, Any] | None = None) -> Config:
    values = dict(DEFAULTS)
    values["repo_root"] = str(DEFAULT_REPO_ROOT)

    file_values = _load_json_file()
    values.update({k: v for k, v in file_values.items() if v is not None})

    for key, env_name in _ENV_KEYS.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        if key in _NUMERIC_KEYS:
            values[key] = _NUMERIC_KEYS[key](raw)
        elif key in _BOOL_KEYS:
            values[key] = raw.strip().lower() in _TRUE_STRINGS
        else:
            values[key] = raw
    repo_root_env = os.environ.get("ORCHESTRATOR_REPO_ROOT")
    if repo_root_env:
        values["repo_root"] = repo_root_env
    db_env = os.environ.get("ORCHESTRATOR_DB")
    if db_env:
        values["db_path"] = db_env

    if overrides:
        for key, val in overrides.items():
            if val is not None:
                values[key] = val

    provider = str(values.get("provider", "openai"))
    if not values.get("base_url"):
        values["base_url"] = PROVIDER_DEFAULT_BASE_URLS.get(provider, PROVIDER_DEFAULT_BASE_URLS["openai"])

    repo_root = Path(values.pop("repo_root")).expanduser().resolve()
    db_path = values.pop("db_path", None)
    finetune_dir = values.pop("finetune_dir", None)

    cfg = Config(
        base_url=str(values["base_url"]),
        api_key=str(values["api_key"]),
        model=str(values["model"]),
        temperature=float(values["temperature"]),
        max_tokens=int(values["max_tokens"]),
        max_steps=int(values["max_steps"]),
        max_consecutive_errors=int(values["max_consecutive_errors"]),
        request_timeout=int(values["request_timeout"]),
        shell_timeout=int(values["shell_timeout"]),
        repo_root=repo_root,
        provider=provider,
        vision=bool(values.get("vision", False)),
        interactive=bool(values.get("interactive", False)),
        browse=bool(values.get("browse", False)),
        send_images=bool(values.get("send_images", True)),
        effort=values.get("effort") or None,
        db_path=Path(db_path).expanduser().resolve() if db_path else None,
        finetune_dir=Path(finetune_dir).expanduser().resolve() if finetune_dir
        else (ORCHESTRATOR_DIR / "data" / "finetune"),
    )
    return cfg


def with_overrides(cfg: Config, **kwargs: Any) -> Config:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return replace(cfg, **kwargs)
