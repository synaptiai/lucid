"""Runtime configuration loaded from environment.

API keys are held as SecretStr so their contents never appear in repr output
or in log records that pickle config objects. ANTHROPIC_LOG is explicitly
unset on import to prevent the Anthropic SDK from emitting request bodies
(which would leak user corpus content) when `ANTHROPIC_LOG=debug` is in the
environment.

We deliberately avoid pydantic-settings (not in the locked dep set) and read
dotenv files by hand. The format is the trivial subset: KEY=VALUE lines,
blank lines, and `#` comments.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from rich.traceback import install as _install_rich_traceback

# Neutralize any externally-set ANTHROPIC_LOG before the SDK imports it.
# The SDK reads this env var at import time to decide whether to log full
# request/response bodies; in Lucid those would include user corpus excerpts.
os.environ.pop("ANTHROPIC_LOG", None)

# Human-readable tracebacks in CLI, with locals suppressed to keep key material
# out of crash logs.
_install_rich_traceback(show_locals=False)


_DOTENV_CANDIDATES = (Path(".env.local"), Path(".env"))


def _load_dotenv_files() -> None:
    """Populate os.environ from .env / .env.local without overriding existing values.

    Precedence (highest wins): real env > .env.local > .env. Missing files are
    silently skipped. Malformed lines raise ValueError with the file and line
    number so the user sees the exact source.
    """
    for path in _DOTENV_CANDIDATES:
        if not path.is_file():
            continue
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{lineno}: expected KEY=VALUE, got {raw!r}")
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


class Settings(BaseModel):
    """Top-level settings. Instantiate via `load_settings()`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anthropic_api_key: SecretStr = Field(
        ..., description="Claude API key used for all orchestrator and module calls."
    )
    voyage_api_key: SecretStr = Field(
        ..., description="Voyage AI key for Module H embeddings (voyage-3-large)."
    )
    data_dir: Path = Field(
        default=Path(".lucid"),
        description="Directory for SQLite corpus/findings and ephemeral artifacts.",
    )
    report_dir: Path = Field(
        default=Path("report"),
        description="Directory for generated HTML reports.",
    )
    log_level: str = Field(default="INFO")


def load_settings() -> Settings:
    """Load settings from environment (with .env / .env.local layered in first).

    Raises pydantic.ValidationError if ANTHROPIC_API_KEY or VOYAGE_API_KEY is
    missing.
    """
    _load_dotenv_files()
    return Settings(
        anthropic_api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
        voyage_api_key=SecretStr(os.environ["VOYAGE_API_KEY"]),
        data_dir=Path(os.environ.get("LUCID_DATA_DIR", ".lucid")),
        report_dir=Path(os.environ.get("LUCID_REPORT_DIR", "report")),
        log_level=os.environ.get("LUCID_LOG_LEVEL", "INFO"),
    )
