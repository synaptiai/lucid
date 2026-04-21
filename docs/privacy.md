# Privacy

Placeholder — filled during Phase 9/10.

## Threat model (to be expanded)

Lucid processes private conversation history. Treat it like a medical record.

- **Sensitive inputs:** Claude.ai export (`conversations.json`, `memories.json`), Claude Code session JSONL.
- **Where data lives:** locally, in a SQLite database the user controls. Never uploaded to Lucid's servers (there are none).
- **What leaves the machine:** sampled conversation content sent as LLM input to the Anthropic API (for module classification) and, for Module H only, to the Voyage AI API (for embeddings). Both are subject to those providers' retention policies.

## Engineering controls (to be expanded)

- API keys via `pydantic.SecretStr`; never logged verbatim.
- `ANTHROPIC_LOG` unset on startup to prevent SDK request/response body dumps.
- `SafeFormatter` drops log records tagged `contains_user_content`.
- Canary-sentinel test: memory text containing a known token is verified absent from DEBUG logs.
- Generated HTML reports use Jinja2 autoescape + a CSP meta tag; corpus quotes are escaped before rendering.
- Ingest path-traversal guards: `Path.resolve(strict=True)` + symlink rejection outside the declared root.

## What we do not do

- Redact corpus content automatically. The demo video uses a synthetic seeded corpus instead.
- Send corpus content anywhere other than Anthropic + Voyage (for their explicit API use).
- Persist any data under `/tmp` or a shared user cache.

---

*Seeded Phase 1. Finalized Phase 9 (report security) and Phase 10 (pre-release pass).*
