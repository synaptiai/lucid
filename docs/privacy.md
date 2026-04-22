# Privacy & data handling

Lucid audits **personal AI conversation history**. That is the most
sensitive class of personal data most users hold: unfiltered thought,
drafts never shared, half-formed opinions, medical and financial
detail, work material covered by NDA. Every choice in Lucid's design
starts from that.

This document is the complete privacy story: what leaves your machine
during an audit, what doesn't, how keys are handled, what survives on
disk, and how to audit or delete everything Lucid touches.

If something here feels wrong, that is a bug — open an issue.

## What leaves your machine

### Anthropic API (required)

During an audit, these HTTP request bodies are sent to Anthropic's API
(`api.anthropic.com`):

1. **`messages.count_tokens` pre-pass** (free, rate-limit-exempt).
   Input: the text of sampled conversations. No output; Anthropic
   returns only a token count. Used to compute the cost estimate you
   see before authorizing spend.

2. **`messages.create` module calls.** Input: the system prompt
   (Lucid's rubric — under `prompts/`, open-source) plus a bounded
   chunk of your conversation content. Output: the classifier's
   reasoning + JSON verdict. Module A uses 10-turn windows; Modules
   B–H use smaller per-call chunks.

3. **Managed Agents orchestrator** (when enabled). The orchestrator's
   system prompt; no user corpus is uploaded to the agent environment.
   Custom-tool calls (`query_corpus`, `get_turn_window`, etc.) return
   corpus excerpts to the orchestrator as tool results, which does
   place excerpts in the message context. Lucid never mounts the
   corpus as files to the agent environment, so there is no out-of-
   band upload.

Anthropic's privacy policy applies to everything in (1)–(3). See the
Anthropic Consumer Terms and Privacy Policy for their current
retention and training-use policies. API content is not used to train
Anthropic's models by default.

### Voyage API (Module H only)

If you run Module H (memory-corpus consistency), corpus text chunks
plus memory claims are sent to Voyage AI (`api.voyageai.com`) for
embedding. Each chunk is a `(user_turn + adjacent_assistant_turn)`
pair; each memory claim is a short atomic assertion (≤25 words).
Embeddings are returned as 1024-dim float32 vectors.

Voyage's privacy policy applies to these requests. If you don't want
chunks going to Voyage, don't set `VOYAGE_API_KEY` — Module H will
skip with `no_embedding_provider` and the rest of the audit continues
normally.

### What doesn't leave

- **No corpus upload to cloud storage.** Lucid operates against a
  local SQLite database (`~/.lucid/audit-<run-id>.sqlite3` or a path
  you specify). The DB never leaves your machine.
- **No tracking, no analytics, no crash reporter.** Lucid is a
  standalone CLI. No telemetry of any kind is sent.
- **No third-party services beyond Anthropic + Voyage.** The plan
  documents OpenAI embeddings as a fallback, but the fallback is not
  wired in v0.1.0; no request goes to OpenAI unless you explicitly
  add and configure it.
- **The HTML report is local.** `report/<run-id>.html` is written to
  your filesystem. It contains no external scripts, no tracking
  pixels, no CDN links. Open it directly in your browser; nothing
  phones home.

## Keys & secrets

### Storage

Anthropic and Voyage keys are loaded from environment variables
(`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`) or from `.env.local` /
`.env` files in the project root at startup. They are wrapped in
`pydantic.SecretStr` the moment they're read — this is the same
pattern used across FastAPI / Pydantic projects and it means:

- The key never appears in `repr(settings)` output.
- The key never appears in a logged exception traceback as a field
  value.
- The key is accessible only via explicit `.get_secret_value()` calls
  at the boundary where it's passed to an SDK client.

`lucid.cli` does `os.environ.pop("ANTHROPIC_LOG", None)` on startup
so the Anthropic SDK's HTTP debug logger (which would leak request
bodies) cannot be re-enabled during a Lucid run. If you're debugging
the SDK yourself, re-enable it explicitly after Lucid has started.

### Canary tests

`lucid/logging.py` ships a `SafeFormatter` that drops any log record
tagged `contains_user_content=True`. The test suite plants a literal
sentinel string (`LUCID_CANARY_MEMORY`) in fixture memory content,
runs the relevant code paths at `DEBUG` level with pytest's `caplog`,
and asserts the sentinel never appears in any log record. Any new
code path that handles sensitive content must add its own canary
test (see `tests/test_ingest_contract.py` for the pattern).

## What's stored on disk

Every audit creates:

1. **SQLite database** at `~/.lucid/audit-<run-id>.sqlite3` (or a
   custom path via `LUCID_DB_PATH`). Contains: conversations, turns,
   findings, memory claims, and (Module H only) embedding vectors.
   The embeddings are keyed on `sha256(chunk_text)` — you can delete
   them individually if you want to re-embed.

2. **HTML report** at `report/<run-id>.html`. Static file, no
   scripts, no external resources. Open directly in a browser.

3. **Logs** (if you passed `--log-level DEBUG` or set
   `LUCID_LOG_FILE`). Default is stderr only; no log file is
   written unless you ask for one.

Everything Lucid creates lives under your home directory or the
project working directory. Nothing is written to `/tmp` or system
caches.

### Deleting a Lucid audit

```bash
rm -f ~/.lucid/audit-<run-id>.sqlite3
rm -f report/<run-id>.html
```

To clean up every Lucid audit at once:

```bash
rm -rf ~/.lucid/
rm -rf report/
```

This is a complete wipe of Lucid-side state. The Anthropic/Voyage
side retains whatever their terms of service allow; see those
providers' privacy controls.

## Threat model

Lucid is designed against three threat classes.

### 1. Accidental leakage through logs

The primary mitigation is the SafeFormatter + canary sentinel pattern
above. Secondary: the `ANTHROPIC_LOG` unset at startup. If you set
`--log-level DEBUG`, finding metadata is logged (module, behavior,
intensity) but never quote text; the test suite enforces this.

### 2. Key exfiltration via dependency compromise

Dependencies are pinned to exact versions in `pyproject.toml`. A
supply-chain compromise of any pinned dep would require us to update
the pin; this makes silent drift impossible. The CI workflow (not
yet wired in v0.1.0) should verify the lockfile hasn't changed
unexpectedly between commits.

### 3. Prompt-injection from corpus content

User conversation content is untrusted input. Every module's prompt
template instructs the model to treat transcript blocks as data, not
instructions. Every delimiter token (`<TRANSCRIPT_BLOCK>`,
`<ORIGINAL_ANSWER>`, `<CONVERSATION>`, etc.) is escaped by the ingest
layer if it appears literally inside user content, so no corpus turn
can smuggle a closing delimiter + attacker-controlled instructions.
The test suite includes injection fixtures against Modules A, B, C,
D, E, F, and H; they pass because Jinja autoescape handles the
report-rendering surface and the module prompts handle the LLM
surface.

See the "Input hygiene" section at the top of every prompt file
(`prompts/module_*/`) for the prompt-side defense.

## What Lucid's findings contain

Every finding persisted to the SQLite DB (and rendered to the HTML
report) contains:

- A behavior label (e.g. `sycophancy`, `feedback-sycophancy`,
  `contradicted`).
- Short verbatim excerpts from the turns that motivated the finding
  (≤140 characters per excerpt by prompt design).
- A confidence score and (for calibrated modules) a Beta-posterior
  credibility interval.
- Prompt version + hash for audit reproducibility.
- Timestamps and turn ids.

No finding ever contains:

- Raw API keys or tokens.
- Full user-turn content outside the ≤140-char excerpts.
- Other users' conversation content (Lucid only audits the corpus
  you point it at).
- Summaries of your conversations beyond the per-finding excerpts.

If you're sharing a report for peer review, the report is a
self-contained HTML file. You can open it, inspect the ≤140-char
excerpts, and decide whether to share. Lucid gives you the artifact;
it doesn't make sharing decisions on your behalf.

## Redaction before sharing

Lucid v0.1.0 does **not** ship a `--redact` flag. Synthetic/seeded
corpora (see `demo/corpus/`) are the recommended path for demos and
public sharing.

Post-hackathon, a redaction pass over the rendered report is a
likely add — patches welcome.

## Questions, concerns, bugs

If you find any of the following, open a GitHub issue:

- A log line that contains user corpus text.
- A finding field that includes more content than the ≤140-char
  excerpt budget.
- A dependency update that changes the privacy surface.
- Any other behavior that contradicts this document.

Privacy issues are a release-blocker. Fixes ship at patch level
(0.1.x).
