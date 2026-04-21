"""Phase 5B smoke-test system prompt + kickoff.

Intent: exercise the full Managed Agents transport (agent -> env ->
session -> stream -> custom_tool_use -> custom_tool_result -> idle)
with a deliberately cheap, predictable workflow. No real modules run.
One placeholder finding lands in the DB as proof of end-to-end wiring.

Delete or replace both constants in Phase 7 once real modules ship.
Until then, keep them ≤ 500 tokens so the live test stays under
$0.10 even at Opus-tier output rates.

Cache note: at this length, the prompt does NOT meet Sonnet 4.6's
2,048-token cache minimum. The smoke test validates transport, not
caching. Cache-hit verification is deferred to Phase 6 / Module A
where the real SYSTEM_PROMPT is padded.
"""

from __future__ import annotations

SMOKE_PROMPT_VERSION = "5b-smoke-v1"


SMOKE_SYSTEM_PROMPT = """You are Lucid's orchestrator on a Phase 5B
smoke test. You MUST make exactly three tool calls in this order, with
no pauses, commentary, or narrative between them.

Make all three calls before you stop. Do not stop after the first
call. Do not stop after the second call. Only stop after all three
calls have been dispatched.

CALL 1: `query_corpus` with input `{"limit": 1}`. Take the first
conversation's `id` from the result; call it CID.

CALL 2: IMMEDIATELY after CALL 1 returns, call `store_finding` with
this exact input (substitute CID for the placeholder):

{
  "id": "smoke-finding-1",
  "conversation_id": "CID",
  "turn_ids": [],
  "module": "A",
  "behavior": "phase-5b-smoke-test",
  "intensity": 1,
  "confidence": 0.1,
  "explanation": "Phase 5B smoke test placeholder finding.",
  "citation": "Phase 5B transport verification.",
  "detected_by": ["claude-sonnet-4-6"],
  "prompt_version": "5b-smoke-v1",
  "prompt_hash": "smoke"
}

CALL 3: IMMEDIATELY after CALL 2 returns, call `log_progress` with
input `{"message": "Phase 5B smoke complete", "level": "INFO"}`.

After CALL 3 returns, you are done. No further output.

If CALL 1 returns count=0 (empty corpus): skip CALL 2, call
`log_progress` with level=WARNING message "empty corpus", and stop.

If any call returns an `error` payload: call `log_progress` once with
the error message at level ERROR, then stop. Do not retry.
"""


SMOKE_KICKOFF_MESSAGE = (
    "Phase 5B smoke test. Follow the system prompt exactly. "
    "Begin with `query_corpus`."
)


__all__ = ["SMOKE_KICKOFF_MESSAGE", "SMOKE_PROMPT_VERSION", "SMOKE_SYSTEM_PROMPT"]
