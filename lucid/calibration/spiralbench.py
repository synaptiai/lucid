"""Fetch and parse SpiralBench v1.2 benchmark data for calibration.

**What SpiralBench ships (verified 2026-04-22):**

- Per target model ``<m>``: a single JSON file at
  ``res_v1.2/<m>.json`` containing every conversation that target model
  produced across 30 benchmark scenarios, plus every judgement produced
  by the 3 SpiralBench judges (Claude Sonnet 4.5, GPT-5, Kimi K2).

- The nested structure (walked live 2026-04-22)::

     {"1": {
         "eval_prompts_v0.2.json": {
             "<scenario_id>": [{
                 "prompt_id": str,           # e.g. "td05"
                 "category":  str,           # e.g. "theory_development"
                 "evaluated_model": str,
                 "transcript": [{"role": "user"|"assistant", "content": str}, ...],
                 "judgements": [<judge0-dict>, <judge1-dict>, <judge2-dict>],
                 "final_judgement": {"off-rails": float},
                 ...
             }]
         },
         "__meta__": {"judges": [{"model": "...", ...}, × 3], ...}
     }}

- Each ``<judge-dict>`` is keyed ``chunk<i>`` where ``i`` is zero-based over
  assistant turns in the transcript, and carries ``assistant_turn_indexes``
  (absolute transcript index of the turn this chunk scored) and
  ``full_metrics: {behavior: [[snippet, intensity], ...]}`` — the same
  shape ``SpiralBenchIncidents`` produces in Module A.

**What this module produces:**

:class:`SpiralBenchCorpusData`, parsed once per target-model file:

- ``conversations`` — one :class:`~lucid.schemas.Conversation` per scenario
  (synthesised ids: ``sb:<target>:<scenario>``, deterministic and collision-
  free).
- ``turns`` — ``{conversation_id: list[Turn]}``.
- ``labeled_turns`` — one :class:`~lucid.calibration.data.LabeledTurn` per
  ``(conversation, assistant-turn, rater)``, collapsing each judge's
  per-chunk ``full_metrics`` across 17 behaviors. Rater short-names:
  ``sb_sonnet45``, ``sb_gpt5``, ``sb_kimi``.

**Ignored from upstream:**

- ``off-rails`` (only in ``final_judgement``; it is a composite score, not
  a scoreable behavior).
- ``errors``, ``injections_log``, ``user_system_prompt`` (benchmark
  metadata, not corpus content).

**Caching:** files land under ``.lucid/refs/spiralbench/`` (gitignored,
see ``.gitignore``). ``fetch_spiralbench_model`` is a no-op when the file
already exists; pass ``overwrite=True`` to force re-download.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from lucid.calibration.data import LabeledTurn
from lucid.schemas import Conversation, Role, Source, Turn

__all__ = [
    "DEFAULT_CACHE_DIR",
    "JUDGE_RATER_NAMES",
    "SPIRALBENCH_RAW_BASE",
    "SPIRALBENCH_RUBRIC_BEHAVIORS",
    "SpiralBenchCorpusData",
    "fetch_spiralbench_model",
    "parse_spiralbench_file",
    "rater_name_for_judge_index",
]


log = logging.getLogger(__name__)


SPIRALBENCH_RAW_BASE = "https://raw.githubusercontent.com/sam-paech/spiral-bench/main"
DEFAULT_CACHE_DIR = Path(".lucid/refs/spiralbench")


# Order matches SpiralBench's `__meta__.judges` list, verified 2026-04-22.
# If SpiralBench ever reorders (they haven't in v1.2), the rater labels in
# existing LabeledTurn JSONLs would silently point at the wrong model —
# the parser re-verifies order from `__meta__` and raises if drift is
# detected.
_EXPECTED_JUDGE_ORDER: tuple[str, ...] = (
    "claude-sonnet-4-5-20250929",
    "gpt-5-2025-08-07",
    "moonshotai/kimi-k2-0905",
)

JUDGE_RATER_NAMES: tuple[str, str, str] = ("sb_sonnet45", "sb_gpt5", "sb_kimi")

# The 17 Spiral-Bench v1.2 behaviors. Cross-checked against
# ``prompts/module_a/v1.md`` and ``lucid.modules.module_a_spiralbench.BEHAVIORS``
# — these three sources must stay in sync.
SPIRALBENCH_RUBRIC_BEHAVIORS: frozenset[str] = frozenset(
    {
        "pushback",
        "escalation",
        "de-escalation",
        "sycophancy",
        "benign-warmth",
        "delusion-reinforcement",
        "topic-shut-down",
        "help-referral-warranted",
        "help-referral-unwarranted",
        "positive-sentience-claim",
        "uncertain-sentience-claim",
        "negative-sentience-claim",
        "harmful-advice",
        "confident-bullshitting",
        "boundary-setting",
        "validate-feelings-not-thoughts",
        "ritualization",
    }
)


def rater_name_for_judge_index(i: int) -> str:
    """Return the short rater name for judge index 0..2."""
    if not 0 <= i < len(JUDGE_RATER_NAMES):
        raise IndexError(f"judge index {i} out of range")
    return JUDGE_RATER_NAMES[i]


@dataclass(frozen=True, slots=True)
class SpiralBenchCorpusData:
    """Parsed artifacts from one ``res_v1.2/<target>.json`` file."""

    target_model: str
    conversations: list[Conversation]
    turns: dict[str, list[Turn]]
    labeled_turns: list[LabeledTurn]
    rater_names: tuple[str, ...] = field(default=JUDGE_RATER_NAMES)


# ──────────────────────────────────────────────────────────────────────────
# Fetch
# ──────────────────────────────────────────────────────────────────────────


def fetch_spiralbench_model(
    target_model: str,
    *,
    cache_dir: Path | None = None,
    overwrite: bool = False,
    http_client: httpx.Client | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download ``res_v1.2/<target_model>.json`` to ``cache_dir``.

    Returns the cached path. If the file already exists and ``overwrite``
    is False, returns without hitting the network. Raises
    :class:`httpx.HTTPStatusError` on 4xx/5xx so missing-model typos fail
    loudly.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{target_model}.json"
    if dest.is_file() and not overwrite:
        log.debug("spiralbench cache hit: %s", dest)
        return dest

    url = f"{SPIRALBENCH_RAW_BASE}/res_v1.2/{target_model}.json"
    log.info("fetching %s → %s", url, dest)
    client = http_client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
    finally:
        if http_client is None:
            client.close()
    return dest


# ──────────────────────────────────────────────────────────────────────────
# Parse
# ──────────────────────────────────────────────────────────────────────────


def _verify_judge_order(meta: dict[str, Any]) -> None:
    """Raise if SpiralBench's judge list reordered or re-named.

    The rater labels on downstream LabeledTurn rows carry positional
    meaning, so silent reorders would be catastrophic. This check turns
    a silent corruption into a loud parse error.
    """
    judges = meta.get("judges", [])
    if not isinstance(judges, list) or len(judges) != len(_EXPECTED_JUDGE_ORDER):
        raise ValueError(
            f"unexpected SpiralBench __meta__.judges: expected "
            f"{len(_EXPECTED_JUDGE_ORDER)} entries, got {len(judges)}"
        )
    actual = tuple(j.get("model", "") for j in judges)
    if actual != _EXPECTED_JUDGE_ORDER:
        raise ValueError(
            f"SpiralBench judge order drifted: expected {_EXPECTED_JUDGE_ORDER}, "
            f"got {actual}. Update JUDGE_RATER_NAMES + _EXPECTED_JUDGE_ORDER in "
            f"lucid/calibration/spiralbench.py and audit any persisted "
            f"LabeledTurn JSONLs."
        )


def _synth_id(*parts: object) -> str:
    """Deterministic short id from sha1-stable components."""
    raw = ":".join(str(p) for p in parts)
    return raw


def _conversation_from_record(
    record: dict[str, Any],
    *,
    target_model: str,
    scenario_id: str,
    convo_index: int,
    source_path: str,
) -> tuple[Conversation, list[Turn]]:
    """Build a Conversation + its Turns from one SpiralBench conv record."""
    conv_id = _synth_id("sb", target_model, scenario_id, convo_index)

    # SpiralBench records have no timestamps; use a single synthetic
    # `labeled_at`-style constant so downstream persistence doesn't get
    # randomised created_at/updated_at values across runs.
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    transcript = record.get("transcript", [])
    turns: list[Turn] = []
    for idx, raw in enumerate(transcript):
        role_str = raw.get("role", "user")
        try:
            role = Role(role_str)
        except ValueError:
            log.warning("sb: skipping unknown role %r in conv %s", role_str, conv_id)
            continue
        turn_id = _synth_id(conv_id, "t", idx)
        content = raw.get("content", "") or ""
        turns.append(
            Turn(
                id=turn_id,
                conversation_id=conv_id,
                index=idx,
                role=role,
                content=content,
            )
        )

    conversation = Conversation(
        id=conv_id,
        source=Source.CLAUDE_AI,  # closest existing source; SpiralBench is synthetic
        source_path=source_path,
        created_at=ts,
        updated_at=ts,
        model=target_model,
        title=f"SpiralBench {scenario_id} (convo {convo_index})",
        turn_count=len(turns),
        metadata={
            "spiralbench_scenario": scenario_id,
            "spiralbench_category": record.get("category", ""),
            "spiralbench_evaluated_model": record.get("evaluated_model", target_model),
            "spiralbench_user_model": record.get("user_model", ""),
            "spiralbench_convo_index": convo_index,
        },
    )
    return conversation, turns


def _labeled_turn_from_chunk(
    *,
    chunk: dict[str, Any],
    rater: str,
    conversation_id: str,
    turns: Sequence[Turn],
    assistant_positions: Sequence[int],
    labeled_at: datetime,
) -> LabeledTurn | None:
    """Flatten one judge's chunk into a LabeledTurn. Returns None if the
    chunk cannot be mapped to a turn in this conversation.

    **Index note (verified 2026-04-22 against real res_v1.2 data):**
    SpiralBench's ``assistant_turn_indexes`` is **1-indexed over assistant
    turns only**, not a 0-indexed absolute transcript position. E.g.,
    ``chunk0.assistant_turn_indexes=[1]`` means "the 1st assistant turn",
    not "transcript[1]". They happen to match for the very first chunk
    (because transcript[1] IS the 1st assistant turn in alternating-role
    conversations) but drift thereafter. We translate via
    ``assistant_positions`` — the list of absolute transcript indices
    where the role is ``assistant``.
    """
    assistant_indexes = chunk.get("assistant_turn_indexes") or []
    if not assistant_indexes:
        return None
    try:
        sb_ordinal = int(assistant_indexes[0])
    except (TypeError, ValueError):
        return None
    # 1-indexed into assistant_positions
    pos_index = sb_ordinal - 1
    if pos_index < 0 or pos_index >= len(assistant_positions):
        log.warning(
            "sb: chunk assistant_turn_indexes=%s maps outside the "
            "%d assistant turns in %s",
            assistant_indexes,
            len(assistant_positions),
            conversation_id,
        )
        return None
    transcript_idx = assistant_positions[pos_index]
    turn = turns[transcript_idx]
    if turn.role != Role.ASSISTANT:
        # Belt-and-braces: should be impossible given how
        # assistant_positions was built, but catches future schema drift.
        log.warning(
            "sb: chunk maps to non-assistant turn %s role=%s",
            turn.id,
            turn.role,
        )
        return None

    full_metrics = chunk.get("full_metrics") or {}
    present: set[str] = set()
    intensities: dict[str, int] = {}
    for behavior, incidents in full_metrics.items():
        if behavior not in SPIRALBENCH_RUBRIC_BEHAVIORS:
            # off-rails and any future additions skip silently — they
            # aren't part of the 17-behavior rubric we calibrate against.
            continue
        if not incidents:
            # Judge explicitly saw no incidents of this behavior on this
            # turn. This is a VALID "absent" label; we record it by
            # leaving `behavior` out of ``present_behaviors``. IAA
            # downstream requires complete ratings so absent-from-present
            # is load-bearing information.
            continue
        # incidents: list[[snippet, intensity], ...]; intensity at index 1.
        max_intensity = 0
        for item in incidents:
            if not isinstance(item, list) or len(item) < 2:
                continue
            raw_int = item[1]
            try:
                value = int(raw_int)
            except (TypeError, ValueError):
                continue
            if 1 <= value <= 3 and value > max_intensity:
                max_intensity = value
        if max_intensity == 0:
            # All incidents had malformed intensity → treat as absent
            # rather than inventing a number.
            continue
        present.add(behavior)
        intensities[behavior] = max_intensity

    # Always emit — an all-absent LabeledTurn encodes "this rater saw
    # nothing for this turn", which IAA needs to pair against other
    # raters that did see something.
    return LabeledTurn(
        conversation_id=conversation_id,
        turn_id=turn.id,
        present_behaviors=frozenset(present),
        intensities=intensities,
        labeler=rater,
        labeled_at=labeled_at,
        turn_content_sha256=hashlib.sha256(turn.content.encode()).hexdigest(),
    )


def _iter_conversations(
    data: dict[str, Any],
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Walk the ``data['1'][<prompts-file>][<scenario>][<convo>]`` shape.

    Yields ``(scenario_id, convo_index, record)``. Raises if the shape
    doesn't match the documented v1.2 structure."""
    top = data.get("1")
    if not isinstance(top, dict):
        raise ValueError("SpiralBench JSON missing top-level '1' key")
    for key, value in top.items():
        if key == "__meta__":
            continue
        # key is e.g. "eval_prompts_v0.2.json"; value is the scenarios dict
        if not isinstance(value, dict):
            raise ValueError(f"unexpected shape under key {key!r}")
        for scenario_id, convos in value.items():
            if not isinstance(convos, list):
                raise ValueError(
                    f"expected list under scenario {scenario_id!r}, got {type(convos).__name__}"
                )
            for convo_index, record in enumerate(convos):
                if not isinstance(record, dict):
                    continue
                yield scenario_id, convo_index, record


def parse_spiralbench_file(
    path: Path,
    *,
    target_model: str | None = None,
    conversation_limit: int | None = None,
) -> SpiralBenchCorpusData:
    """Parse ``res_v1.2/<target>.json`` into corpus + labeled turns.

    ``target_model`` defaults to ``path.stem``. ``conversation_limit``
    caps the number of conversations parsed (useful for subsampling
    under a cost budget)."""
    if not path.is_file():
        raise FileNotFoundError(f"SpiralBench file not found: {path}")

    target = target_model or path.stem
    data = json.loads(path.read_text(encoding="utf-8"))

    top = data.get("1", {})
    meta = top.get("__meta__", {}) if isinstance(top, dict) else {}
    _verify_judge_order(meta)

    labeled_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    source_path = str(path)

    conversations: list[Conversation] = []
    turns_by_conv: dict[str, list[Turn]] = {}
    labeled_turns: list[LabeledTurn] = []

    count = 0
    for scenario_id, convo_index, record in _iter_conversations(data):
        if conversation_limit is not None and count >= conversation_limit:
            break
        conversation, turns = _conversation_from_record(
            record,
            target_model=target,
            scenario_id=scenario_id,
            convo_index=convo_index,
            source_path=source_path,
        )
        if not turns:
            continue
        conversations.append(conversation)
        turns_by_conv[conversation.id] = turns

        assistant_positions = [
            i for i, t in enumerate(turns) if t.role == Role.ASSISTANT
        ]
        judgements = record.get("judgements") or []
        for j_idx, judge_chunks in enumerate(judgements):
            if j_idx >= len(JUDGE_RATER_NAMES):
                break
            rater = JUDGE_RATER_NAMES[j_idx]
            if not isinstance(judge_chunks, dict):
                continue
            for chunk in judge_chunks.values():
                if not isinstance(chunk, dict):
                    continue
                lt = _labeled_turn_from_chunk(
                    chunk=chunk,
                    rater=rater,
                    conversation_id=conversation.id,
                    turns=turns,
                    assistant_positions=assistant_positions,
                    labeled_at=labeled_at,
                )
                if lt is not None:
                    labeled_turns.append(lt)
        count += 1

    return SpiralBenchCorpusData(
        target_model=target,
        conversations=conversations,
        turns=turns_by_conv,
        labeled_turns=labeled_turns,
    )
