"""Tests for :mod:`lucid.calibration.spiralbench`.

Parser tested against a hand-crafted fixture whose shape mirrors the real
``res_v1.2/<target>.json`` schema verified 2026-04-22. Network fetch
tested via a ``respx``-style httpx mock — ``httpx`` already ships
``MockTransport`` so no extra dep needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from lucid.calibration.spiralbench import (
    JUDGE_RATER_NAMES,
    SPIRALBENCH_RUBRIC_BEHAVIORS,
    fetch_spiralbench_model,
    parse_spiralbench_file,
    rater_name_for_judge_index,
)
from lucid.schemas import Role, Source

# ──────────────────────────────────────────────────────────────────────────
# Fixture builder — mirrors real SpiralBench v1.2 shape
# ──────────────────────────────────────────────────────────────────────────


def _mini_spiralbench_file() -> dict:
    """A tiny but schema-accurate SpiralBench file.

    Two scenarios, one conversation each, each with 4 turns (u/a/u/a) and
    judgements from all 3 judges on both assistant turns. One judge
    reports sycophancy at intensity 3 on chunk0; another reports nothing;
    the third reports pushback at intensity 1 on chunk1. Plus an
    ``off-rails`` entry in full_metrics that the parser should drop.
    """

    def _chunk(sb_ordinal: int, full_metrics: dict) -> dict:
        # sb_ordinal is 1-indexed over assistant turns, matching
        # SpiralBench's verified semantics.
        return {
            "metrics": {b: 0 for b in SPIRALBENCH_RUBRIC_BEHAVIORS},
            "full_metrics": full_metrics,
            "raw_text": "…",
            "assistant_turn_indexes": [sb_ordinal],
            "assistant_length_chars": 200,
        }

    def _empty_chunk(sb_ordinal: int) -> dict:
        return _chunk(sb_ordinal, {b: [] for b in SPIRALBENCH_RUBRIC_BEHAVIORS})

    transcript_s1 = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},  # 1st assistant turn
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},  # 2nd assistant turn
    ]

    # Judge 0 (sb_sonnet45): sycophancy i=3 on chunk0 (1st assistant), otherwise empty
    judge0_s1 = {
        "chunk0": _chunk(
            1,  # 1st assistant turn
            {
                **{b: [] for b in SPIRALBENCH_RUBRIC_BEHAVIORS},
                "sycophancy": [["You're so smart", 3], ["And brilliant", 2]],
            },
        ),
        "chunk1": _empty_chunk(2),  # 2nd assistant turn
    }
    # Judge 1 (sb_gpt5): absolutely nothing
    judge1_s1 = {
        "chunk0": _empty_chunk(1),
        "chunk1": _empty_chunk(2),
    }
    # Judge 2 (sb_kimi): pushback i=1 on chunk1 (2nd asst) + stray off-rails (dropped)
    judge2_s1 = {
        "chunk0": _empty_chunk(1),
        "chunk1": _chunk(
            2,  # 2nd assistant turn
            {
                **{b: [] for b in SPIRALBENCH_RUBRIC_BEHAVIORS},
                "pushback": [["Actually that's not…", 1]],
                "off-rails": [["some final comment", 2]],  # ← must be dropped
            },
        ),
    }

    return {
        "1": {
            "eval_prompts_v0.2.json": {
                "td01": [
                    {
                        "prompt_id": "td01",
                        "category": "theory_development",
                        "evaluated_model": "anthropic/claude-sonnet-4.5",
                        "user_model": "moonshotai/kimi-k2",
                        "transcript": transcript_s1,
                        "judgements": [judge0_s1, judge1_s1, judge2_s1],
                        "final_judgement": {"off-rails": 5.0},
                    }
                ],
                "sc01": [
                    {
                        "prompt_id": "sc01",
                        "category": "safety_critique",
                        "evaluated_model": "anthropic/claude-sonnet-4.5",
                        "user_model": "moonshotai/kimi-k2",
                        "transcript": transcript_s1,  # reuse
                        "judgements": [judge1_s1, judge1_s1, judge1_s1],  # all empty
                        "final_judgement": {"off-rails": 0.0},
                    }
                ],
            },
            "__meta__": {
                "judges": [
                    {"model": "claude-sonnet-4-5-20250929", "base_url": "x"},
                    {"model": "gpt-5-2025-08-07", "base_url": "y"},
                    {"model": "moonshotai/kimi-k2-0905", "base_url": "z"},
                ]
            },
        }
    }


@pytest.fixture
def mini_sb_file(tmp_path: Path) -> Path:
    path = tmp_path / "claude-sonnet-4.5.json"
    path.write_text(json.dumps(_mini_spiralbench_file()), encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────


def test_parse_emits_one_conversation_per_scenario(mini_sb_file: Path) -> None:
    data = parse_spiralbench_file(mini_sb_file)
    assert len(data.conversations) == 2
    ids = {c.id for c in data.conversations}
    assert "sb:claude-sonnet-4.5:td01:0" in ids
    assert "sb:claude-sonnet-4.5:sc01:0" in ids


def test_parse_populates_turns_with_correct_roles(mini_sb_file: Path) -> None:
    data = parse_spiralbench_file(mini_sb_file)
    conv_id = "sb:claude-sonnet-4.5:td01:0"
    turns = data.turns[conv_id]
    assert len(turns) == 4
    assert [t.role for t in turns] == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]
    # Turn ids are deterministic
    assert turns[1].id == f"{conv_id}:t:1"


def test_parse_sets_conversation_model_to_target(mini_sb_file: Path) -> None:
    data = parse_spiralbench_file(mini_sb_file, target_model="claude-sonnet-4.5")
    for conv in data.conversations:
        assert conv.model == "claude-sonnet-4.5"
        assert conv.source == Source.CLAUDE_AI
        # Metadata preserves SpiralBench-specific provenance
        assert "spiralbench_scenario" in conv.metadata
        assert "spiralbench_category" in conv.metadata


def test_parse_emits_complete_ratings_across_raters(mini_sb_file: Path) -> None:
    """Every rater must produce a LabeledTurn for every chunk they scored
    — even if all-empty. IAA computations require complete ratings,
    so "this rater saw nothing" is a load-bearing label.
    """
    data = parse_spiralbench_file(mini_sb_file)
    # Fixture: 2 scenarios × 1 conversation each × 2 chunks × 3 judges = 12 LabeledTurns
    raters = {lt.labeler for lt in data.labeled_turns}
    assert raters == {"sb_sonnet45", "sb_gpt5", "sb_kimi"}
    assert len(data.labeled_turns) == 12

    # Each (conv, turn, rater) tuple is unique and fully populated
    from collections import Counter

    per_turn = Counter((lt.conversation_id, lt.turn_id) for lt in data.labeled_turns)
    for key, count in per_turn.items():
        assert count == 3, f"turn {key} has {count} raters, expected 3"


def test_parse_empty_chunk_yields_absent_labeled_turn(mini_sb_file: Path) -> None:
    """A chunk whose judge saw nothing still produces a LabeledTurn with
    empty present_behaviors — that is the "absent" rating other raters
    need to compare against."""
    data = parse_spiralbench_file(mini_sb_file)
    # Judge 1 (sb_gpt5) emitted all-empty for every chunk in the fixture
    gpt5_lts = [lt for lt in data.labeled_turns if lt.labeler == "sb_gpt5"]
    assert gpt5_lts, "sb_gpt5 must still produce LabeledTurns for empty chunks"
    for lt in gpt5_lts:
        assert lt.present_behaviors == frozenset()
        assert lt.intensities == {}


def test_parse_maps_sb_ordinal_to_correct_assistant_turn(mini_sb_file: Path) -> None:
    """Regression test for the 1-indexed-assistant-ordinal parsing bug.

    Verified against real res_v1.2/claude-sonnet-4.5.json on 2026-04-22:
    SpiralBench's ``assistant_turn_indexes`` is 1-indexed over
    **assistant turns only**, not a 0-indexed transcript position. For a
    4-turn [user, assistant, user, assistant] transcript, a chunk with
    ``assistant_turn_indexes=[2]`` must map to the SECOND assistant turn,
    which is at transcript position 3.
    """
    data = parse_spiralbench_file(mini_sb_file)
    kimi_pushback = [
        lt
        for lt in data.labeled_turns
        if lt.labeler == "sb_kimi" and "pushback" in lt.present_behaviors
    ]
    assert len(kimi_pushback) == 1
    # Kimi's pushback landed on chunk1 (sb_ordinal=2) → 2nd assistant → transcript[3]
    expected_turn_id = "sb:claude-sonnet-4.5:td01:0:t:3"
    assert kimi_pushback[0].turn_id == expected_turn_id


def test_parse_rolls_up_intensity_to_max_per_chunk(mini_sb_file: Path) -> None:
    data = parse_spiralbench_file(mini_sb_file)
    # sonnet45's td01 chunk0: incidents at intensities 3 and 2 → max=3
    sonnet_lts = [
        lt
        for lt in data.labeled_turns
        if lt.labeler == "sb_sonnet45" and "sycophancy" in lt.present_behaviors
    ]
    assert sonnet_lts
    assert sonnet_lts[0].intensities["sycophancy"] == 3


def test_parse_drops_off_rails_and_unknown_behaviors(mini_sb_file: Path) -> None:
    """``off-rails`` in full_metrics must not leak through as a behavior."""
    data = parse_spiralbench_file(mini_sb_file)
    for lt in data.labeled_turns:
        assert "off-rails" not in lt.present_behaviors
        assert "off-rails" not in lt.intensities
        # All behaviors must be from the canonical 17
        assert lt.present_behaviors <= SPIRALBENCH_RUBRIC_BEHAVIORS


def test_parse_conversation_limit(mini_sb_file: Path) -> None:
    data = parse_spiralbench_file(mini_sb_file, conversation_limit=1)
    assert len(data.conversations) == 1


def test_parse_rejects_unexpected_judge_order(tmp_path: Path) -> None:
    """If SpiralBench reorders judges, we fail loudly rather than emit
    mis-attributed rater labels."""
    bad = _mini_spiralbench_file()
    # Swap judge 0 and judge 1 in meta
    meta = bad["1"]["__meta__"]
    meta["judges"][0], meta["judges"][1] = meta["judges"][1], meta["judges"][0]

    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="judge order drifted"):
        parse_spiralbench_file(path)


def test_parse_rejects_missing_meta_judges(tmp_path: Path) -> None:
    bad = _mini_spiralbench_file()
    bad["1"]["__meta__"]["judges"] = []
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 3 entries"):
        parse_spiralbench_file(path)


# ──────────────────────────────────────────────────────────────────────────
# LabeledTurn shape — matches validate.py expectations
# ──────────────────────────────────────────────────────────────────────────


def test_labeled_turns_match_data_loader_contract(mini_sb_file: Path) -> None:
    """LabeledTurn instances from SpiralBench should round-trip through
    the same JSONL format ``lucid.calibration.data.load_hand_labels`` uses,
    so downstream tooling doesn't need two separate formats."""
    from lucid.calibration.data import LabeledTurn

    data = parse_spiralbench_file(mini_sb_file)
    for lt in data.labeled_turns:
        encoded = lt.model_dump_json()
        decoded = LabeledTurn.model_validate_json(encoded)
        assert decoded == lt


# ──────────────────────────────────────────────────────────────────────────
# Fetch (httpx MockTransport — no network)
# ──────────────────────────────────────────────────────────────────────────


def test_fetch_writes_file_when_missing(tmp_path: Path) -> None:
    payload = json.dumps({"hello": "world"}).encode()

    def _handler(request: httpx.Request) -> httpx.Response:
        assert "res_v1.2/claude-sonnet-4.5.json" in str(request.url)
        return httpx.Response(200, content=payload)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    try:
        dest = fetch_spiralbench_model(
            "claude-sonnet-4.5",
            cache_dir=tmp_path,
            http_client=client,
        )
    finally:
        client.close()

    assert dest.is_file()
    assert dest.read_bytes() == payload


def test_fetch_is_idempotent_when_cached(tmp_path: Path) -> None:
    dest = tmp_path / "claude-sonnet-4.5.json"
    dest.write_bytes(b"cached")

    calls: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=b"from-network")

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    try:
        fetch_spiralbench_model("claude-sonnet-4.5", cache_dir=tmp_path, http_client=client)
    finally:
        client.close()

    assert dest.read_bytes() == b"cached"  # not overwritten
    assert calls == []  # no network call


def test_fetch_overwrite_force_refreshes(tmp_path: Path) -> None:
    dest = tmp_path / "claude-sonnet-4.5.json"
    dest.write_bytes(b"stale")

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"fresh")

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    try:
        fetch_spiralbench_model(
            "claude-sonnet-4.5",
            cache_dir=tmp_path,
            overwrite=True,
            http_client=client,
        )
    finally:
        client.close()
    assert dest.read_bytes() == b"fresh"


def test_fetch_raises_on_http_error(tmp_path: Path) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            fetch_spiralbench_model("does-not-exist", cache_dir=tmp_path, http_client=client)
    finally:
        client.close()


# ──────────────────────────────────────────────────────────────────────────
# Rater-name lookup
# ──────────────────────────────────────────────────────────────────────────


def test_rater_name_for_judge_index() -> None:
    assert rater_name_for_judge_index(0) == "sb_sonnet45"
    assert rater_name_for_judge_index(1) == "sb_gpt5"
    assert rater_name_for_judge_index(2) == "sb_kimi"


def test_rater_name_rejects_out_of_range() -> None:
    with pytest.raises(IndexError):
        rater_name_for_judge_index(3)


def test_judge_rater_names_cardinality() -> None:
    assert len(JUDGE_RATER_NAMES) == 3
