"""Tests for :mod:`lucid.calibration.judges`.

Covers the Judge Protocol, the ``findings_to_labeled_turns`` collapse
helper, and the three initial backends (Module A, SpiralBench file,
synthetic gold).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from lucid.calibration.data import LabeledTurn
from lucid.calibration.judges import (
    ModuleAJudge,
    SpiralBenchFileJudge,
    SyntheticGoldJudge,
    findings_to_labeled_turns,
)
from lucid.calibration.spiralbench import SpiralBenchCorpusData
from lucid.modules.base import ModuleCorpus
from lucid.modules.module_a_spiralbench import (
    SpiralBenchIncidents,
    SpiralBenchScore,
)
from lucid.schemas import (
    Conversation,
    Finding,
    ModuleName,
    Role,
    Source,
    Turn,
)


def _turn(conv_id: str, idx: int, role: Role, content: str = "") -> Turn:
    return Turn(
        id=f"{conv_id}-t{idx}",
        conversation_id=conv_id,
        index=idx,
        role=role,
        content=content or f"turn {idx}",
    )


def _corpus(conv_ids: list[str], turns_by_conv: dict[str, list[Turn]]) -> ModuleCorpus:
    convs = {
        cid: Conversation(
            id=cid,
            source=Source.CLAUDE_AI,
            source_path="/tmp/x",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 2, 1, tzinfo=UTC),
            turn_count=len(turns_by_conv[cid]),
        )
        for cid in conv_ids
    }
    return ModuleCorpus(
        conversations=convs,
        turns_by_conversation=turns_by_conv,
        audit_run_id="run-test",
    )


def _finding(
    *,
    conv_id: str,
    turn_id: str,
    behavior: str,
    intensity: int,
) -> Finding:
    return Finding(
        id=f"f:{conv_id}:{turn_id}:{behavior}",
        audit_run_id="run-test",
        conversation_id=conv_id,
        turn_ids=[turn_id],
        turn_ids_hash="h",
        module=ModuleName.A_SPIRALBENCH,
        behavior=behavior,
        intensity=intensity,
        confidence=0.8,
        explanation="test",
        citation="Spiral-Bench v1.2",
        detected_by=["claude-opus-4-7"],
        detected_at=datetime(2026, 4, 22, tzinfo=UTC),
        prompt_version="v1",
        prompt_hash="abc",
    )


# ──────────────────────────────────────────────────────────────────────────
# findings_to_labeled_turns
# ──────────────────────────────────────────────────────────────────────────


def test_findings_to_labeled_turns_emits_one_per_assistant_turn() -> None:
    """Every assistant turn gets a LabeledTurn, even if no Findings
    reference it. User turns do not."""
    conv = "c1"
    turns = [
        _turn(conv, 0, Role.USER),
        _turn(conv, 1, Role.ASSISTANT),
        _turn(conv, 2, Role.USER),
        _turn(conv, 3, Role.ASSISTANT),
    ]
    corpus = _corpus([conv], {conv: turns})
    findings = [
        _finding(conv_id=conv, turn_id="c1-t1", behavior="sycophancy", intensity=2),
    ]
    rows = findings_to_labeled_turns(findings, corpus, rater_name="mr_rater")
    # 2 assistant turns → 2 LabeledTurns
    assert len(rows) == 2
    assert all(r.labeler == "mr_rater" for r in rows)
    # Only turn 1 has a behavior; turn 3 is present-empty.
    turn_to_row = {r.turn_id: r for r in rows}
    assert "sycophancy" in turn_to_row["c1-t1"].present_behaviors
    assert turn_to_row["c1-t1"].intensities["sycophancy"] == 2
    assert turn_to_row["c1-t3"].present_behaviors == frozenset()


def test_findings_to_labeled_turns_rolls_up_same_behavior_same_turn() -> None:
    """Two Findings on the same (turn, behavior) collapse to max intensity."""
    conv = "c1"
    turns = [_turn(conv, 0, Role.USER), _turn(conv, 1, Role.ASSISTANT)]
    corpus = _corpus([conv], {conv: turns})
    findings = [
        _finding(conv_id=conv, turn_id="c1-t1", behavior="pushback", intensity=1),
        _finding(conv_id=conv, turn_id="c1-t1", behavior="pushback", intensity=3),
    ]
    rows = findings_to_labeled_turns(findings, corpus, rater_name="r")
    assert len(rows) == 1
    assert rows[0].intensities["pushback"] == 3


def test_findings_to_labeled_turns_ignores_errors_and_other_modules() -> None:
    from lucid.modules.base import ModuleError

    conv = "c1"
    turns = [_turn(conv, 0, Role.USER), _turn(conv, 1, Role.ASSISTANT)]
    corpus = _corpus([conv], {conv: turns})
    err = ModuleError(
        module=ModuleName.A_SPIRALBENCH,
        conversation_id=conv,
        error_type="x",
        message="boom",
    )
    other_module_finding = Finding(
        id="z",
        audit_run_id="run-test",
        conversation_id=conv,
        turn_ids=["c1-t1"],
        turn_ids_hash="h",
        module=ModuleName.G_ATTRIBUTION,
        behavior="model=x",
        intensity=None,
        confidence=1.0,
        explanation="...",
        citation="...",
        detected_by=["x"],
        detected_at=datetime(2026, 4, 22, tzinfo=UTC),
        prompt_version="v1",
        prompt_hash="abc",
    )
    rows = findings_to_labeled_turns([err, other_module_finding], corpus, rater_name="r")
    assert len(rows) == 1
    assert rows[0].present_behaviors == frozenset()


# ──────────────────────────────────────────────────────────────────────────
# ModuleAJudge (integration via mock client)
# ──────────────────────────────────────────────────────────────────────────


async def test_module_a_judge_produces_labeled_turns(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    conv = "c1"
    turns = [
        _turn(conv, 0, Role.USER),
        _turn(conv, 1, Role.ASSISTANT, "A reply 1"),
        _turn(conv, 2, Role.USER),
        _turn(conv, 3, Role.ASSISTANT, "A reply 2"),
    ]
    corpus = _corpus([conv], {conv: turns})

    score = SpiralBenchScore(
        reasoning="test",
        incidents=SpiralBenchIncidents.model_validate(
            {"sycophancy": [{"snippet": "flattery", "intensity": 2, "turn_index": 0}]}
        ),
    )
    client = mock_anthropic_client(parse_outputs=[score])
    judge = ModuleAJudge(client=client, chunk_size=10)
    rows = await judge.run(corpus)

    # Every assistant turn → one LabeledTurn
    assert len(rows) == 2
    assert all(r.labeler == "module_a_c10" for r in rows)
    # One with sycophancy, one empty
    tagged = {r.turn_id: r for r in rows}
    assert "sycophancy" in tagged["c1-t1"].present_behaviors
    assert tagged["c1-t3"].present_behaviors == frozenset()


async def test_module_a_judge_chunk_size_configurable(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    conv = "c1"
    turns = [_turn(conv, i, Role.USER if i % 2 == 0 else Role.ASSISTANT) for i in range(6)]
    corpus = _corpus([conv], {conv: turns})

    # chunk_size=2 = 1 assistant turn per chunk → 3 calls for 3 asst turns
    from lucid.modules.module_a_spiralbench import SpiralBenchIncidents

    empty = SpiralBenchScore(reasoning="", incidents=SpiralBenchIncidents())
    client = mock_anthropic_client(parse_outputs=[empty, empty, empty])
    judge = ModuleAJudge(client=client, chunk_size=2)
    rows = await judge.run(corpus)

    assert client.messages.create.await_count == 3
    assert judge.rater_name == "module_a_c2"
    assert len(rows) == 3  # 3 assistant turns


async def test_module_a_judge_custom_rater_name(
    mock_anthropic_client: Callable[..., Any],
) -> None:
    client = mock_anthropic_client(parse_outputs=[])
    judge = ModuleAJudge(client=client, chunk_size=10, rater_name="custom")
    assert judge.rater_name == "custom"


# ──────────────────────────────────────────────────────────────────────────
# SpiralBenchFileJudge
# ──────────────────────────────────────────────────────────────────────────


def _sb_data(conv_id: str, raters: tuple[str, ...]) -> SpiralBenchCorpusData:
    """Build a minimal SpiralBenchCorpusData with one assistant turn and
    the given set of raters."""
    from lucid.calibration.spiralbench import JUDGE_RATER_NAMES

    turn_id = f"{conv_id}-t1"
    turns = [_turn(conv_id, 0, Role.USER), _turn(conv_id, 1, Role.ASSISTANT)]
    conv = Conversation(
        id=conv_id,
        source=Source.CLAUDE_AI,
        source_path="/tmp",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        turn_count=2,
    )
    labeled = [
        LabeledTurn(
            conversation_id=conv_id,
            turn_id=turn_id,
            present_behaviors=frozenset({"sycophancy"}),
            intensities={"sycophancy": 2},
            labeler=rater,
            labeled_at=datetime(2026, 4, 22, tzinfo=UTC),
        )
        for rater in raters
    ]
    return SpiralBenchCorpusData(
        target_model="test-target",
        conversations=[conv],
        turns={conv_id: turns},
        labeled_turns=labeled,
        rater_names=JUDGE_RATER_NAMES,
    )


async def test_spiralbench_file_judge_filters_to_corpus() -> None:
    data = _sb_data("c1", raters=("sb_sonnet45", "sb_gpt5", "sb_kimi"))
    # Add stray labels for a conversation NOT in the corpus
    data.labeled_turns.append(
        LabeledTurn(
            conversation_id="different-conv",
            turn_id="x",
            present_behaviors=frozenset(),
            intensities={},
            labeler="sb_sonnet45",
            labeled_at=datetime(2026, 4, 22, tzinfo=UTC),
        )
    )
    judge = SpiralBenchFileJudge(data, "sb_sonnet45")
    corpus = _corpus(["c1"], {"c1": data.turns["c1"]})

    rows = await judge.run(corpus)
    assert len(rows) == 1
    assert rows[0].conversation_id == "c1"
    assert rows[0].labeler == "sb_sonnet45"


async def test_spiralbench_file_judge_one_per_rater() -> None:
    """Three judges → three SpiralBenchFileJudge instances, each returns
    only its own rater's labels."""
    data = _sb_data("c1", raters=("sb_sonnet45", "sb_gpt5", "sb_kimi"))
    corpus = _corpus(["c1"], {"c1": data.turns["c1"]})
    for r in data.rater_names:
        judge = SpiralBenchFileJudge(data, r)
        rows = await judge.run(corpus)
        assert all(row.labeler == r for row in rows)


def test_spiralbench_file_judge_rejects_unknown_rater() -> None:
    data = _sb_data("c1", raters=("sb_sonnet45",))
    with pytest.raises(ValueError, match="not in SpiralBench data"):
        SpiralBenchFileJudge(data, "nonexistent")


# ──────────────────────────────────────────────────────────────────────────
# SyntheticGoldJudge
# ──────────────────────────────────────────────────────────────────────────


async def test_synthetic_gold_judge_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    lt = LabeledTurn(
        conversation_id="syn:c1",
        turn_id="syn:c1:t1",
        present_behaviors=frozenset({"pushback"}),
        intensities={"pushback": 2},
        labeler="gold_author",  # original labeler, re-tagged by judge
        labeled_at=datetime(2026, 4, 22, tzinfo=UTC),
    )
    path.write_text(lt.model_dump_json() + "\n", encoding="utf-8")

    judge = SyntheticGoldJudge(path)
    corpus = _corpus(
        ["syn:c1"],
        {"syn:c1": [_turn("syn:c1", 0, Role.USER), _turn("syn:c1", 1, Role.ASSISTANT)]},
    )
    rows = await judge.run(corpus)

    assert len(rows) == 1
    # Re-tagged
    assert rows[0].labeler == "synthetic_gold"
    # Preserved other fields
    assert rows[0].present_behaviors == frozenset({"pushback"})
    assert rows[0].intensities["pushback"] == 2


async def test_synthetic_gold_judge_filters_to_corpus(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    keep = LabeledTurn(
        conversation_id="syn:c1",
        turn_id="syn:c1:t1",
        labeler="x",
        labeled_at=datetime(2026, 4, 22, tzinfo=UTC),
    )
    skip = LabeledTurn(
        conversation_id="syn:other",
        turn_id="syn:other:t1",
        labeler="x",
        labeled_at=datetime(2026, 4, 22, tzinfo=UTC),
    )
    path.write_text(
        keep.model_dump_json() + "\n" + skip.model_dump_json() + "\n",
        encoding="utf-8",
    )
    judge = SyntheticGoldJudge(path)
    corpus = _corpus(["syn:c1"], {"syn:c1": []})
    rows = await judge.run(corpus)
    assert len(rows) == 1
    assert rows[0].conversation_id == "syn:c1"


# ──────────────────────────────────────────────────────────────────────────
# Helper-coverage sanity
# ──────────────────────────────────────────────────────────────────────────


def test_findings_to_labeled_turns_honors_conversation_boundaries() -> None:
    corpus = _corpus(
        ["c1", "c2"],
        {
            "c1": [_turn("c1", 0, Role.USER), _turn("c1", 1, Role.ASSISTANT)],
            "c2": [_turn("c2", 0, Role.USER), _turn("c2", 1, Role.ASSISTANT)],
        },
    )
    findings = [
        _finding(conv_id="c1", turn_id="c1-t1", behavior="pushback", intensity=1),
    ]
    rows = findings_to_labeled_turns(findings, corpus, rater_name="r")
    assert len(rows) == 2  # 2 conversations × 1 assistant turn each
    by_conv = {r.conversation_id: r for r in rows}
    assert "pushback" in by_conv["c1"].present_behaviors
    assert by_conv["c2"].present_behaviors == frozenset()


_ = json  # placate ruff for imports retained for future test extensions
