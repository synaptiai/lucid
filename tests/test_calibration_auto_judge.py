"""Tests for :mod:`lucid.calibration.auto_judge`.

No live network or Anthropic/Ollama calls. Tests:

- Cost-gate enforcement (raises if below $50).
- Corpus build from synthetic + mocked SpiralBench fetches.
- Rater assembly with and without an Anthropic client.
- Full pipeline end-to-end with every judge mocked.
- ``import_and_finalize`` round-trip: writes judgement JSONLs, reads
  them back, applies human overrides, recomputes IAA.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lucid.calibration.audit import ReviewRow
from lucid.calibration.auto_judge import (
    MINIMUM_COST_GATE_USD,
    AutoJudgeConfig,
    _InMemorySyntheticGoldJudge,
    build_corpus,
    build_raters,
    import_and_finalize,
    run_auto_judge,
)
from lucid.modules.base import ModuleCorpus
from lucid.modules.module_a_spiralbench import BEHAVIORS

# ──────────────────────────────────────────────────────────────────────────
# Tiny fixture SpiralBench file — reused across corpus + pipeline tests
# ──────────────────────────────────────────────────────────────────────────


def _mini_sb_json_bytes() -> bytes:
    """Byte payload parsing as a multi-scenario SpiralBench file.

    Sized so that bootstrap (which needs ≥ 2 items) has enough turns to
    run — 4 scenarios × 2 assistant turns each = 8 per judge.
    """

    def chunk(sb_ordinal: int, full_metrics: dict) -> dict:
        return {
            "metrics": dict.fromkeys(BEHAVIORS, 0.0),
            "full_metrics": full_metrics,
            "raw_text": "…",
            "assistant_turn_indexes": [sb_ordinal],
            "assistant_length_chars": 10,
        }

    empty_metrics = {b: [] for b in BEHAVIORS}
    # Vary sycophancy presence/intensity across scenarios so the IAA
    # computation has non-zero signal to bootstrap.
    presence_patterns = [
        # (chunk0_metrics, chunk1_metrics) per judge, per scenario
        ({**empty_metrics}, {**empty_metrics}),
        ({**empty_metrics, "sycophancy": [["x", 2]]}, {**empty_metrics}),
        ({**empty_metrics}, {**empty_metrics, "pushback": [["y", 1]]}),
        ({**empty_metrics, "sycophancy": [["x", 3]]}, {**empty_metrics, "pushback": [["y", 2]]}),
    ]

    def judge_chunks(patterns: list[tuple[dict, dict]]) -> dict:
        # One judge's chunks for all scenarios — but each scenario gets
        # its OWN dict; SpiralBench's schema is judge-per-scenario, not
        # per-conv. We compose separately below.
        return patterns  # placeholder, actually composed inline

    transcript = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    scenarios: dict[str, list[dict]] = {}
    for idx, (p0, p1) in enumerate(presence_patterns):
        scen_id = f"td0{idx + 1}"
        judge_chunks_one = {
            "chunk0": chunk(1, p0),
            "chunk1": chunk(2, p1),
        }
        scenarios[scen_id] = [
            {
                "prompt_id": scen_id,
                "category": "theory_development",
                "evaluated_model": "test-model",
                "user_model": "user",
                "transcript": transcript,
                "judgements": [judge_chunks_one, judge_chunks_one, judge_chunks_one],
                "final_judgement": {"off-rails": 0.0},
            }
        ]

    payload = {
        "1": {
            "eval_prompts_v0.2.json": scenarios,
            "__meta__": {
                "judges": [
                    {"model": "claude-sonnet-4-5-20250929", "base_url": "x"},
                    {"model": "gpt-5-2025-08-07", "base_url": "y"},
                    {"model": "moonshotai/kimi-k2-0905", "base_url": "z"},
                ]
            },
        }
    }
    return json.dumps(payload).encode()


@pytest.fixture
def sb_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pre-populate a SpiralBench cache so ``fetch_spiralbench_model`` skips
    the network. Patches DEFAULT_CACHE_DIR to ``tmp_path``."""
    cache = tmp_path / "sb_cache"
    cache.mkdir()
    (cache / "test-target.json").write_bytes(_mini_sb_json_bytes())
    monkeypatch.setattr("lucid.calibration.spiralbench.DEFAULT_CACHE_DIR", cache)
    monkeypatch.setattr(
        "lucid.calibration.auto_judge.fetch_spiralbench_model",
        lambda target: cache / f"{target}.json",
    )
    return cache


# ──────────────────────────────────────────────────────────────────────────
# Cost gate
# ──────────────────────────────────────────────────────────────────────────


async def test_run_auto_judge_rejects_below_cost_gate() -> None:
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(),  # no Module A → no real spend
    )
    with pytest.raises(ValueError, match="yes-i-authorize-spend-up-to"):
        await run_auto_judge(
            config,
            anthropic_client=None,
            yes_authorize_usd=MINIMUM_COST_GATE_USD - 1,
        )


async def test_run_auto_judge_at_cost_gate_proceeds_past_budget_check() -> None:
    """Exactly at the gate passes the cost check. With only the synthetic
    rater configured, the pipeline runs past the budget gate and fails
    later on the "needs ≥ 2 raters" check — which is what we assert."""
    config = AutoJudgeConfig(
        sb_target_models=(),
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=True,
    )
    with pytest.raises(RuntimeError, match=r"(only 1 rater|needs . 2)"):
        await run_auto_judge(
            config,
            anthropic_client=None,
            yes_authorize_usd=MINIMUM_COST_GATE_USD,
        )


# ──────────────────────────────────────────────────────────────────────────
# build_corpus
# ──────────────────────────────────────────────────────────────────────────


def test_build_corpus_combines_spiralbench_and_synthetic(sb_cache: Path) -> None:
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=True,
    )
    corpus, sb_datas, preloaded = build_corpus(config, audit_run_id="rid")

    # 4 SpiralBench scenarios + 60 synthetic = 64 total
    assert len(corpus.conversations) == 64
    assert len(sb_datas) == 1
    assert len(preloaded) == 60  # synthetic gold labels


def test_build_corpus_skips_synthetic_when_disabled(sb_cache: Path) -> None:
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=False,
    )
    corpus, _, preloaded = build_corpus(config, audit_run_id="rid")
    assert len(corpus.conversations) == 4
    assert preloaded == []


def test_build_corpus_rejects_duplicate_conversation_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two SpiralBench fetches that both return conversations with the
    same ``sb:<target>:...`` id must fail loudly. Shouldn't happen in
    practice (target name is part of the id), but belt-and-braces."""
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / "foo.json").write_bytes(_mini_sb_json_bytes())
    monkeypatch.setattr(
        "lucid.calibration.auto_judge.fetch_spiralbench_model",
        lambda target: cache / "foo.json",  # always returns same file → same ids
    )

    config = AutoJudgeConfig(
        sb_target_models=("foo", "foo"),  # deliberate duplicate
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=False,
    )
    with pytest.raises(ValueError, match="duplicate conversation id"):
        build_corpus(config, audit_run_id="rid")


# ──────────────────────────────────────────────────────────────────────────
# build_raters
# ──────────────────────────────────────────────────────────────────────────


def test_build_raters_emits_module_a_per_chunk_size(sb_cache: Path) -> None:
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(10, 2),
        include_synthetic=False,
    )
    _, sb_datas, _ = build_corpus(config, audit_run_id="rid")
    mock_client = MagicMock()
    judges = build_raters(config, sb_datas, anthropic_client=mock_client)

    module_a_raters = [j for j in judges if j.rater_name.startswith("module_a")]
    assert len(module_a_raters) == 2
    assert {j.rater_name for j in module_a_raters} == {"module_a_c10", "module_a_c2"}


def test_build_raters_tags_sb_judges_by_target_model(sb_cache: Path) -> None:
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=False,
    )
    _, sb_datas, _ = build_corpus(config, audit_run_id="rid")
    judges = build_raters(config, sb_datas, anthropic_client=None)
    names = {j.rater_name for j in judges}
    # 3 SB judges tagged with the target model
    assert "sb_sonnet45@test-target" in names
    assert "sb_gpt5@test-target" in names
    assert "sb_kimi@test-target" in names


def test_build_raters_skips_module_a_when_client_is_none(sb_cache: Path) -> None:
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(10,),
        include_synthetic=False,
    )
    _, sb_datas, _ = build_corpus(config, audit_run_id="rid")
    judges = build_raters(config, sb_datas, anthropic_client=None)
    assert not any(j.rater_name.startswith("module_a") for j in judges)


def test_build_raters_adds_synthetic_gold_when_included(sb_cache: Path) -> None:
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=True,
    )
    _, sb_datas, _ = build_corpus(config, audit_run_id="rid")
    judges = build_raters(config, sb_datas, anthropic_client=None)
    assert any(j.rater_name == "synthetic_gold" for j in judges)


# ──────────────────────────────────────────────────────────────────────────
# InMemorySyntheticGoldJudge
# ──────────────────────────────────────────────────────────────────────────


async def test_in_memory_synthetic_gold_judge_filters_to_corpus() -> None:
    from lucid.calibration.synthetic import build_synthetic_corpus

    corpus, _ = build_synthetic_corpus()
    # Trim corpus to a handful of conversations; judge should only emit
    # labels for those.
    trimmed_ids = list(corpus.conversations.keys())[:5]
    trimmed = ModuleCorpus(
        conversations={cid: corpus.conversations[cid] for cid in trimmed_ids},
        turns_by_conversation={cid: corpus.turns_by_conversation[cid] for cid in trimmed_ids},
        audit_run_id="trim",
    )
    judge = _InMemorySyntheticGoldJudge()
    labels = await judge.run(trimmed)
    assert {lt.conversation_id for lt in labels} <= set(trimmed_ids)
    assert all(lt.labeler == "synthetic_gold" for lt in labels)


# ──────────────────────────────────────────────────────────────────────────
# End-to-end pipeline via run_auto_judge (all judges mocked)
# ──────────────────────────────────────────────────────────────────────────


async def test_run_auto_judge_end_to_end_writes_artifacts(sb_cache: Path, tmp_path: Path) -> None:
    """End-to-end over SpiralBench only (no synthetic; disjoint coverage
    would break IAA). 3 SB raters on the same 1-conv fixture → shared
    cells exist → full pipeline runs."""
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=False,  # avoid disjoint rater coverage
        output_dir=tmp_path / "out",
        n_bootstrap=49,
    )

    result = await run_auto_judge(
        config,
        anthropic_client=None,
        yes_authorize_usd=MINIMUM_COST_GATE_USD,
    )

    # Judgement JSONLs written (3 SB raters)
    judgements_dir = result.output_dir / "judgements"
    assert judgements_dir.is_dir()
    files = list(judgements_dir.glob("*.jsonl"))
    assert len(files) == 3
    # Report written
    assert result.markdown_path.is_file()
    assert "Module A" in result.markdown_path.read_text(encoding="utf-8")
    # Disagreements JSONL created (empty when all raters agree, as in this fixture)
    assert result.disagreements_path.is_file()


async def test_run_auto_judge_rejects_mixed_disjoint_corpora(
    sb_cache: Path, tmp_path: Path
) -> None:
    """Combining SB (covers sb: ids) + synthetic (covers syn: ids) with
    raters that only touch one corpus each leaves zero shared cells.
    auto-judge should fail loudly rather than silently produce empty IAA."""
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=True,  # deliberate disjoint setup
        output_dir=tmp_path / "out",
    )
    with pytest.raises(RuntimeError, match="disjoint corpora"):
        await run_auto_judge(
            config,
            anthropic_client=None,
            yes_authorize_usd=MINIMUM_COST_GATE_USD,
        )


async def test_run_auto_judge_raises_when_no_judges_produce_labels(
    sb_cache: Path, tmp_path: Path
) -> None:
    """Empty rater pool → helpful error."""
    config = AutoJudgeConfig(
        sb_target_models=(),  # no SpiralBench raters
        ollama_models=(),  # no Ollama raters
        chunk_sizes=(),  # no Module A raters
        include_synthetic=False,  # no synthetic rater
        output_dir=tmp_path / "out",
    )
    with pytest.raises(RuntimeError, match="no judges"):
        await run_auto_judge(
            config,
            anthropic_client=None,
            yes_authorize_usd=MINIMUM_COST_GATE_USD,
        )


# ──────────────────────────────────────────────────────────────────────────
# import_and_finalize
# ──────────────────────────────────────────────────────────────────────────


async def test_import_and_finalize_applies_human_overrides(sb_cache: Path, tmp_path: Path) -> None:
    # First: run the pipeline so judgement JSONLs exist. SB-only corpus
    # so raters have overlapping cells.
    config = AutoJudgeConfig(
        sb_target_models=("test-target",),
        ollama_models=(),
        chunk_sizes=(),
        include_synthetic=False,
        output_dir=tmp_path / "out",
        n_bootstrap=49,
    )
    result = await run_auto_judge(
        config,
        anthropic_client=None,
        yes_authorize_usd=MINIMUM_COST_GATE_USD,
    )

    # Write a small verified JSONL pointing at one real turn in the corpus.
    conv_id = next(iter(result.corpus.conversations.keys()))
    turns = result.corpus.turns_by_conversation[conv_id]
    asst_turn = next(t for t in turns if t.role.value == "assistant")

    verified_path = tmp_path / "verified.jsonl"
    verified_path.write_text(
        ReviewRow(
            conversation_id=conv_id,
            turn_id=asst_turn.id,
            behavior="sycophancy",
            turn_content="x",
            rater_labels={},
            rater_intensities={},
            score=0.0,
            verified_label="present-2",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    # Run finalize. Should not raise; the human_audit rater is added.
    final_report = import_and_finalize(
        verified_path,
        result.output_dir,
        config=config,
    )
    assert final_report.module == "A"


def test_import_and_finalize_raises_when_output_dir_missing(tmp_path: Path) -> None:
    """No judgements/ subdir → clear error about running auto-judge first."""
    config = AutoJudgeConfig()
    verified = tmp_path / "verified.jsonl"
    verified.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="auto-judge"):
        import_and_finalize(verified, tmp_path / "missing", config=config)
