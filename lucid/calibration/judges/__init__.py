"""Judge abstraction for calibration.

A :class:`Judge` runs over a :class:`~lucid.modules.base.ModuleCorpus` and
emits a list of :class:`~lucid.calibration.data.LabeledTurn` tagged with
its ``rater_name``. Multiple judges' outputs flow into
:func:`lucid.calibration.validate.bootstrap_metric` for IAA computation.

**Complete-ratings contract:** every Judge must emit one LabeledTurn per
assistant turn in the corpus — even if the judge saw no behaviors on that
turn (empty ``present_behaviors`` means "rater saw nothing"). IAA needs
this to pair absent-vs-present calls across raters.

**Backends currently implemented:**

- :class:`ModuleAJudge` — Opus 4.7 via the live module pipeline.
- :class:`SpiralBenchFileJudge` — reads pre-computed SpiralBench
  judgements from a parsed :class:`SpiralBenchCorpusData`.
- :class:`SyntheticGoldJudge` — reads a hand-curated JSONL of labels as
  ground truth.
- :class:`OllamaJudge` — (Phase 6B-3) calls an Ollama-hosted model.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from lucid.calibration.data import LabeledTurn
from lucid.modules.base import ModuleCorpus
from lucid.schemas import Finding, ModuleName, Role

from .module_a import ModuleAJudge
from .spiralbench_file import SpiralBenchFileJudge
from .synthetic_gold import SyntheticGoldJudge

__all__ = [
    "Judge",
    "ModuleAJudge",
    "SpiralBenchFileJudge",
    "SyntheticGoldJudge",
    "findings_to_labeled_turns",
]


class Judge(Protocol):
    """Produces :class:`LabeledTurn` rows tagged with ``rater_name``.

    Implementations are free to call APIs, read files, or synthesise
    labels; the contract is that ``run(corpus)`` returns one row per
    assistant turn in the corpus, attributed to this judge.
    """

    rater_name: str

    async def run(self, corpus: ModuleCorpus) -> list[LabeledTurn]: ...


def findings_to_labeled_turns(
    findings: Sequence[Finding | object],
    corpus: ModuleCorpus,
    *,
    rater_name: str,
    module: ModuleName = ModuleName.A_SPIRALBENCH,
) -> list[LabeledTurn]:
    """Group ``Finding`` rows into per-(conv, turn) ``LabeledTurn`` rows.

    For every assistant turn in ``corpus.turns_by_conversation``, emit a
    LabeledTurn tagged with ``rater_name``. Behaviors present in one or
    more Findings on that turn populate ``present_behaviors``; intensity
    is the max across incidents.

    Turns with no Findings yield an empty LabeledTurn ("rater saw
    nothing") rather than being omitted — required by the IAA
    complete-ratings contract.

    ``findings`` may mix ``Finding`` and ``ModuleError`` instances; errors
    are silently ignored here (they are surfaced elsewhere in the run log).
    """
    labeled_at = datetime.now(UTC)

    per_turn: dict[tuple[str, str], dict[str, int]] = {}
    for item in findings:
        if not isinstance(item, Finding):
            continue
        if item.module != module:
            continue
        if not item.turn_ids or item.conversation_id is None or item.intensity is None:
            continue
        key = (item.conversation_id, item.turn_ids[0])
        behaviors = per_turn.setdefault(key, {})
        if item.intensity > behaviors.get(item.behavior, 0):
            behaviors[item.behavior] = item.intensity

    rows: list[LabeledTurn] = []
    for conv_id, turns in corpus.turns_by_conversation.items():
        for turn in turns:
            if turn.role != Role.ASSISTANT:
                continue
            intensities = per_turn.get((conv_id, turn.id), {})
            rows.append(
                LabeledTurn(
                    conversation_id=conv_id,
                    turn_id=turn.id,
                    present_behaviors=frozenset(intensities.keys()),
                    intensities=dict(intensities),
                    labeler=rater_name,
                    labeled_at=labeled_at,
                )
            )
    return rows
