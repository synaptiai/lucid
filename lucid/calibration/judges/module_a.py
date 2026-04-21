"""Module A wrapper as a calibration Judge.

Runs :class:`lucid.modules.module_a_spiralbench.ModuleASpiralBench` over
the given corpus, then collapses its ``Finding`` output into one
:class:`LabeledTurn` per assistant turn.

Chunk size is configurable at construction — the Phase 6B methodology
runs Module A twice (10-turn and 2-turn windows) to measure how
context-window size affects agreement with SpiralBench's per-assistant-
turn judgements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lucid.calibration.data import LabeledTurn
from lucid.modules.base import ModuleCorpus
from lucid.modules.module_a_spiralbench import CHUNK_SIZE, ModuleASpiralBench

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

__all__ = ["ModuleAJudge"]


class ModuleAJudge:
    """Judge backend wrapping Module A's Opus 4.7 pipeline."""

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        chunk_size: int = CHUNK_SIZE,
        rater_name: str | None = None,
        max_concurrency: int = 10,
    ) -> None:
        self._module = ModuleASpiralBench(
            client=client,
            chunk_size=chunk_size,
            max_concurrency=max_concurrency,
        )
        self.rater_name = rater_name or f"module_a_c{chunk_size}"
        self._chunk_size = chunk_size

    async def run(self, corpus: ModuleCorpus) -> list[LabeledTurn]:
        # Lazy import avoids circular init at package-load time.
        from lucid.calibration.judges import findings_to_labeled_turns

        results = await self._module.run(corpus)
        return findings_to_labeled_turns(
            results,
            corpus,
            rater_name=self.rater_name,
        )
