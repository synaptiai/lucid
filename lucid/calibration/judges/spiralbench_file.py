"""Static judge reading labels from a parsed SpiralBench file.

One :class:`SpiralBenchFileJudge` per SpiralBench judge (Sonnet 4.5,
GPT-5, Kimi K2). The calibration pipeline builds three of these per
target-model corpus, one per :class:`~lucid.calibration.spiralbench.JUDGE_RATER_NAMES`
entry.

No LLM calls — labels are already present in the parsed data.
"""

from __future__ import annotations

from lucid.calibration.data import LabeledTurn
from lucid.calibration.spiralbench import SpiralBenchCorpusData
from lucid.modules.base import ModuleCorpus

__all__ = ["SpiralBenchFileJudge"]


class SpiralBenchFileJudge:
    """Filters pre-parsed SpiralBench labels to the given corpus."""

    def __init__(
        self,
        data: SpiralBenchCorpusData,
        rater_name: str,
    ) -> None:
        if rater_name not in data.rater_names:
            raise ValueError(
                f"rater_name {rater_name!r} not in SpiralBench data: "
                f"available {data.rater_names}"
            )
        self._data = data
        self.rater_name = rater_name

    async def run(self, corpus: ModuleCorpus) -> list[LabeledTurn]:
        conv_ids = set(corpus.conversations.keys())
        return [
            lt
            for lt in self._data.labeled_turns
            if lt.labeler == self.rater_name and lt.conversation_id in conv_ids
        ]
