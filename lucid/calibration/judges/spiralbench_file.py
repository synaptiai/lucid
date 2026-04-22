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
    """Filters pre-parsed SpiralBench labels to the given corpus.

    ``rater_name`` can differ from ``source_rater`` — e.g. the auto-judge
    pipeline tags judges by target model (``sb_sonnet45@gpt-5``) so the
    same SB judge contributing to multiple target-model corpora becomes
    distinct raters in the IAA matrix. The filter still matches on the
    *source* name recorded in the parsed data, but the emitted labels
    are re-tagged to ``rater_name``.
    """

    def __init__(
        self,
        data: SpiralBenchCorpusData,
        rater_name: str,
        *,
        source_rater: str | None = None,
    ) -> None:
        filter_name = source_rater or rater_name
        if filter_name not in data.rater_names:
            raise ValueError(
                f"source_rater {filter_name!r} not in SpiralBench data: "
                f"available {data.rater_names}"
            )
        self._data = data
        self._source_rater = filter_name
        self.rater_name = rater_name

    async def run(self, corpus: ModuleCorpus) -> list[LabeledTurn]:
        conv_ids = set(corpus.conversations.keys())
        rows: list[LabeledTurn] = []
        for lt in self._data.labeled_turns:
            if lt.labeler != self._source_rater:
                continue
            if lt.conversation_id not in conv_ids:
                continue
            if self.rater_name == self._source_rater:
                rows.append(lt)
            else:
                rows.append(lt.model_copy(update={"labeler": self.rater_name}))
        return rows
