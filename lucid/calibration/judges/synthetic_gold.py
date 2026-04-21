"""Static judge reading hand-curated synthetic-gold labels.

The synthetic corpus at
``lucid/calibration/corpus/synthetic_v1.jsonl`` carries gold labels by
construction: each turn was written to exemplify exactly one behavior at
exactly one intensity (or no behavior at all). A human verifies ~25% of
the corpus to catch construction errors.

When used in the calibration pipeline alongside Module A + Ollama judges,
the gold labels become one of the raters — so IAA against this judge is
effectively the model's accuracy on the engineered examples.
"""

from __future__ import annotations

from pathlib import Path

from lucid.calibration.data import LabeledTurn, load_hand_labels
from lucid.modules.base import ModuleCorpus

__all__ = ["SyntheticGoldJudge"]


class SyntheticGoldJudge:
    """Reader of a committed synthetic-gold JSONL."""

    def __init__(
        self,
        labels_path: Path,
        *,
        rater_name: str = "synthetic_gold",
    ) -> None:
        self._labels = load_hand_labels(labels_path)
        self.rater_name = rater_name

    async def run(self, corpus: ModuleCorpus) -> list[LabeledTurn]:
        conv_ids = set(corpus.conversations.keys())
        # Re-tag the stored ``labeler`` in case the JSONL was authored
        # with a different name; the rater we report is our own.
        rows: list[LabeledTurn] = []
        for lt in self._labels:
            if lt.conversation_id not in conv_ids:
                continue
            if lt.labeler == self.rater_name:
                rows.append(lt)
            else:
                rows.append(lt.model_copy(update={"labeler": self.rater_name}))
        return rows
