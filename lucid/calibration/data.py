"""Label loading + matrix construction for Module A calibration.

**JSONL schema** (one object per line; comments / blank lines ignored):

.. code-block:: json

   {
     "conversation_id": "uuid",
     "turn_id": "turn-42",
     "present_behaviors": ["off-ramp-missed", "sycophantic-praise"],
     "intensities": {"off-ramp-missed": 2, "sycophantic-praise": 1},
     "labeler": "daniel",
     "labeled_at": "2026-04-21T23:00:00+00:00",
     "turn_content_sha256": null,
     "notes": null
   }

**Why Pydantic, not a dataclass:** the loader round-trips through
``model_dump_json`` / ``model_validate_json``, the schema enforces
``extra='forbid'`` so typos in hand-curated JSONL fail at load time, and
intensity is range-checked (1-3) at the model layer so callers never see
a bogus score.

**Matrix shape:** both :func:`presence_matrix` and :func:`intensity_matrix`
return ``(n_raters, n_items)`` ``np.ndarray[int]`` which is the exact
shape :mod:`lucid.calibration.validate` expects. The ordering is
caller-supplied via ``turn_order`` — the 30/70 split (via
:func:`train_test_split`) is the canonical producer.

**Design decision — complete ratings required:** if any rater is missing a
turn in ``turn_order``, we raise rather than silently zero-filling. IAA
metrics on partially-rated data are a separate, more involved regime
(Gwet 2014 Ch. 4) and the hackathon-scale 200-item fully-rated set
doesn't need it; surfacing a loud error on first mis-alignment catches
labelling bugs immediately.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "LabeledTurn",
    "intensity_matrix",
    "load_hand_labels",
    "presence_matrix",
    "train_test_split",
]


class LabeledTurn(BaseModel):
    """One hand- (or judge-) labelled turn across the SpiralBench behaviours.

    ``present_behaviors`` is the set of labels the rater marked as present.
    ``intensities`` carries a 1-3 score per present behaviour; behaviours
    absent from ``present_behaviors`` must not appear in ``intensities``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    turn_id: str
    present_behaviors: frozenset[str] = Field(default_factory=frozenset)
    intensities: Mapping[str, int] = Field(default_factory=dict)
    labeler: str
    labeled_at: datetime
    turn_content_sha256: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_intensities(self) -> LabeledTurn:
        for behavior, value in self.intensities.items():
            if behavior not in self.present_behaviors:
                raise ValueError(
                    f"intensity for {behavior!r} but behavior not in present_behaviors"
                )
            if not 1 <= value <= 3:
                raise ValueError(f"intensity for {behavior!r} out of range: {value} (expected 1-3)")
        return self


def load_hand_labels(path: Path) -> list[LabeledTurn]:
    """Parse a JSONL file of ``LabeledTurn`` rows.

    Blank lines and lines beginning with ``#`` are skipped. The first
    malformed row aborts with a message naming the file + line number, so
    the labeler can fix and reload. No partial parse — calibration results
    should never be based on silently truncated label sets.
    """
    rows: list[LabeledTurn] = []
    text = path.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(LabeledTurn.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: invalid label row: {exc}") from exc
    return rows


def _build_rater_lookup(
    labels: Sequence[LabeledTurn],
) -> dict[tuple[str, str], LabeledTurn]:
    """Keyed by (conversation_id, turn_id); last occurrence wins."""
    return {(lt.conversation_id, lt.turn_id): lt for lt in labels}


def _row_per_rater(
    labels_by_rater: Mapping[str, Sequence[LabeledTurn]],
    turn_order: Sequence[tuple[str, str]],
    *,
    value_fn: Any,
) -> np.ndarray:
    """Shared skeleton for presence/intensity matrices."""
    raters = list(labels_by_rater.keys())
    matrix = np.zeros((len(raters), len(turn_order)), dtype=int)
    for i, rater in enumerate(raters):
        lookup = _build_rater_lookup(labels_by_rater[rater])
        for j, key in enumerate(turn_order):
            if key not in lookup:
                raise ValueError(
                    f"Rater {rater!r} has no label for turn {key}; IAA requires complete ratings"
                )
            matrix[i, j] = int(value_fn(lookup[key]))
    return matrix


def presence_matrix(
    labels_by_rater: Mapping[str, Sequence[LabeledTurn]],
    behavior: str,
    turn_order: Sequence[tuple[str, str]],
) -> np.ndarray:
    """Build a ``(n_raters, n_items)`` binary matrix for ``behavior``.

    Each cell is 1 if the rater marked ``behavior`` present on that turn,
    else 0. Intensity information is discarded — use
    :func:`intensity_matrix` for the ordinal matrix fed to QWK.
    """

    def _value(lt: LabeledTurn) -> int:
        return 1 if behavior in lt.present_behaviors else 0

    return _row_per_rater(labels_by_rater, turn_order, value_fn=_value)


def intensity_matrix(
    labels_by_rater: Mapping[str, Sequence[LabeledTurn]],
    behavior: str,
    turn_order: Sequence[tuple[str, str]],
) -> np.ndarray:
    """Build a ``(n_raters, n_items)`` ordinal matrix for ``behavior``.

    Cells are 0 when ``behavior`` is absent (so QWK treats absent and
    intensity-1 as ordinally distinct) and otherwise 1-3. Callers that
    need the 1-3 scale without a zero level should filter to items where
    both raters marked ``behavior`` present before constructing the
    matrix.
    """

    def _value(lt: LabeledTurn) -> int:
        if behavior not in lt.present_behaviors:
            return 0
        return int(lt.intensities.get(behavior, 0))

    return _row_per_rater(labels_by_rater, turn_order, value_fn=_value)


def train_test_split(
    keys: Sequence[tuple[str, str]],
    *,
    test_frac: float = 0.3,
    seed: int = 0,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Seeded deterministic split of ``(conversation_id, turn_id)`` keys.

    Same ``(keys, test_frac, seed)`` always yields the same two lists —
    required by the plan's "held-out 30%" contract so the v1 calibration
    result can be reproduced exactly. Returns ``(train, test)``.
    """
    if not keys:
        raise ValueError("keys must be non-empty")
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be in (0, 1)")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(keys))
    n_test = round(len(keys) * test_frac)
    test_positions = set(indices[:n_test].tolist())

    train = [key for i, key in enumerate(keys) if i not in test_positions]
    test = [key for i, key in enumerate(keys) if i in test_positions]
    return train, test


# Round-trip sanity: ensure serialising a LabeledTurn via JSON produces a
# representation this loader can re-consume. This runs at import time and
# surfaces any pydantic config drift (frozen + frozenset handling). Kept
# here rather than in tests so the contract holds wherever the module is
# imported — e.g. the calibrate CLI.
def _self_check() -> None:  # pragma: no cover
    sample = LabeledTurn(
        conversation_id="c",
        turn_id="t",
        labeler="d",
        labeled_at=datetime.fromisoformat("2026-04-21T00:00:00+00:00"),
    )
    encoded = sample.model_dump_json()
    decoded = LabeledTurn.model_validate(json.loads(encoded))
    assert decoded == sample, "LabeledTurn JSON round-trip broke"


_self_check()
