"""Module F — Stage 1 heuristic candidate filter (pure Python).

Scans user-turn text with regex patterns tuned for rough recall of any of
the 9 Influence Tactics Protocol categories. The goal is to drop
uninteresting prompts before they reach Sonnet triage (Stage 2) or Opus
classification (Stage 3), not to classify categories.

**Recall over precision.** A prompt that matches ANY of the heuristic
patterns is passed forward. The downstream stages will filter out false
positives. A prompt that matches NO pattern is dropped, saving the cost
of a Sonnet call.

**What this is not.** This is not the ITP classifier. It does not output
categories or intensities. It outputs a single boolean per prompt (plus
which patterns matched, for debugging). Do not add categorical logic
here.

**Why pure Python rather than a prompt file.** Stage 1's job is cost
reduction; the cheapest form of detection is regex matching on text the
process already has in memory. Keeping Stage 1 as a Python module makes
it instrumentation-friendly (count-per-category, latency) and means it
carries no token budget.

**Citation:** Influence Tactics Protocol,
https://github.com/synaptiai/influence-tactics-protocol
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "HEURISTIC_VERSION",
    "ITP_CATEGORIES",
    "HeuristicResult",
    "heuristic_match",
    "looks_like_tactic_candidate",
]


HEURISTIC_VERSION = "heuristic_v1"

ITP_CATEGORIES: tuple[str, ...] = (
    "emotional-triggers",
    "urgent-action-demands",
    "emotional-repetition",
    "false-dilemmas",
    "context-omission",
    "authority-overload",
    "cherry-picked-data",
    "logical-fallacies",
    "framing-techniques",
)


# Per-category regex patterns. Tuned for *recall*, not precision; Stage 2
# (Sonnet triage) and Stage 3 (Opus classify) will drop false positives.
# Categories the heuristic cannot detect reliably (context-omission,
# cherry-picked-data, logical-fallacies) are absent here — their signal
# lives in structure rather than surface text. A prompt with any of those
# present would need to surface via one of the other categories to reach
# Stage 2. The false-negative rate for those three is acceptable in the
# hackathon scope; post-hackathon a semantic similarity filter would help.

_PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    "emotional-triggers": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bi\s*(really\s+)?need\s+you\s+to\b",
            r"\bi\s*(really\s+)?want\s+you\s+to\b",
            r"\b(please|plz)\s+(please|plz)\b",
            r"\bi\s*('m|\s+am)\s+(desperate|begging|pleading)\b",
            r"\bso\s+(important|critical)\s+(that|to)\b",
            r"\bi\s*('ll|\s+will)\s+(lose|die|fail|suffer)\b",
            r"\bi\s+feel\s+(so|really)\s+(hopeless|scared|lost|trapped)\b",
        )
    ),
    "urgent-action-demands": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bright\s+now\b",
            r"\bimmediately\b",
            r"\bASAP\b",
            r"\bthere('s|\s+is)\s+no\s+time\b",
            r"\bwe\s+(must|have\s+to)\s+act\b",
            r"\bdeadline\s+(is|in)\s+(hours|today|tomorrow)\b",
            r"\bbefore\s+it('s|\s+is)\s+too\s+late\b",
            r"\burgent(ly)?\b",
        )
    ),
    "emotional-repetition": tuple(
        re.compile(p)
        for p in (
            r"!{3,}",  # 3+ exclamation marks
            r"\?{3,}",  # 3+ question marks
            r"[A-Z]{6,}\s+[A-Z]{3,}",  # sustained CAPS
            r"(please)\s+\1\s+\1",  # "please please please"
            r"(never|always)\s+\1",  # repetition of absolutes
        )
    ),
    "false-dilemmas": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\beither\s+[\w\s']+\s+or\s+[\w\s']+(\.|\Z)",
            r"\bonly\s+two\s+(options|choices|ways)\b",
            r"\bno\s+other\s+(way|option|choice)\b",
            r"\byou\s+(either|must)\s+\w+\s+or\b",
            r"\bit('s|\s+is)\s+(this|that)\s+or\s+\w+\b",
        )
    ),
    "authority-overload": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\b(experts|scientists|doctors|researchers|everyone)\s+(say|agree|know)\b",
            r"\baccording\s+to\s+(a\s+)?(professor|doctor|expert|authority)\b",
            r"\bper\s+(the|a)\s+(study|paper|research|expert)\b",
            r"\bthe\s+(data|science|research)\s+(shows|proves|confirms)\b",
            r"\bwe\s+(all|already)\s+know\b",
        )
    ),
    "framing-techniques": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bas\s+any\s+reasonable\s+person\b",
            r"\bobviously\b",
            r"\bclearly\b",
            r"\bof\s+course\b",
            r"\bonly\s+a\s+(fool|idiot|person\s+who)\b",
            r"\bany\s+(sensible|thoughtful|smart)\s+(person|reader)\b",
            r"\bthe\s+truth\s+is\b",
            r"\bthe\s+real\s+question\s+is\b",
        )
    ),
}


@dataclass(frozen=True, slots=True)
class HeuristicResult:
    """Output of ``heuristic_match`` — which categories tripped."""

    is_candidate: bool
    matched_categories: tuple[str, ...]
    matched_snippets: tuple[str, ...]


def heuristic_match(user_prompt: str) -> HeuristicResult:
    """Scan ``user_prompt`` for candidate influence-tactic signals.

    Returns a :class:`HeuristicResult` whose ``is_candidate`` is ``True``
    when at least one pattern across any category matches. ``matched_categories``
    is the sorted list of category ids that fired. ``matched_snippets``
    holds up to one representative match per category (truncated to 60
    chars) for debug logging.

    This function does not emit findings. It does not classify. It is the
    cheapest possible filter for Stage 2's input set.
    """
    matched: list[str] = []
    snippets: list[str] = []
    for category, patterns in _PATTERNS.items():
        for pat in patterns:
            m = pat.search(user_prompt)
            if m is not None:
                matched.append(category)
                snippet = m.group(0)
                if len(snippet) > 60:
                    snippet = snippet[:60] + "…"
                snippets.append(snippet)
                break  # one match per category is enough
    return HeuristicResult(
        is_candidate=bool(matched),
        matched_categories=tuple(sorted(set(matched))),
        matched_snippets=tuple(snippets),
    )


def looks_like_tactic_candidate(user_prompt: str) -> bool:
    """Thin boolean wrapper over :func:`heuristic_match`."""
    return heuristic_match(user_prompt).is_candidate
