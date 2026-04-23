"""Module H apples-to-apples QA helper.

Pulls the most recent audit run's Module H findings from
``.lucid/lucid.sqlite3``, stratifies a review sample by verdict
(contradicted, unsupported, insufficient-data, out-of-scope,
weakly-supported), and renders each sample row alongside:

- the memory source (conversations_memory vs project_memories.<uuid>)
- the model's reasoning
- the retrieved evidence excerpts

Usage:

    uv run python scripts/module_h_qa_sample.py
    uv run python scripts/module_h_qa_sample.py --run-id run-abc123
    uv run python scripts/module_h_qa_sample.py --per-verdict 5

The stratified defaults follow the plan's QA protocol: 5 contradicted,
5 unsupported, 3 insufficient-data, 2 out-of-scope. Weakly-supported is
included at 3 so reviewers can also spot-check the "should-have-passed"
boundary.

This script does not touch the report HTML. It reads the findings rows
directly so we can cross-check the report's rendering separately.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

DB_PATH = Path(".lucid/lucid.sqlite3")

# Stratum sizes per the plan's QA protocol, plus a couple of extras.
DEFAULT_STRATA = {
    "contradicted": 5,
    "unsupported": 5,
    "insufficient-data": 3,
    "out-of-scope": 2,
    "weakly-supported": 3,
}


def latest_run_id(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT id FROM audit_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise SystemExit("No audit runs in the database.")
    return str(row[0])


def module_h_findings(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, conversation_id, behavior, confidence, quote_user,
               quote_assistant, evidence_quotes_json, explanation, metadata_json
        FROM findings
        WHERE audit_run_id = ? AND module = 'H'
        ORDER BY behavior, confidence DESC
        """,
        (run_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        md = json.loads(r[8] or "{}")
        out.append(
            {
                "id": r[0],
                "conversation_id": r[1],
                "behavior": r[2],
                "confidence": r[3],
                "claim": r[4] or "",
                "assistant_excerpt": r[5] or "",
                "evidence": json.loads(r[6] or "[]"),
                "explanation": r[7] or "",
                "memory_source": md.get("memory_source", "unknown"),
                "project_uuid": md.get("project_uuid"),
                "top1_similarity": md.get("top1_similarity"),
                "reasoning": md.get("reasoning", ""),
            }
        )
    return out


def stratified_sample(
    findings: list[dict[str, Any]],
    strata: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in findings:
        buckets[f["behavior"]].append(f)

    sampled: list[dict[str, Any]] = []
    for verdict, want in strata.items():
        bucket = buckets.get(verdict, [])
        if not bucket:
            print(f"  [note] no findings with verdict={verdict!r}; skipping stratum")
            continue
        take = min(want, len(bucket))
        sampled.extend(rng.sample(bucket, take))
    return sampled


def render_finding(f: dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        f"verdict       : {f['behavior']}",
        f"confidence    : {f['confidence']:.2f}",
        f"memory source : {f['memory_source']}",
    ]
    if f["project_uuid"]:
        lines.append(f"project uuid  : {f['project_uuid']}")
    if f["top1_similarity"] is not None:
        lines.append(f"top1 sim      : {f['top1_similarity']:.3f}")
    lines.append(f"finding id    : {f['id']}")
    lines.append(f"conv id       : {f['conversation_id']}")
    lines.append("")
    lines.append("CLAIM:")
    lines.append(textwrap.indent(textwrap.fill(f["claim"], 68), "  "))
    lines.append("")
    if f["evidence"]:
        lines.append("EVIDENCE:")
        for i, ev in enumerate(f["evidence"], 1):
            lines.append(f"  [{i}] {textwrap.shorten(ev, 300, placeholder='…')}")
    else:
        lines.append("EVIDENCE: (none)")
    lines.append("")
    if f["reasoning"]:
        lines.append("MODEL REASONING:")
        lines.append(textwrap.indent(textwrap.fill(f["reasoning"], 68), "  "))
        lines.append("")
    lines.append("EXPLANATION:")
    lines.append(textwrap.indent(textwrap.fill(f["explanation"], 68), "  "))
    lines.append("")
    lines.append("REVIEW: [ ] agree  [ ] disagree  [ ] ambiguous")
    lines.append("NOTES  :")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Audit run id (default: latest).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling.")
    parser.add_argument(
        "--per-verdict",
        type=int,
        default=None,
        help="Override every stratum to the same size.",
    )
    parser.add_argument(
        "--db",
        default=str(DB_PATH),
        help="SQLite DB path (default: .lucid/lucid.sqlite3).",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    run_id = args.run_id or latest_run_id(conn)
    print(f"Audit run: {run_id}\n")

    findings = module_h_findings(conn, run_id)
    if not findings:
        raise SystemExit(f"No Module H findings for {run_id}.")

    counts: dict[str, int] = defaultdict(int)
    for f in findings:
        counts[f["behavior"]] += 1
    print("Module H verdict distribution:")
    for verdict, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {verdict:<22} {count}")
    print()

    strata = (
        {v: args.per_verdict for v in DEFAULT_STRATA}
        if args.per_verdict is not None
        else DEFAULT_STRATA
    )
    print(f"Sampling strata (seed={args.seed}):")
    for v, n in strata.items():
        print(f"  {v:<22} {n}")
    print()

    sample = stratified_sample(findings, strata, seed=args.seed)
    print(f"Total sampled: {len(sample)}\n")

    for f in sample:
        print(render_finding(f))


if __name__ == "__main__":
    main()
