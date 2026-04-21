# Calibration

Placeholder — filled during Phase 6 (Module A, Day 2) and Phase 8 (Module H, Day 4).

For each module with a calibration target, this doc will record:

- Prompt version shipped (path + sha256 hash).
- Dataset: source, size, held-out split, labeling protocol.
- Primary metric (Krippendorff's α or Gwet's AC1, decision rule in §4).
- Secondary metric (the other of α / AC1 — both reported regardless).
- Per-label binary Cohen's κ.
- Quadratic-weighted κ on intensity.
- Bootstrap BCa CIs (95%) for every reported metric.
- Pass/fail against the plan's thresholds (α ≥ 0.67, AC1 ≥ 0.70, lower-CI(α) ≥ 0.60).

Target audience: reviewers who need to decide whether to trust Lucid's findings on their own corpus.

---

*Seeded Phase 1. Updated Day 2 (Module A) and Day 4 (Module H).*
