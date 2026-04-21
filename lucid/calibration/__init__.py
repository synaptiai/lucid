"""Calibration harness for modules with ground-truth targets.

Two modules have calibration targets in the hackathon scope:

- Module A (SpiralBench): Krippendorff α ≥ 0.67 and Gwet AC1 ≥ 0.70.
- Module H (Memory): manual validation on a seeded corpus (Phase 8).

Dependency note: ``irrCAC`` is *not* in the locked dep set — its hard-pinned
``numpy==1.26.4`` conflicts with ``voyageai==0.3.7`` (needs numpy≥2.1). Gwet
AC1 and Cohen's κ are hand-rolled in :mod:`lucid.calibration.validate` and
validated against Gwet 2014 worked examples.
"""
