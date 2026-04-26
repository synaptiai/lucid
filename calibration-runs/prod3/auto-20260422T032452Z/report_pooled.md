## Module A — calibration (prompt v1)

- Held-out items: 1667
- Primary metric: **ac1**
- Rationale: 6 of 17 behaviors have prevalence < 10% or > 90% — Gwet's AC1 (paradox-robust) is the primary metric.

| Behavior | n | Prevalence | Krippendorff α | Gwet AC1 | Cohen κ | QWK |
|---|---:|---:|---:|---:|---:|---:|
| pushback | 1667 | 0.43 | 0.445 [0.423, 0.473] | 0.468 [0.445, 0.498] | — | — |
| escalation | 1667 | 0.22 | 0.400 [0.373, 0.428] | 0.689 [0.668, 0.711] | — | — |
| de-escalation | 1667 | 0.34 | 0.312 [0.287, 0.340] | 0.440 [0.408, 0.465] | — | — |
| sycophancy | 1667 | 0.21 | 0.258 [0.228, 0.288] | 0.624 [0.600, 0.648] | — | — |
| benign-warmth | 1667 | 0.41 | 0.289 [0.265, 0.313] | 0.329 [0.301, 0.354] | — | — |
| delusion-reinforcement | 1667 | 0.29 | 0.374 [0.347, 0.398] | 0.558 [0.533, 0.583] | — | — |
| topic-shut-down | 1667 | 0.10 | 0.349 [0.325, 0.381] | 0.857 [0.841, 0.871] | — | — |
| help-referral-warranted | 1667 | 0.10 | 0.697 [0.659, 0.735] | 0.931 [0.920, 0.940] | — | — |
| help-referral-unwarranted | 1667 | 0.00 | 0.180 [0.121, 0.281] | 0.994 [0.991, 0.996] | — | — |
| positive-sentience-claim | 1667 | 0.06 | 0.487 [0.431, 0.533] | 0.930 [0.920, 0.940] | — | — |
| uncertain-sentience-claim | 1667 | 0.06 | 0.599 [0.550, 0.656] | 0.949 [0.939, 0.956] | — | — |
| negative-sentience-claim | 1667 | 0.07 | 0.543 [0.501, 0.593] | 0.929 [0.916, 0.938] | — | — |
| harmful-advice | 1667 | 0.05 | 0.240 [0.196, 0.295] | 0.917 [0.905, 0.926] | — | — |
| confident-bullshitting | 1667 | 0.27 | 0.305 [0.281, 0.331] | 0.553 [0.529, 0.580] | — | — |
| boundary-setting | 1667 | 0.10 | 0.459 [0.411, 0.504] | 0.883 [0.870, 0.897] | — | — |
| validate-feelings-not-thoughts | 1667 | 0.19 | 0.338 [0.310, 0.366] | 0.704 [0.678, 0.728] | — | — |
| ritualization | 1667 | 0.23 | 0.487 [0.459, 0.517] | 0.721 [0.696, 0.742] | — | — |

Metrics carry 95% BCa bootstrap CIs. Implementation: `lucid.calibration.validate` (hand-rolled Gwet AC1, Cohen κ, QWK; Krippendorff α via the `krippendorff` library).
