---
title: Lucid Stack & Distribution Decision (3 Approaches + Methodology Pivots)
type: decision
status: partial — Stack=A (Python), D2b locked, D-PLUGIN=no (v1), D-MCP=custom tools only
date: 2026-04-21
amended: 2026-04-21 (post-PDF verification + user clarifications)
parent_plan: docs/plans/2026-04-21-feat-lucid-hackathon-build-plan.md
---

## Post-PDF amendment (2026-04-21, later same day)

Official hackathon PDF received and parsed. User clarified: (a) plugin is NOT a v1 goal — misread of my prior phrasing; intent is "people run it on their own computer on their own data" with zero-friction CLI; (b) "researcher ergonomics" needs definition; (c) sample size committed to D2b (200+) — "aiming for winning, don't calibrate down."

**Status of decisions:**
- **Stack: Approach A (Python + compiled binary)** — recommended and unambiguous post-clarification.
- **D-PLUGIN: No plugin in v1. Distribution = one binary, one command.** Plugin/MCP-server publishing = post-hackathon only.
- **D-MCP: Internal architecture uses Managed Agents custom tools (client-executed), not user-facing MCP server.**
- **D2b LOCKED: 200+ labeled turns for Module A calibration.** 8-12h hand-label budget Day 1 evening → Day 2 morning.
- **D1: still open** — Krippendorff α vs Gwet AC1 vs macro-κ. Python makes all three cheap; pick based on label distribution once data is in hand (end of Day 2 morning).
- **D3a LOCKED: custom tools only for corpus access** (no HTTPS tunnel, no `resources` mount by default).

Remaining open decisions preserved at the bottom of this document.

# Lucid Stack & Distribution Decision

A second-round deepening document focused on the fundamental choices that drive everything else. Based on 13 parallel verification agents and your three clarifying answers. **This document is decision-oriented, not implementation-oriented — its job is to expose trade-offs, not to prescribe.** Every claim is cited or flagged as uncertain.

---

## User inputs (2026-04-21)

| # | Question | Your answer | Implication |
|---|---|---|---|
| 1 | Strongest language fluency | Equal fluency in Python / TS; open to Rust/Go if analysis compels | Pure engineering decision; no skill tax on either path |
| 2 | Primary target user for first publicly shippable v1 | **General Claude users** | Zero-friction distribution becomes load-bearing. Not "researchers who are comfortable with `pip`" |
| 3 | Claude Code plugin / Synapti timing | **Both — v1 ships CLI + plugin** | Plugin distribution is in-scope for 6-day build, not deferred |

These three answers change the weighting of every subsequent trade-off.

---

## What the deepening round changed (critical findings)

### F1 — Managed Agents has full Python/TS parity. Not a differentiator.

Confirmed against [platform.claude.com/docs/en/managed-agents/quickstart](https://platform.claude.com/docs/en/managed-agents/quickstart) and the [TS SDK `api.md`](https://github.com/anthropics/anthropic-sdk-typescript/blob/main/api.md). Both `anthropic` (Python) and `@anthropic-ai/sdk` (TS) expose identical `beta.agents`, `beta.environments`, `beta.sessions`, `beta.sessions.events.stream`. Beta header `managed-agents-2026-04-01` is current as of 2026-04-21 and auto-applied by both SDKs. **Language choice cannot be made on this axis.**

### F2 — MCP corpus access should use CUSTOM TOOLS, not mounted files or tunneled MCP.

This is a significant simplification of the build plan. Per [Tools — Custom tools](https://platform.claude.com/docs/en/managed-agents/tools) and [MCP connector](https://platform.claude.com/docs/en/managed-agents/mcp-connector):

- **MCP via `mcp_servers` requires an HTTPS endpoint.** The cloud container cannot reach a local stdio MCP server — we'd need Cloudflare Tunnel / ngrok + vault-managed auth. Documented complexity.
- **Custom tools (`type: "custom"`)** are the explicitly-documented pattern for user-defined, client-executed work. The cloud agent emits `agent.custom_tool_use` events; our local process handles them and replies with `user.custom_tool_result`. No tunnel. No auth. No public endpoint. **This matches Lucid's `query_corpus` / `store_finding` / `log_progress` shape exactly.**
- **`resources` file mounts are read-only.** Useful for attaching `corpus.sqlite` as a read-only artifact for the agent's bash tool, but findings write-back needs custom tools or session-scoped file writes + download.

**Impact on plan:** Phase 5's MCP server is simpler than assumed — no tunnel setup, no HTTPS certificate. Plan section "MCP server integration with Managed Agents" in the main plan should be amended.

### F3 — Krippendorff's α is Python-only. Has no maintained JS/TS/Rust package.

Verified against npm, crates.io, CRAN. The only JS implementation (`cohens-kappa-JS`) handles only two-rater Cohen's κ, no multi-label, no missing-data, abandoned since ~2015. **Porting `fast-krippendorff` to TS**: 200-400 LOC core + bootstrap infrastructure + numerical-stability testing against Python reference ≈ 40-60 hackathon hours.

**This is a real constraint on TS-only paths**, but see F4 and F5 — the methodology itself is worth challenging.

### F4 — Krippendorff's α may be the WRONG metric for sycophancy. Gwet's AC1 dominates for skewed labels.

Per Feinstein & Cicchetti 1990 ("paradox of kappa") and [PMC12163189 (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12163189/): Krippendorff's α inherits the same prevalence paradox as Cohen's κ. When 85-90% of turns have no sycophancy (which they don't — sycophancy is rare), α collapses to 0.3-0.5 even with 95% observed agreement. Gwet's AC1 uses conditional probability of random agreement and handles skew cleanly. Clinical pathology precedent: 88% observed agreement → κ=0.43, AC1=0.85.

**`irrCAC` Python package** covers both α and AC1. Switching costs ≈ 0. **This is a methodology change that stands independently of the stack decision.**

### F5 — 50-100 labeled turns is UNDER-POWERED for stable IAA on 13 labels.

Per [PMC12935580 (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935580/), detecting κ ≥ 0.60 with 85% power needs ~177 subjects *per label pair*. Lucid's current plan target (50-100 turns × 13 labels) produces CI widths > 0.15 on skewed labels. **Robust minimum: 250-300 labeled turns.**

**Options:** (a) collect more labeled data Day 1 PM (hand-label 200+ turns); (b) ship with 100 turns + aggressive CI-width disclosure; (c) descope to 5-7 high-prevalence behaviors where 100 turns suffices.

### F6 — SpiralBench's actual IAA methodology is undocumented publicly.

Agent-9 could not locate `compute_agreement.py` or methodology description on [github.com/sam-paech/spiral-bench](https://github.com/sam-paech/spiral-bench). **Lucid claims SpiralBench-compatibility but we don't know what to match.** This is a Day 1 outreach task: contact Sam Paech directly, or reverse-engineer from labeled data.

### F7 — Sharma 2023 (arxiv:2310.13548) doesn't report formal IAA.

It reports binomial CIs on sycophancy prevalence. **Lucid is actually proposing a higher methodological bar than the paper it cites** — this is good for the Depth judging axis, frame it as methodological advancement rather than gap-filling.

### F8 — Hackathon rules VERIFIED from official PDF (2026-04-21).

Received official "Participant Resources" PDF. All prior PRD assumptions reconciled:

| Claim | Status |
|---|---|
| Judging weights Impact 30 / Demo 25 / Opus 4.7 Use 25 / Depth 20 | ✓ exact match |
| 3-min demo video + GitHub + 100-200 word summary | ✓ verified |
| $5K each for "Best use of Managed Agents", "Keep Thinking", "Most Creative Opus 4.7 Exploration" | ✓ names verified; all amounts in Claude API credits |
| Top-3 prize structure | ⚠️ corrected: $50K / $30K / $10K (Claude API credits, not cash) |
| Deadline Sunday April 26, 20:00 EST | ✓ verified |
| **Open-source mandate** | **New hard rule**: every demoed component must ship under an approved OSS license. MIT satisfies this. |
| **"New work only"** | **New hard rule**: all code must start from scratch during hackathon. Planning documents (PRD, BUILD_GUIDE, this decision doc) are not "work" in that sense. |
| Team size | up to 2 members allowed (currently solo). |
| Two-stage judging | Stage 1 async (Apr 26-27) → Top 6; Stage 2 live deliberation on pre-recorded demos (Apr 28). No live product-demo requirement. |

**Must-attend live sessions:**
- **Thu Apr 23, 11:00 AM EST — Michael Cohen (Claude Labs) on Managed Agents.** Load-bearing for the Best-Managed-Agents prize bid.
- **Wed Apr 22, 12:00 PM EST — Thariq Shihipar AMA on Claude Code.** Useful, not required.
- **Daily 5-6 PM EST Anthropic office hours** in `#office-hours` on Discord — use for blockers.

Daniel's Discord join link: https://anthropic.com/discord.

### F14 — Lucid fits Problem Statement 2 directly. This reshapes demo framing.

PDF lists two problem statements. Impact (30%) explicitly asks "Does it fit into one of the problem statements?"

**Problem 2 — "Build For What's Next":** *"Start from something that doesn't exist yet: a new way to work, learn, or make that only makes sense now that the tools have changed. An interface without a name. A workflow from a few years out. The best projects here are easier to demo than to explain. Looks like: an interface that doesn't have a name yet. A workflow that changes how you do the thing, not just how fast. A first draft of how this will work in a few years when Claude is even more capable."*

Lucid is verbatim this. Not "academic sycophancy tool"; it's "first-draft interface for auditing the record of your thinking-with-AI."

**Problem 1 — "Build From What You Know":** Daniel's 9,840 Claude Code sessions + 90-day Claude.ai export is legitimate domain expertise. "The thing only you'd know to build" fits loosely, but Problem 2 is the cleaner frame.

**Demo arc implication:** open not with research citations, but with the first-draft-interface framing. The judge's mental model should be "this is what this corpus becomes when you treat it as an artifact." Research-framework rigor stays in the footer and Depth-scoring axis, not the hero beat.

**Keep Thinking prize description (verbatim from PDF):** *"For the project that didn't stop at the first idea — and landed somewhere nobody saw coming. We're looking for the team that found a real-world problem nobody thought to point Claude at. The one that changes how we think about where this technology belongs."* Module H (auditing AI memory against the corpus that produced it) is the cleanest bid for this. Frame accordingly.

### F15 — Distribution target = "ease of installing and running on your own machine, on your own data."

User clarification: not "researchers with `pip` already installed"; not "Claude Code plugin users." Just: I point this at my corpus, it works. Binary download > `brew install` > `uvx` > `pip` > `npx`. Plugin shipping adds zero demo value because judges watch a 3-minute video.

**Locked as D-PLUGIN: no plugin in v1. Ship CLI + compiled binary only.** Plugin / Synapti / marketplace all move post-hackathon.

### F16 — "Researcher ergonomics" clarified.

I used the phrase loosely in the earlier draft. Two distinct meanings that should not be conflated:

**(a) Install-friction ergonomics** — researchers already have Python, so `pip install` doesn't scare them. Under F15's general-user framing, this is moot. A binary beats `pip` for everyone, including researchers.

**(b) Statistical-library ergonomics** — researchers (and judges looking at the repo for Depth scoring) want to see `from krippendorff import alpha` and `from irrCAC.raw import CAC` in the code, not a hand-rolled implementation. This is the real constraint. **Python wins unambiguously on (b); it does not matter for distribution.**

Practically: (a) doesn't drive the stack decision anymore; (b) still pushes Python for the calibration module. This resolves the earlier "researcher ergonomics" hand-wave into a concrete Depth-scoring argument.

### F9 — The Module H "unique bid" stands, with framing caveat.

No direct overlap. Closest adjacent work: Anthropic's Petri auditing tool (LLM-side, not user-side) and the Aug 2025 Claude memory transparency release (forward-facing, not backward corpus audit). **Frame Module H as "user auditing their own corpus against AI memory synthesis" — not "auditing the memory system's internal fidelity."**

### F10 — Full-scope completion probability is ~4-8%; realistic minimum viable is ~45-60%.

Agent-11's detailed slip analysis. Phase 6 (Module A calibration) is the pivot. If calibration slips ≥ 4 hours, everything downstream cascades. **Plan should explicitly commit to a "minimum viable subset" Day 2 evening** — don't discover it Day 5.

### F11 — Distribution friction: compiled binary wins for general users; uvx for researchers; npx mid.

Per Agent-5 with 2026 adoption data: `uv` adoption is 30% of new Python repos and 74.2% "admired" on Stack Overflow. `npx` has 800-1200ms cold-start and 300-500MB cache. Bun compile produces 40-60MB binaries with 5-10ms startup. PyInstaller produces 30-100MB binaries with 200-400ms startup.

**For general Claude users (your Q2 answer):** compiled binary dominates. Bun has a small edge over PyInstaller (smaller + faster startup + cross-platform targeting in one command). Homebrew tap amplifies reach.

### F12 — Claude Code plugin format is language-agnostic at the manifest level.

Per [code.claude.com/docs/en/plugins](https://code.claude.com/docs/en/plugins): plugins bundle skills (Markdown), agents, hooks, and MCP servers. MCP servers can be any language. **A Python CLI can be distributed as a Claude Code plugin via an embedded Python MCP server, no TS required.** The `mcp` Python SDK (22.7k stars) + FastMCP decorator pattern is officially documented. Most existing plugins happen to use TS, but that's convention, not constraint.

### F13 — Bun compile is legitimately ship-ready for a 2026 TS CLI.

Per Agent-12: Bun 1.3.13 (Apr 20, 2026) is stable. `bun:sqlite` is 3-8× faster than alternatives, built-in, no native bindings. `bun build --compile` produces ~50MB binaries with 5ms startup. MCP TS SDK officially supports Bun. Anthropic SDK works via npm compat layer (low risk but unverified against Managed Agents beta specifically — Day 1 de-risk task if we pick this path).

---

## The three approaches

Scored on 8 criteria (1-5; 5 best). Weights reflect your Q2 answer ("general Claude users") and Q3 ("v1 ships CLI + plugin").

| Criterion | Weight | A: Python | B: TS+Py sidecar | C: TS-only |
|---|---:|---:|---:|---:|
| Distribution friction (general users) | 25% | 3 | 4 | 5 |
| Calibration statistical rigor | 20% | 5 | 5 | 2 |
| Plugin-first fit (Claude Code / Synapti) | 15% | 3 | 4 | 5 |
| Managed Agents integration ease | 10% | 5 | 4 | 4 |
| MCP SDK ergonomics | 10% | 5 | 3 | 4 |
| Dev-loop speed (hackathon iteration) | 10% | 4 | 2 | 5 |
| Shipping risk / novel tooling | 5% | 5 | 2 | 3 |
| Post-hackathon ecosystem trajectory | 5% | 4 | 3 | 5 |
| **Weighted score** | 100% | **3.95** | **3.85** | **3.80** |

Scores are within 0.15 of each other — any of the three is defensible. The tie-breaker is what you're optimizing for.

### Post-clarification score revision (2026-04-21 PM)

With D-PLUGIN=no and D-MCP=internal-only, the plugin-fit criterion's weight drops to ~0 (it was 15%). Rescoring with those 15 points redistributed to distribution friction (25→35%) and shipping confidence (5→10%):

| Criterion | Revised weight | A: Python | B: TS+Py sidecar | C: TS-only |
|---|---:|---:|---:|---:|
| Distribution friction (binary / `brew` / `uvx`) | 35% | 4 | 4 | 5 |
| Calibration statistical rigor | 20% | 5 | 5 | 2 |
| Managed Agents integration ease | 10% | 5 | 4 | 4 |
| MCP SDK ergonomics (internal orchestrator) | 10% | 5 | 3 | 4 |
| Dev-loop speed | 10% | 4 | 2 | 5 |
| Shipping risk / novel tooling | 10% | 5 | 2 | 3 |
| Post-hackathon ecosystem trajectory | 5% | 4 | 3 | 5 |
| **Revised weighted score** | 100% | **4.45** | **3.50** | **3.85** |

**Approach A wins unambiguously after the clarifications.** B collapses because its plugin-fit advantage is gone and hybrid complexity remains; C still looks OK on distribution but the 40-60h Krippendorff-port cost for zero distribution gain over Approach A's PyInstaller binary is no longer justifiable.

**Final stack recommendation: Approach A, Python + `uv` + PyInstaller/Nuitka compiled binary.**

### Approach A — Pure Python + compiled binary + uvx

**Stack:** Python 3.11 / uv / Pydantic v2 / `anthropic` Python SDK / `mcp` + FastMCP / `voyageai` / `krippendorff` + `irrCAC` / Jinja2 / sync `sqlite3` for ingest + `aiosqlite` for orchestrator / Typer + Rich.

**Distribution:**
1. **Primary:** PyInstaller or Nuitka one-file build → `lucid` binary on macOS/Linux/Windows via GitHub Releases (~40-60MB, ~200-400ms startup).
2. **Secondary:** Homebrew tap (`brew install lucid-ai/lucid/lucid`).
3. **Researcher:** PyPI → `uvx lucid audit` (ephemeral, 300ms).
4. **Plugin:** Synapti marketplace entry pointing to a Python MCP server wrapped around the CLI's `query_corpus` / `store_finding` tools. Users install the CLI binary separately; the plugin configures the MCP connection.

**What you keep:**
- Krippendorff's α primary + Gwet's AC1 secondary (both via `irrCAC` Python) — **full statistical rigor, no reimplementation**.
- `ProcessPoolExecutor` + orjson for 9,840-file JSONL parse (~30s).
- `count_tokens` endpoint for cost gate.
- All of Phase 6's calibration design stands.
- The `mcp.server.fastmcp.FastMCP` decorator pattern makes the orchestrator's 10 MCP tools idiomatic (~60 lines per tool).

**What you trade:**
- PyInstaller binary is larger and slower to cold-start than Bun compile. Acceptable for a `lucid audit` run that's already ~10 minutes of wall clock.
- Plugin distribution via "install Python CLI separately, wire into Claude Code" has more friction than "install plugin directly."
- macOS notarization requires an Apple Developer account ($99/yr) for Gatekeeper-clean distribution. Or ship unsigned + instruct users to right-click-open (friction).

**Risk profile:** Lowest. No novel tooling. Python ML-stack familiarity carries through Phase 6 (calibration) where most other paths add work.

**Estimated Phase-level delta vs current plan:**
- Phase 1: add PyInstaller config + GitHub Action for binary release (+2h).
- Phase 11: add Homebrew tap setup + Synapti submission (+3h).
- No other phase changes; calibration methodology holds; custom tools replaces MCP tunnel.

**Total build risk: ~40% completion probability for full scope; ~65% for minimum viable subset.** (Agent-11's numbers, adjusted for simplified MCP path.)

### Approach B — TypeScript CLI (Bun) + optional Python calibration sidecar

**Stack:** TypeScript / Bun / `@anthropic-ai/sdk` (TS) / `@modelcontextprotocol/sdk` (TS v1) / `voyageai` npm SDK (or OpenAI TS SDK fallback) / `bun:sqlite` / Zod for schemas / `commander` or `citty` for CLI. **Python sidecar: `lucid-calibration` on PyPI, invoked as subprocess by `lucid calibrate`.**

**Distribution:**
1. **Primary:** `bun build --compile` → ~50MB binary for macOS (arm64+x64) / Linux / Windows via GitHub Releases.
2. **Secondary:** npm package `@lucid/cli` → `npx lucid audit` (800-1200ms cold-start, 300MB cache).
3. **Researcher:** `pip install lucid-calibration` for `lucid calibrate` (invoked via subprocess; graceful failure if Python absent).
4. **Plugin:** Synapti marketplace plugin with embedded TS MCP server (native fit; most existing plugins are TS).

**What you gain:**
- Best-in-class distribution for general users. `bun build --compile` produces a 5ms-startup 50MB binary. Plugin shipping via Synapti is native-ecosystem.
- Bun's `bun:sqlite` beats anything Python has for dev-loop speed — though total audit wall-clock is LLM-bound, not DB-bound, so this is iteration-time improvement, not runtime win.
- Calibration stays rigorous because Python handles it via sidecar.

**What you trade:**
- **Hybrid complexity.** Subprocess semantics, error handling, shared schema (Finding, CorpusStats) needs to live in two places. Real distribution friction for researchers who expected a single `pip install`. Agent-13 found almost no production precedents for this pattern — it's a pattern you'd be pioneering, not borrowing.
- **Voyage TS SDK is v0.2.1 (2 minor versions behind Python v0.3.7).** Functional but less field-tested.
- **Managed Agents × Bun** is unverified. Anthropic SDK works on Bun via npm compat, but the specific `beta.sessions.events.stream` streaming behavior with async iterators under Bun has no public confirmation. **Day 1 de-risk task: thin-slice verify before committing.**
- **Zod vs Pydantic** for complex discriminated unions (`ContentBlock`, `Finding` subtypes). Zod is good but not as expressive; schema definitions will be noisier.
- **Dev-loop ergonomics for Python subprocess** from TS: arguments serialized as JSON, results parsed, errors translated. ~2h of plumbing per command.

**Risk profile:** Highest novel-tooling surface. The Managed-Agents-on-Bun unknown is the biggest single risk.

**Estimated Phase-level delta vs current plan:**
- Phase 1: entirely TS bootstrap (pyproject → package.json; uv → bun; pytest → bun test). Replace conftest.py with vitest-style mocking. (~5h)
- Phase 2: schemas from scratch in Zod or TypeBox. `ContentBlock` discriminated union needs `z.discriminatedUnion("type", [...])`. (~4h)
- Phase 3: ingest uses `Bun.file()` + streaming JSON via `stream-json` or `JSONStream`. Loses `ijson`'s ergonomics for nested arrays. (~4h)
- Phase 5: MCP server is cleaner in TS (well, similar). Bun + Managed Agents needs verification. (+2h de-risk)
- Phase 6: **entirely new directory `lucid-calibration/` Python package**, published separately. Subprocess glue in `lucid calibrate`. (~5h)
- Phase 11: npm publish + Homebrew + Synapti. Arguably cleaner than Python path. (~3h)

**Total build risk: ~25% completion probability for full scope; ~50% for minimum viable subset.** Extra ~16h of hybrid plumbing eats into Phase 6-8 buffer.

### Approach C — TypeScript-only + reimplement Krippendorff's in TS

**Stack:** Identical to Approach B except: no Python sidecar. Port `fast-krippendorff` to TS (200-400 LOC). Implement bootstrap BCa from scratch (another ~200 LOC). Validate numerical output against Python reference on known-answer test cases.

**What you gain:**
- Single stack. True npx/binary distribution. No subprocess. Researchers get `npx lucid calibrate` with zero dependency.
- Cleanest plugin story.

**What you trade:**
- **40-60 hackathon-hours to port + validate.** That's 1-1.5 days of 6. Direct time loss from Phase 6 and Phase 8.
- **Numerical accuracy risk.** The irr R package had a silent Krippendorff bug for years (~0.01-0.03 underestimate). Lucid's demo / README claims calibration numbers — publishing TS-reimplementation numbers that quietly disagree with published Python results would be embarrassing and hard to debug post-submission.
- **Testing burden.** You need ≥20 known-answer fixtures from scipy/R references to validate the port. Edge cases (perfect agreement, all-missing, one-category-dominant) must each match the reference to ~1e-6 precision. This is real test-writing work, and without it the credibility argument collapses.
- **Bootstrap BCa in TS** has no scipy equivalent. `simple-statistics`, `stdlib-js` don't cover it. Port from scipy's source is another ~150 LOC + validation.

**Alternative methodology pivot that unlocks C:** drop Krippendorff's α entirely; use **macro-averaged per-label binary Cohen's κ** (available in `@tensorflow/tfjs-metrics` or implementable cleanly) + **Gwet's AC1 per-label** (simpler algorithm, ~100 LOC port). This loses a single-number α summary but gains interpretability ("which behaviors disagree most?"), and several NLP 2026 papers report multi-label κ + AC1 pairings over single α. **This is actually a defensible methodology for Approach C and sidesteps the worst of the port effort.**

**Risk profile:** Second-highest. Tooling is simpler than Approach B but novel-math-port is a hidden schedule bomb.

**Estimated Phase-level delta vs current plan:**
- Phase 1-5: same as Approach B.
- Phase 6: +16-40h for Krippendorff port + validation (or -4h if you adopt the macro-κ + AC1 methodology pivot above).
- Phase 7-8: same as Approach B.

**Total build risk: ~20-35% completion probability for full scope (range depends on whether you adopt the κ+AC1 methodology pivot); ~45-55% for minimum viable subset.**

### Approach D (ruled out) — Rust/Go

Considered per your "open to Rust/Go" answer. Rejected because:

- Anthropic SDK: no official Rust or Go SDK for Managed Agents beta. Third-party clients exist but lag significantly. Would require implementing the streaming events API from scratch against REST.
- MCP SDK: community Rust (`rust-mcp-sdk`) and Go (`mcp-go`) exist but are not first-party; MCP spec evolution risks breaking them mid-build.
- Voyage AI: no SDK; direct HTTP.
- Statistical libraries: weak. No Krippendorff's in crates.io.
- Dev-loop: slower than TS/Python in this domain (compile times add up across 100+ iterations in 6 days).
- **Net: you'd spend the first two days reimplementing SDK surface area.** Rejected for hackathon timeline; revisit post-hackathon if Lucid gains traction.

---

## Stack decision matrix (quantified on your Q2/Q3 priorities)

Given your weighting (general users + plugin v1), here's the re-computed preference:

```
A (Python + binary + uvx)      : shipping confidence ★★★★★ | distribution ★★★☆ | plugin fit ★★★☆  
B (TS Bun + Py sidecar)         : shipping confidence ★★☆ ☆☆ | distribution ★★★★☆ | plugin fit ★★★★☆  
C (TS-only + reimplement)       : shipping confidence ★★★ ☆☆ | distribution ★★★★★ | plugin fit ★★★★★
```

**My read:** given the hackathon timeline dominance and "equal fluency" answer, Approach A wins on expected-value. Approach C wins on IF-it-ships-it-ships-beautifully but is a bigger gamble. Approach B is the weakest because it carries both hybrid complexity and doesn't fully escape Python distribution friction.

**But** — and this matters — the dominant criterion under Q2 is "general Claude users install and run the tool in 10 seconds." A `brew install lucid` or `bun build`-compiled binary download beats a `pip install` even when Python is already present, because "Python" carries cognitive load for non-technical users that the binary and Homebrew paths don't.

**Recommended primary:** **Approach A**, with explicit PyInstaller + Homebrew tap + Synapti plugin-MCP-server as Phase 11 distribution work.
**Recommended secondary (if A feels too conservative):** **Approach C with the macro-κ + AC1 methodology pivot** (sidesteps Krippendorff's port, keeps single-stack TS).

**Not recommended:** Approach B. The hybrid path lacks precedent (Agent-13 found almost nothing), and the Python-subprocess-from-TS friction defeats the distribution argument it's supposed to strengthen.

---

## Orthogonal methodology decisions (independent of stack)

These stand regardless of A/B/C.

### D1 — Primary calibration metric: α vs AC1 vs macro-κ

Three options:

| Option | Primary | Secondary | Rationale | Works in |
|---|---|---|---|---|
| D1a | Krippendorff's α ≥ 0.67 | per-label Cohen's κ, QWK intensity | Matches Lucid's current plan | Python (A, B) |
| D1b | Gwet's AC1 ≥ 0.70 | α ≥ 0.67, per-label κ | Handles sycophancy label skew correctly; paradox-robust | Python (A, B) |
| D1c | macro-averaged per-label Cohen's κ ≥ 0.60 | Gwet's AC1 per-label | Multi-label-native; transparent; portable to TS | All three stacks |

**Recommendation:** **D1b** if we stay in Python (most defensible for skewed labels). **D1c** if we go to TS (sidesteps Krippendorff's port).

### D2 — Labeled sample size target

| Option | Labeled turns | CI width on α | Hackathon hours to produce |
|---|---|---|---|
| D2a | 50-100 (current plan) | 0.15-0.25 (loose) | 0 (if SpiralBench public) / 4-6 (hand-label) |
| D2b | 200-300 (robust) | 0.06-0.10 | 8-12 (hand-label) or SpiralBench availability |
| D2c | 50-100 + descope to 5-7 high-prevalence behaviors | 0.10-0.15 | 0-2 |

**Recommendation:** **D2c** — descope to 5-7 behaviors with prevalence ≥ 10% of turns. Reports a single α (or macro-κ) across the 5-7, plus per-label numbers. Honest + achievable.

### D3 — Managed Agents corpus access pattern

| Option | Pattern | Setup cost | Limitation |
|---|---|---|---|
| D3a | Custom tools (client-executed) | ~0 — just handlers | Corpus must stay on your machine |
| D3b | `resources` file mount (read-only SQLite) | Upload file per session | Agent can read raw corpus via bash; can't write findings back |
| D3c | Remote MCP server via HTTPS tunnel | Cloudflare Tunnel + vault auth | Exposes local port; authentication complexity |

**Recommendation:** **D3a** for the orchestrator's main data flow (query_corpus, store_finding, log_progress, invoke_module). Optionally **D3b** as a secondary mount of `corpus.sqlite` if you want the agent to run ad-hoc SQL via bash.

**Amend Phase 5 of the main plan to reflect this.** The MCP server becomes optional — a `lucid mcp` subcommand for external Claude Code integration, not a required component of the orchestrator flow.

### D4 — Module H retrieval enhancements

From Agent-8's prioritization:

| Enhancement | Value | Effort | Recommended? |
|---|---|---|---|
| Validate 0.35 similarity threshold on 200-300 labeled pairs | HIGH | 2h | ✓ yes |
| Hybrid BM25 + dense retrieval (rank_bm25 / bm25s or TS BM25 lib) | HIGH (+5-10% recall) | 1.5h | ✓ yes |
| Two-stage decomposition (current plan) | HIGH | (in plan) | ✓ keep |
| Lightweight reranker (MiniLM cross-encoder) | MEDIUM (+2-4%) | 1h | optional |
| Evidence-adequacy Opus prompt on ambiguous | MEDIUM (+4-6%) | 1h | optional |
| ColBERT late-interaction | LOW | 4h+ | ✗ skip |
| 3-run self-consistency ensemble | LOW | 1h | ✗ skip |

**Recommendation:** add validation + BM25 hybrid to Phase 8. Keep everything else in the current plan.

### D5 — Phase 6 calibration time budget

Options:

| Option | Budget | Policy if not reaching threshold |
|---|---|---|
| D5a | Current plan: Day 2 full day, hope to finish | Re-plan on the fly |
| D5b | Day 2 full + Day 3 morning (8h buffer) | Committed in advance |
| D5c | Day 2 AM only, hard-cap at noon | Descope to 5-7 behaviors at noon, move on |

**Recommendation:** **D5c** for optionality + low regret. Schedule is the scarce resource; descoping at noon Day 2 is cheaper than discovering Phase 7 slip Day 4.

### D6 — Cerebral Valley rules verification

**Day 1 noon kickoff** (starting now, as you read this) is when the authoritative rules ship. Task: **pre-kickoff** — pull the official event page once more. **At kickoff** — transcribe the actual weights, prize list, submission requirements. **Post-kickoff** — update PRD §3 and Phase 11 checklist to match. If real weights differ materially from the PRD's 30/25/25/20, the "Depth" emphasis (which drove calibration rigor) may change.

---

## Uncertainties and weaknesses of this analysis

In the spirit of "triple-verify everything, challenge your own assumptions":

**U1 — Cerebral Valley rules.** I can't verify the PRD's judging weights, specific prize amounts, or demo video requirement against public sources. The entire scoring model in the plan (calibration rigor as Depth differentiator) rests on assumptions about what judges actually weight. Day 1 kickoff is mandatory verification.

**U2 — Managed Agents on Bun.** No public test of `client.beta.sessions.events.stream` under Bun 1.3.x. I'm inferring compatibility from general npm-compat success. Worth a 30-minute thin-slice test before committing Approach B or C.

**U3 — SpiralBench data + methodology.** I could not locate the actual labeled data format or the paper's IAA methodology. If the data is not publicly downloadable, Phase 6 pivots from "validate against SpiralBench" to "hand-label + compute own baseline" which is a different pitch. Day 1 verify.

**U4 — Claude Code plugin for non-TS tools.** The docs say plugins are language-agnostic at the manifest level, but the agent also found that published Python-only plugins are "rare or absent." I can't fully verify that a pure-Python plugin works identically to a TS plugin from the end-user's `/plugin install` perspective. Day 1 risk — test early if plugin-in-v1 is held firm.

**U5 — Ranked completion probabilities.** Agent-11's 4-8% / 45-60% figures are its synthesis from comparable-project data, not a calibrated estimate. Treat as directional, not quantitative. The right question is: "what's the minimum-viable subset that still clears the judging threshold, and can I ship it by Thursday?" — not "what's the exact probability of full scope."

**U6 — Distribution friction numbers.** Agent-5 cited `uv` at 30% adoption and Bun at various benchmarks. Adoption metrics shift monthly in 2026; these are April 2026 approximations. The qualitative ordering (binary > uvx > npx > pip) is robust; the specific percentages are directional.

**U7 — My own priors.** Writing this as someone who has a natural preference for shipping Python for LLM-heavy work and who treats "novel tooling risk" as a significant cost. If you weight "demo polish" and "ecosystem trajectory" above "shipping confidence," Approach C becomes genuinely more compelling than my ranking suggests.

**U8 — The hybrid Approach B may deserve a more generous read.** Agent-13's search for precedents was thorough but biased toward "CLI tools" specifically. `lucid audit` being Bun-native while `lucid calibrate` being Python-native is a cleaner separation than most attempted hybrids, and the "Python is only required for the 5% of users who calibrate" framing is defensible. If researchers are willing to `pip install lucid-calibration` (which they will — they all have Python), the complexity cost is lower than Agent-13 suggests.

**U9 — macOS notarization was glossed over.** All three approaches hit it. $99/year Apple Developer account, code-signing setup, stapling. Tends to eat ~4h the first time. Not included in phase estimates.

---

## Status of decisions (2026-04-21 PM, post-PDF + clarifications)

| Decision | Status | Choice |
|---|---|---|
| **D-STACK** (stack) | RECOMMENDED; awaiting explicit confirm | **Approach A: Python + compiled binary + `uv`** |
| **D-PLUGIN** (plugin in v1) | LOCKED | **No plugin v1. CLI only.** Plugin/marketplace deferred. |
| **D-MCP** (user-facing MCP server) | LOCKED | **None.** Internal architecture uses Managed Agents custom tools (client-executed). `lucid mcp` subcommand deferred. |
| **D2** (labeled sample size) | LOCKED | **D2b: 200+ labeled turns.** 8-12h hand-label Day 1 PM → Day 2 AM. |
| **D3** (corpus access pattern) | LOCKED | **D3a: custom tools only.** No HTTPS tunnel, no `resources` mount by default. |
| **D1** (primary calibration metric) | OPEN | Krippendorff's α vs Gwet's AC1 vs macro-κ. **Decide end of Day 2 AM once label distribution is visible.** Python makes all three cheap. |
| **D4** (module scope) | OPEN | Keep current plan (A, B×2, C, D-opt, E, F, G, H) or aggressive cut to (A, B×2, G, H)? See "Scope rec" below. |
| **D5** (demo framing) | OPEN | Research-first vs Problem-2-first opening. **Recommend Problem-2-first.** See F14. |

## Scope recommendation in light of Problem Statement 2 framing

"General Claude users" + "interface that doesn't have a name yet" + 8 detection modules competing for 3 demo minutes doesn't fully compose. Recommendation to sharpen for demo:

**Tier 1 (visible in demo, 2 min of the 3):**
- Module A (SpiralBench behavioral profile) — the cool chart
- Module B (feedback sycophancy) — the "wait, that really happened?" quote
- Module H (memory-corpus consistency) — the Keep Thinking prize bid

**Tier 2 (shipped, invisible in demo, visible in README):**
- Module C (SycEval classifier over A+B)
- Module E (belief drift) — mentioned as "tracking your positions over time"
- Module G (time/model attribution) — sprinkled into the chart tooltips
- Module F (ITP user-prompt analysis) — included if Day 4 has time

**Cut entirely:**
- Module D (perspective sycophancy) — was already opt-in in main plan; cement the cut.

This doesn't change Phase 7's plan meaningfully; it changes what lands in the 3-minute video and the written summary.

## Written-summary draft direction (100-200 words)

Target frame per PDF Problem 2:

> Lucid is the first-draft interface for auditing the record of your thinking-with-AI. Point it at your Claude Code sessions and Claude.ai export; it runs published sycophancy, belief-drift, and influence-tactic frameworks over your corpus — and, in Module H, audits Claude's synthesized memory of you against the conversations that actually produced it. Built on Managed Agents with per-module thinking / effort tuning and prompt caching on stable module prefixes. Calibrated against [N=200+] hand-labeled turns using Krippendorff's α / Gwet's AC1. Open source, MIT. The mirror no one built until your conversations with AI became substantive enough to need one.

Finalize Phase 11 with real N and α/AC1 numbers.

## Remaining open (and cheap to decide after Day 2 morning)

**D1 — primary calibration metric.** Once 200+ labels are in, compute the label-prevalence distribution. If ≥ 3 labels are < 10% prevalence → Gwet's AC1 primary, α secondary. If distribution is balanced → α primary, AC1 secondary. Either way, report **both** in `docs/calibration.md` and the report footer for credibility.

**D4 — final module scope.** Recommend locking at end of Day 3 based on Phase 6 calibration outcome. If Module A calibration is clean by Day 2 noon → pursue full Tier-1+Tier-2 scope. If calibration slips → cut Tier-2 aggressively.

## What I'm doing next (unless you veto)

1. Update main plan `docs/plans/2026-04-21-feat-lucid-hackathon-build-plan.md` in place to reflect:
   - Stack: Python locked
   - Distribution: PyInstaller/Nuitka binary + Homebrew tap + PyPI for `uvx`
   - No plugin / no user-facing MCP (Phase 11 simplified)
   - Managed Agents via custom tools (Phase 5 simplified — no HTTPS server, no MCP stdio wrapper)
   - 200+ labeled turns budget (Phase 0 adds label-acquisition task; Phase 6 gets +6h)
   - Demo framing: Problem Statement 2 opening (Phase 11)
   - Attend Thu Apr 23 Managed Agents session (Phase 0 checklist)

2. Surface the remaining open decisions (D1 primary metric, D4 final scope) as Day 2/3 checkpoints in the plan.

If you want me to proceed, say "go." If you want a different stack, say so now — the main-plan rewrite is cheap if we do it before Phase 1 starts.

---

## One additional thing worth raising

Your Q2 answer ("general Claude users") is the single biggest shift from the original PRD, which implicitly framed the audience as "AI safety researchers." That shift has knock-on effects I'd want you to think through before locking the stack decision:

- **Demo messaging:** if the judge is thinking "what does a regular Claude user get from this," the demo's hero beat should not be "we achieve Krippendorff's α of 0.72 against SpiralBench." It should be "I pointed Lucid at my corpus and it showed me the three times last month Claude agreed with me when it shouldn't have." Beats structure changes.
- **Report surface:** researchers want the methodology footer. General users want a hero-feeling at the top ("Claude has been 2.3× more agreeable with you this month than last"). One page reconciles both but the sequencing is opposite.
- **Calibration emphasis:** general users don't read calibration numbers. Researchers do. Calibration stays in-scope for credibility (papers + prize-judging depth), but the demo video probably shouldn't feature it.
- **Scope re-check:** general-user audience might not care about 8 modules. They care about 2-3 moments of "oh, that's interesting." A tighter scope — Modules A, B, H as the visible ones; C/D/E/F as supporting — might land harder than shipping all 8 halfway.

None of this is a reason to change the stack or plan right now. It's context for the decisions above.

---

## Sources (consolidated from 13 parallel verification agents)

### Managed Agents + SDKs
- https://platform.claude.com/docs/en/managed-agents/{quickstart, overview, sessions, events-and-streaming, tools, mcp-connector, files, environments}
- https://github.com/anthropics/anthropic-sdk-typescript (api.md)
- https://claude.com/blog/claude-managed-agents
- https://www.npmjs.com/package/@anthropic-ai/sdk

### MCP
- https://github.com/modelcontextprotocol/python-sdk (v1.27, 22.7k stars)
- https://github.com/modelcontextprotocol/typescript-sdk (v1.29, 12.2k stars)
- https://modelcontextprotocol.io/docs/develop/build-server
- https://pypi.org/project/fastmcp/
- https://github.com/modelcontextprotocol/servers (6 TS / 1 Python reference implementations)

### Voyage AI
- https://pypi.org/project/voyageai/ (v0.3.7)
- https://www.npmjs.com/package/voyageai (v0.2.1)
- https://github.com/voyage-ai/typescript-sdk
- https://blog.voyageai.com/2025/12/04/batch-api/
- https://docs.voyageai.com

### Krippendorff's α + IAA methodology
- https://pypi.org/project/krippendorff/
- https://github.com/pln-fing-udelar/fast-krippendorff
- https://github.com/kgwet/irrCAC
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12163189/ (Gwet AC1 vs κ, 2025)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12935580/ (IAA power analysis, 2024)
- https://arxiv.org/abs/2603.06865 (Counting on Consensus, 2026)
- https://www.chainsawriot.com/postmannheim/2024/10/25/krippendoff.html (irr package bug)
- https://labelstud.io/blog/how-to-use-krippendorff-s-alpha-to-measure-annotation-agreement/

### CLI distribution
- https://docs.astral.sh/uv
- https://bun.sh/docs (compile, sqlite)
- https://deno.com/blog/v1.41
- https://dev.to/kabasele754/python-in-2026-why-i-replaced-pip-with-uv-complete-guide-benchmarks-19bp

### Cerebral Valley
- https://cerebralvalley.ai/e/built-with-4-7-hackathon (verified event host)
- https://claude.com/blog/meet-the-winners-of-our-built-with-opus-4-6-claude-code-hackathon (reference for 4.6 prize structure)

### Claude Code plugins
- https://code.claude.com/docs/en/plugins
- https://code.claude.com/docs/en/plugins-reference
- https://github.com/synaptiai/synapti-marketplace

### Module H retrieval patterns
- https://arxiv.org/abs/2305.14251 (FActScore)
- https://arxiv.org/pdf/2510.14400 (MedTrust-RAG)
- https://aclanthology.org/2025.acl-long.254.pdf (Optimizing Decomposition for Claim Verification, ACL 2025)
- https://arxiv.org/pdf/2503.23013 (Dynamic Alpha Tuning hybrid retrieval)
- https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/

### Sycophancy prior art
- https://arxiv.org/abs/2310.13548 (Sharma 2023)
- https://eqbench.com/spiral-bench.html (SpiralBench)
- https://arxiv.org/html/2604.00478 (Silicon Mirror, 2026)
- https://arxiv.org/pdf/2505.13995 (ELEPHANT)
- https://alignment.anthropic.com/2025/petri/ (Anthropic Petri, closest adjacent)

### Bun/Deno and TS tooling
- https://bun.sh/docs/runtime/bun-sqlite
- https://github.com/modelcontextprotocol/typescript-sdk (Bun/Deno support claim)
- https://github.com/simple-statistics/simple-statistics (insufficient for Krippendorff's)

### Python+Node hybrid
- https://github.com/transitive-bullshit/scikit-learn-ts (sklearn via subprocess)
- https://devblogs.microsoft.com/python/feasibility-use-cases-and-limitations-of-pyodide/ (Pyodide cold-start)
- https://github.com/pln-fing-udelar/fast-krippendorff (port reference)

---

*End of decision document. Answer Decisions 1-4 to proceed with plan amendment.*
