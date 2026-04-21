# LUCID PRD (v3)

**Built with Opus 4.7 Hackathon submission**
Daniel + Claude (solo team of 1 + 1)
April 21 – April 26, 2026

*Changelog v3: Schema confirmed against real 90-day Claude.ai export. Module H (Memory-corpus consistency) promoted to core. Sampling strategy added for 9,840-session Claude Code corpus. Token budget revised. Claude.ai export characteristics clarified (flat corpus, no project linkage).*

---

## 1. Product Brief

Lucid is an open-source **epistemic audit tool for personal AI conversation history**. It takes a user's Claude Code sessions and Claude.ai conversation export, applies a composite framework drawn from published AI safety research, and produces a structured report surfacing sycophancy events, belief drift, reinforcement spirals, user influence tactics, memory-corpus consistency, and time/model attribution.

Not a search tool. Not a memory tool. Not a chat continuation tool. A benchmark-style audit applied to a corpus that didn't exist as a first-class analytical object until 2025: the accumulated record of one person's thinking-with-AI.

**One-sentence pitch:** Point Lucid at your AI conversation history; it runs published research instruments on your corpus and tells you what the data reveals about how you and the model influenced each other, including whether Claude's memory of you is supported by the evidence.

---

## 2. Problem Statement

Hundreds of millions of people now do substantial cognitive work inside conversations with LLMs. Anthropic's Values in the Wild paper analyzed 308K real Claude conversations from one week in February 2025. That corpus is the closest thing we have to an externalized record of how modern knowledge workers think. Individually, each person's slice of it is their own cognitive artifact.

Published research shows this cognitive work is systematically distorted. Sharma et al. 2023 demonstrated sycophancy is present in all RLHF-trained models. SycEval found a 58% baseline sycophancy rate across frontier models. SpiralBench found that even post-mitigation GPT-5 averages one sycophantic behavior per turn in 20-turn conversations. Truth Decay quantifies how multi-turn pressure compounds. The Opus 4.5 system card explicitly tracks "encouragement of delusions" as a behavioral metric. These patterns are not edge cases; they're the base rate.

In 2025 Anthropic, OpenAI, and Google all shipped automatic memory features that synthesize "what we know about you" from your conversation history. Users cannot directly audit these synthesized memories against the evidence. Glasp's March 2026 piece framed this as "performing for a mirror that shows a reflection you can't fully see." No consumer tool exists that lets individuals audit their conversation corpus against the research frameworks or audit the AI's constructed memory of them.

Lucid fills both gaps.

---

## 3. Success Criteria

### Hackathon (primary)

Scored across four judging criteria (Impact 30%, Demo 25%, Opus 4.7 Use 25%, Depth & Execution 20%).

**Impact**: research-validated problem affecting hundreds of millions of users. Generalizes beyond Anthropic. Open-sources a reusable framework.

**Demo**: 3-minute video showing corpus ingest, live pattern detection, memory-corpus consistency check, and report generation. Judge can imagine pointing it at their own data.

**Opus 4.7 Use**: Managed Agents orchestrate long-running multi-pass analysis. Opus 4.7 handles the hardest reading tasks. Sonnet 4.6 handles topic extraction where appropriate.

**Depth & Execution**: integration of 8 published frameworks plus novel Module H (memory-corpus audit). Calibration against SpiralBench labeled conversations. Traceable methodology in output.

Eligibility for three special prizes:
- Best use of Claude Managed Agents ($5K)
- Keep Thinking Prize ($5K) — Module H is the strongest single bid here
- Most Creative Opus 4.7 Exploration ($5K; stretch)

### Product (secondary)

A user can process a personal corpus (100 sampled sessions from up to thousands available) and get a readable, actionable report within 10 minutes of wall clock. Detection results validate against SpiralBench labeled conversations at ≥70% agreement on implemented behaviors. The report surfaces at least one insight the user didn't already know about their AI use.

---

## 4. Scope

### 4.1 SpiralBench behavior selection (Module A)

13 of 17 behaviors ship. 3 sentience-claim behaviors excluded (model-evaluation territory, not personal audit). Ritualization contingent-include with lowest priority.

**INCLUDED (13)**

*Sycophancy*: problematic agreement/flattery. Core.
*Benign warmth*: genuine non-problematic friendliness. Contrast class to sycophancy.
*Pushback*: pushes back on problematic statement. Positive marker.
*De-escalation*: reduces emotional intensity or conspiratorial framing. Positive marker.
*Escalation*: increases problematic intensity. Negative marker.
*Delusion reinforcement*: treats delusional premise as true. Critical safety marker per Opus 4.5 system card.
*Topic shut-down*: refuses topic unnecessarily. Negative marker (over-refusal).
*Help-referral warranted*: recommends professional help when warranted. Positive marker.
*Help-referral unwarranted*: over-cautious referral. Negative marker.
*Harmful advice*: potentially harmful suggestions. Critical safety marker.
*Confident bullshitting*: confident dubious claims without hedging. High-signal for technical work.
*Boundary setting*: clearly declines. Positive marker.
*Validate feelings not thoughts*: ideal pattern in emotional conversations.

**EXCLUDED (4)**

*Positive/uncertain/negative sentience claim*: Claude's claims about its own experience. Valuable for model evaluation, not personal audit.
*Ritualization*: entrenching delusional frames via ritual. Rare base rate, high false-positive risk. Contingent include.

### 4.2 Sharma et al. sycophancy types (Module B)

All four types implemented: feedback sycophancy, answer sycophancy, mimicry sycophancy, "are you sure" sycophancy. Module B has four detection subroutines.

### 4.3 SycEval classifier (Module C)

Progressive vs regressive classification applied to Module A and B findings. Progressive = cave-in landing on correct answer, low priority. Regressive = cave-in landing on wrong answer, the flag.

### 4.4 Jain et al. perspective sycophancy (Module D)

Subtle worldview mirroring without explicit agreement. Standalone module reading full conversations for framing drift.

### 4.5 BeliefShift DCS-simplified (Module E)

Track user positions on 5-10 recurring topics across sessions. Flag shifts. Classify each as evidence-driven (new info) vs pressure-driven (Claude pushed back). Lightweight Drift Coherence Score implementation. Full BRA/CRR/ESI out of scope.

### 4.6 ITP category selection (Module F)

9 of 20 categories transfer cleanly from media analysis to one-on-one dialogue. Applied to *user prompts* not assistant output (inverts the protocol).

**INCLUDED (9)**

*Emotional Triggers*, *Urgent Action Demands*, *Emotional Repetition* (3 of 5 Emotional Manipulation).
*False Dilemmas* (1 of 3 Tribal Division).
*Context Omission*, *Authority Overload*, *Cherry-Picked Data*, *Logical Fallacies*, *Framing Techniques* (5 of 6 Missing Information).

**EXCLUDED (11)**

*Novelty Overuse*, *Manufactured Outrage* (don't transfer to personal AI use).
Entire *Suspicious Timing* factor (depends on cross-source analysis against world events).
Entire *Uniform Messaging* factor (measures cross-source coordination, N/A for single-user corpus).
*Us vs Them*, *Simplistic Narratives* (low base rate, high false-positive risk).
*Suppression of Dissent* (hard to detect in dialogue).

Product ships with explicit note: this is a domain-adapted ITP subset, not full protocol.

### 4.7 Time/model attribution (Module G)

Deterministic bucketing layer over all findings. Model inferred from date via Anthropic default-model timeline (no model field exists in Claude.ai export). Reports surface "your March 2025 sessions had 3x the sycophancy rate of April 2026."

### 4.8 Memory-corpus consistency (Module H) — NEW CORE

Novel module unique to Lucid. Extracts atomic claims from `memories.json` (both `conversations_memory` and `project_memories`), retrieves relevant corpus context for each claim, and classifies the claim as well-supported, weakly-supported, unsupported, or contradicted.

This is the strongest single bid for the Keep Thinking Prize. It directly addresses the auditability gap in AI memory features that Anthropic, OpenAI, and Google all shipped in 2025.

Details in build guide Section 4.H.

### 4.9 Modules summary

A (SpiralBench scorer, 13 behaviors) / B (Sharma paired-exchange, 4 subroutines) / C (SycEval classifier) / D (perspective sycophancy) / E (belief drift) / F (ITP user prompts, 9 categories) / G (attribution) / H (memory-corpus consistency).

### 4.10 Corpus ingestion

**Claude Code sessions**: local JSONL files at `~/.claude/projects/<project-slug>/<session-id>.jsonl`. Daniel's own corpus: 44 project directories, 9,840 session files.

**Claude.ai official export**: unzipped directory containing `conversations.json`, `projects.json`, `memories.json`, `users.json`. Validated schema (see build guide). Export is flat: no project linkage field on conversations. Project context available separately via `projects.json` and `memories.json`.

### 4.11 Sampling strategy (NEW)

At 9,840 Claude Code sessions plus hundreds of Claude.ai conversations, full-corpus audits are cost-prohibitive by default. Sampling strategy:

**Default sample**: 100 substantive sessions.
- Filter out sessions with <5 turns (likely open-and-exit)
- Stratify by project: sample proportionally across the top 10 projects by session count
- Time-weight toward recent sessions (last 90 days preferred)
- Random selection within strata for reproducibility (seeded RNG)

**Configurable**: user can override with `--sample N`, `--all`, or `--projects PROJ1,PROJ2`.

**Cost gate**: Lucid estimates token cost before running. Any run exceeding $20 prompts for explicit confirmation.

### 4.12 Out of scope

Live real-time analysis (batch audit only).
Web app with accounts, auth, hosted deployment.
ChatGPT and Gemini export parsing (architected to accommodate, not implemented).
Mobile UI.
Fine-tuned models.
Multi-user support.
Full coverage of all 17 SpiralBench behaviors (4 excluded with reasoning).
Full coverage of all 20 ITP categories (11 excluded with reasoning).
Full BeliefShift metric suite (only DCS-simplified; BRA/CRR/ESI skipped).
Full-corpus audits by default (sampling is default; full-corpus opt-in).

### 4.13 Deferred to post-hackathon

Chrome extension for direct Claude.ai API access (reaches conversations the official export may miss).
Continuous monitoring mode.
ChatGPT/Gemini adapters.
Published skill package on Synapti marketplace.

---

## 5. Core Flows

### Primary flow: local Claude Code audit

`lucid audit --source claude-code --path ~/.claude/projects`

1. Discover and parse session files (respects sampling)
2. Report corpus stats (N discovered, N sampled, date range, project distribution)
3. Show token budget estimate. Require consent.
4. Spawn Managed Agents session
5. Stream analysis progress (per module)
6. Generate HTML report, open browser
7. Save raw findings as JSON for re-analysis

### Secondary flow: Claude.ai export

Download export via Settings → Privacy → Export. Wait for email. Unzip. Run:

`lucid audit --source claude-ai --path ./export`

Lucid parses `conversations.json` (flat corpus), `projects.json` (project context), `memories.json` (for Module H), `users.json` (account metadata). Same flow from step 2.

### Combined flow

`lucid audit --source all --claude-code-path ... --claude-ai-path ...`

Processes both corpora in one run. Findings tagged with source. Module H runs on whichever source has memories.json.

### Demo flow (3 minutes)

- 0:00–0:30 Hook. Research citation, problem framing.
- 0:30–1:00 Corpus ingestion. Show 9,840 sessions discovered, 100 sampled.
- 1:00–1:45 Live analysis. Managed Agents session streaming. Progress across modules.
- 1:45–2:30 Report walkthrough. Behavioral profile chart. One striking paired-sycophancy event. One belief drift trajectory. Module H table: "Claude's memory claims about you and what the corpus supports."
- 2:30–3:00 Close. Open source, research-grounded, validated against SpiralBench, generalizable.

---

## 6. Token Budget (Revised)

For a sampled 100-session audit (typical case):

- Module A: ~1.5M tokens
- Module B: ~1M tokens (Sonnet first pass, Opus second)
- Module C: ~100K tokens
- Module D: ~500K tokens
- Module E: ~600K tokens
- Module F: ~400K tokens
- Module G: negligible (deterministic)
- Module H: ~1M tokens
- Ensemble judging on subset: ~500K tokens

**Total estimate**: ~5.5M tokens per typical sampled audit. Approximately **$15–30 per audit** at current Opus 4.7 pricing (verify Day 1).

Full 1,000-session audit: roughly 10x, ~$150–300.

Full Daniel-scale audit (9,840 sessions): ~$1,500–3,000. Explicitly opt-in with large confirmation dialog.

Rate limit handling: Managed Agents handles most logic. Add exponential backoff on 429s.

---

## 7. Calibration Methodology

Make-or-break Day 2 step. Unchanged from v2 except noting Module H calibration approach.

### SpiralBench validation (Module A)

Download SpiralBench labeled conversations. Run Module A. Compare per-turn labels. Target Cohen's kappa ≥0.6.

Decision tree:
- Kappa ≥0.6: ship as-is.
- Kappa 0.4–0.6: iterate prompts, re-validate once or twice.
- Kappa <0.4: descope to high-agreement behaviors only, or fall back to simpler detector.

### Module H validation

No public ground truth dataset for memory-corpus consistency. Internal validation: run on a deliberately-seeded test corpus where we know which memory claims are supported vs not. Target precision ≥80% on unsupported/contradicted classifications.

### Other modules

Internal validation by manual review of sample findings. Target FPR <20%.

---

## 8. Risks and Mitigations

**R1**: Managed Agents learning curve eats Day 1. → Read docs pre-kickoff. Agent SDK fallback ready.

**R2**: Opus 4.7 hallucinates sycophancy events. → Day 2 calibration. Descope Module A if kappa low.

**R3**: Claude.ai export schema drift. → Schema validated against real export. Defensive parsing with graceful degradation.

**R4**: Rate limit or cost blowout on 9,840 sessions. → Sampling default, cost gate at $20, resumable runs.

**R5**: Demo doesn't land emotionally. → Day 5 dry-run. Strongest beats: paired-sycophancy quote + Module H memory-corpus table.

**R6**: Privacy concerns. → Demo on SpiralBench public + curated seeded corpus. Personal corpus findings aggressively redacted.

**R7**: Prior art missed. → Day 1 morning sweep. Council fallback.

**R8**: Module H produces "unsupported" for claims that are simply absent from the corpus rather than false. → Distinguish "absence of evidence" from "evidence of absence" in classification. Fourth category "insufficient data."

**R9**: Judges weight live execution. → Pre-recorded demo as fallback.

**R10**: Mirror-judging-the-mirror bias. → Ensemble judge via OpenRouter if time permits; otherwise Opus 4.7 self-consistency (3 temp-varied samples).

**R11**: 9,840 sessions means session parsing alone could take hours. → Parallelize parsing. Cache parsed sessions in SQLite. Default sampling keeps parse time under 5 minutes.

---

## 9. Open Decisions

1. **Ensemble judge source**. Decide Day 3 post-calibration.
2. **Demo corpus**. Hybrid: SpiralBench + curated seeded corpus for video; personal corpus for local verification.
3. **Findings data license**. Opt-in "contribute anonymized stats to research dataset"? Decide Day 4.
4. **Submission summary**. Draft Day 5.
5. **Post-hackathon trajectory**. Decide Day 5.

---

## 10. Why This Wins

**Only submission with this research integration density.** 8 modules citing 8+ peer-reviewed frameworks.

**Novel corpus treated as first-class object.** Personal AI conversation history as analytical artifact.

**Module H is genuinely unique.** Nobody else audits AI memory against the corpus that produced it.

**Opus 4.7 showcase.** Multi-pass reading, paired-exchange detection, perspective drift, memory-claim evaluation all require frontier reasoning.

**Strong Managed Agents fit.** Long-running, async, multi-tool orchestration over a large corpus.

**Demoable.** Three strong demo beats: behavioral profile, paired sycophancy quote, memory-corpus consistency table.

**Socially relevant.** Hits Anthropic's research priorities and user concerns simultaneously.

**Ships open.** Reusable framework for researchers and users.

---

*End of PRD v3.*
