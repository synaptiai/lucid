# LUCID Podcast Format — Stretch Goal Scope

Supplement to LUCID_PRD.md v3.
Stretch-goal feature. Implement only if core modules (A-H) ship with time remaining.

---

## 1. Concept

After Lucid completes an audit, optionally generate a 5-10 minute personalized audio podcast where two AI hosts discuss the user's findings in a conversational format. Think NotebookLM's Audio Overview, but fed by Lucid's structured findings rather than user-uploaded documents, and scripted with direct references to what the user's corpus actually revealed.

The podcast isn't a replacement for the HTML report. It's a second output format. The user listens on their commute. The report lives on their laptop for detailed drill-through.

## 2. Why this is compelling as a hackathon feature

**Demoable.** 30 seconds of AI-generated podcast in the demo video is more memorable than 30 seconds of scrolling through a report. Judges remember the audio.

**"Most Creative Opus 4.7 Exploration" bid.** Generating a 2-speaker dialogue script from structured findings requires frontier reasoning — the dialogue has to be specific (reference actual findings), personalized (use the user's own quotes), and conversationally natural (not read like a bulleted list). Sonnet could do this but worse.

**Differentiator.** No existing tool in this space generates audio output. NotebookLM does podcasts but from documents, not behavioral analytics. The combination of research-grounded audit + personalized audio output is unique.

**Demonstrates the full pipeline.** The demo arc becomes: corpus ingest → Managed Agents analysis → structured findings → choice of outputs (HTML report or audio podcast). That's a richer story than "here's a report."

## 3. Why this is a stretch goal, not core

**5-day budget is tight.** Core modules A-H plus calibration plus HTML report plus demo video is already a full week for a solo build. Adding TTS pipeline work is real additional scope.

**Pipeline complexity.** Audio generation introduces new dependencies: TTS API integration, audio file handling, dialogue script generation as a separate Claude pass, file output management.

**Marginal demo value over ceiling.** If we can nail a strong 3-minute demo with HTML report + live findings, the podcast adds polish but not essential differentiation. The research-framework backbone is the primary win.

**Cut order decision.** Shipping core well beats shipping more mediocre. Podcast is the first thing to cut if Day 4-5 runs over.

## 4. TTS landscape findings

Key question you asked: does Anthropic have native TTS? **No.**

Claude supports text and image input, text output only. Claude.ai's consumer Voice Mode exists but uses third-party TTS under the hood (Hume AI via partnership). There's no Anthropic API for speech synthesis. For Lucid's podcast output, we need an external TTS service.

### Commercial options

**ElevenLabs Eleven v3 Text-to-Dialogue** — the strongest fit.

Dedicated `/v1/text-to-dialogue` endpoint specifically designed for multi-speaker podcast-style audio. Pass an array of `{voice_id, text}` objects. Supports emotional tags like `[laughs]`, `[whispers]`, `[excited]`, `[sad]` inline. Handles interruptions and overlapping voices. 70+ languages. No limit on number of speakers. 2,000 character total per request (reliable), 5,000 max.

Pricing: Eleven v3 API is in the standard ElevenLabs pricing tiers. Approximately $5-22 per million characters depending on tier. For Lucid, a 10-minute podcast at ~150 words per minute is ~7,500 characters, so roughly $0.04-0.17 per podcast generated. Negligible cost.

This is the path of least resistance for a working demo.

**Google Cloud TTS, Azure Neural TTS, OpenAI TTS** — all capable but single-speaker focused. For multi-speaker dialogue you'd manually splice clips together, which complicates the pipeline. Not ideal.

**Hume EVI** — the Anthropic partnership voice platform. Designed for real-time conversational agents, not asynchronous podcast generation. Wrong shape for our use case.

### NotebookLM path

**NotebookLM Enterprise Podcast API** exists (Google Cloud, standalone API). Takes context elements (text/images/audio/video), outputs MP3. Available GA with allowlist — meaning you have to contact Google Cloud sales to be approved. Not accessible within the hackathon timeline.

**NotebookLM consumer** has no official API. Unofficial Python wrappers exist (israelbls/notebooklm-podcast-automator uses FastAPI + Playwright; teng-lin/notebooklm-py wraps the web UI). These depend on browser automation of the consumer site, which is fragile and against NotebookLM's ToS in spirit if not letter. Skip for hackathon credibility reasons.

**AutoContent API** markets itself as a paid NotebookLM alternative. Credit-based pricing. Adds a third-party dependency with no meaningful advantage over ElevenLabs for our use case.

**Open Notebook (lfnovo/open-notebook)** — open-source NotebookLM implementation with multi-speaker podcast generation. Self-hosted. Worth keeping in mind as a fallback but overkill for a podcast feature when ElevenLabs handles it in one API call.

### Open source options

**VibeVoice (Microsoft Research)** — purpose-built for long-form multi-speaker podcasts. Up to 90 minutes, up to 4 speakers. Uses Qwen2.5 LLM for dialogue flow + diffusion head for audio. Research-grade, ICLR 2026 oral. MIT licensed. English and Chinese only. 1.5B model runs on an 8GB GPU.

Tradeoffs: self-hosted (need GPU to run inference), research quality so output is good but not commercial-polished, adds significant pipeline complexity (model download, GPU dependency, potentially requires modal.com or similar for hosted inference).

**Dia (Nari Labs)** — 1.6B params, Apache 2.0, generates highly realistic dialogue with nonverbal sounds (`(laughs)`, `(sighs)`). English-only. No fixed voice identity unless using audio prompt or seed.

**Chatterbox (Resemble AI)** — 350M params, claims to beat ElevenLabs in blind tests (63.75% preference). Emotion exaggeration control. Voice cloning. English-only. All output watermarked.

**Kokoro** — 82M params, Apache 2.0. Fastest/cheapest. 9 languages. No voice cloning. Runs on CPU.

**Higgs Audio V2 (BosonAI)** — top trending on HF. Expressive emotion, multilingual voice cloning.

All of these are real and usable, but they require running model inference ourselves. For a 5-day hackathon stretch goal, that's too much scaffolding.

## 5. Recommended architecture

**Opus 4.7 generates dialogue script → ElevenLabs Text-to-Dialogue produces MP3.**

```
Lucid findings (structured JSON)
    |
    v
[Dialogue Script Generator]
    - Opus 4.7 call with findings + podcast prompt
    - Produces: [{speaker: "host1", text: "..."}, {speaker: "host2", text: "..."}]
    - Includes emotional tags [excited], [thoughtful], [surprised]
    |
    v
[Voice Assignment]
    - Map host1 → voice_id (e.g. Rachel)
    - Map host2 → voice_id (e.g. Adam)
    - Configurable in lucid config
    |
    v
[ElevenLabs Text-to-Dialogue API]
    - POST /v1/text-to-dialogue
    - Returns MP3 stream
    |
    v
[Save MP3 + embed in HTML report]
    - lucid/report/audio/<run_id>.mp3
    - HTML report gets <audio> player at top
```

### Dialogue script generation prompt (sketch)

```
You are writing a 2-host podcast script that discusses the results of an
epistemic audit of a user's AI conversation history. The audit found specific
patterns using published research frameworks. Your job is to make those patterns
conversational, specific, and occasionally surprising.

Format:
- 2 hosts: Alex (analytical, curious) and Jordan (skeptical, asks hard questions)
- 8-12 minutes total (~1,200-1,800 words)
- Include direct quotes from the user's corpus where flagged
- Reference findings with their paper citations at least once each
- Include emotional tags like [thoughtful], [surprised], [laughing] inline
- End with one concrete insight the user can act on

FINDINGS (JSON):
{findings}

USER CORPUS METADATA:
- {conversation_count} conversations analyzed
- {date_range}
- Models: {model_distribution}

Output a JSON array: [{"speaker": "Alex" | "Jordan", "text": "..."}, ...]

Do not invent findings. Only discuss findings present in the JSON. If a finding
has a quote, use the exact quote. Be specific, not generic.
```

### Voice assignment

Two distinct voices. Both neutral-to-warm. For the demo, use ElevenLabs pre-built voices:

- **Alex (host 1)**: Rachel (female, warm, analytical)
- **Jordan (host 2)**: Adam (male, curious, conversational)

Configurable via `lucid/config.py` so users can substitute their own voice IDs post-hackathon.

## 6. Implementation estimate

**Minimum viable podcast feature: 6-8 hours of work.**

Breakdown:
1. `lucid/report/podcast.py` module (2h)
   - Dialogue script generation via Opus 4.7
   - JSON parsing and validation
   - ElevenLabs API wrapper
2. Prompt iteration to get natural-sounding dialogue (2-3h)
3. Voice assignment config, MP3 file handling (1h)
4. HTML report integration (`<audio>` player embed) (1h)
5. Testing on real Lucid findings + calibration (1-2h)

This fits in an evening if Day 4 ends with core complete. If core runs late, cut.

## 7. Schedule

**Day 4 evening (Friday April 24), if on track:** begin podcast work.
**Day 5 morning (Saturday April 25), if still on track:** finish and test.
**Day 5 afternoon:** only if all above ships, include in demo recording.

**Cut criteria (evaluate end of Day 4):**
- If Module H calibration isn't stable → cut podcast, fix H.
- If HTML report has styling issues → cut podcast, fix report.
- If demo dry-run isn't hitting 3 minutes cleanly → cut podcast, polish demo.
- If all core shipped and working → ship podcast.

## 8. Cost

**ElevenLabs pricing model**: credits per character. Creator tier at $22/month includes 100K credits (roughly 100K characters, or ~100 minutes of audio). For hackathon demo + testing, we need maybe 30-60 minutes of audio generation. One month of Creator tier covers it with margin.

**Opus 4.7 for dialogue generation**: ~5K tokens per podcast script (findings input + 1.5K word output). At $5/$25 per million, roughly $0.15 per podcast. Trivial.

**Total stretch goal budget: ~$25 for a Creator subscription + ~$1-2 in Opus 4.7 calls.** Well within hackathon credit budget.

## 9. Demo integration

If the podcast ships, the demo arc changes slightly:

- 0:00-0:30 Hook (same)
- 0:30-1:00 Corpus ingestion (same)
- 1:00-1:45 Live analysis (same, maybe shorter)
- 1:45-2:15 Report walkthrough with one striking finding (trimmed)
- 2:15-2:45 **"And Lucid can also generate this as a podcast."** Play 30 seconds of the generated podcast where the hosts discuss the same finding in conversation. This is the memorable beat.
- 2:45-3:00 Close (same)

The podcast beat is inherently surprising because the judge hasn't seen a tool in this space do it. Even if the generated dialogue is B+ quality, the *fact of it existing* is the demo moment.

## 10. Risks specific to this feature

**R-P1**: ElevenLabs API rate limits or quality issues under hackathon deadline pressure.
*Mitigation*: generate the demo podcast once on Day 5 morning, not live during recording. Pre-generated audio file is deterministic and safe.

**R-P2**: Generated dialogue reads stilted or too "AI-written" — undermines the "polished" perception.
*Mitigation*: iterate the dialogue prompt 2-3 rounds on Day 4 evening. If by Day 5 morning it's still rough, cut the feature. Bad podcast is worse than no podcast.

**R-P3**: Findings don't have enough "discussable" substance for 8-10 minutes of dialogue.
*Mitigation*: generate only a 3-4 minute podcast for the demo corpus. Scale up post-hackathon when full corpus audits are the norm.

**R-P4**: Voice assignment creates uncanny-valley effect or feels wrong for the "AI auditing AI conversations" framing.
*Mitigation*: pick voices that are clearly AI-host-style, not trying to mimic specific people. Acknowledge in podcast intro: "I'm Alex, and this is Jordan. We're going to walk through what Lucid found in your conversations."

**R-P5**: Ethical concern about generating audio that appears to be about a real person (the user) without their active consent for the audio generation specifically.
*Mitigation*: podcast generation is opt-in, triggered by explicit `--podcast` flag on the audit command. Users aren't surprised by an audio file they didn't request.

## 11. Post-hackathon trajectory

If Lucid wins or gets traction, podcast feature roadmap:

**v1.1**: Configurable podcast length (3/6/10 minute modes).
**v1.2**: User-provided voices (bring your own ElevenLabs voice ID or clone).
**v1.3**: VibeVoice self-hosted option for privacy-conscious users.
**v1.4**: Weekly auto-generation subscription — new podcast every Monday summarizing the prior week's AI usage patterns.

The last one is genuinely novel as a product direction. "Your weekly AI-use-reflection podcast" as an ongoing subscription service is the kind of thing that could become a standalone product.

## 12. Open decisions pre-implementation

1. **Voice selection.** Default voices could skew toward the podcast's personality. Analytical+skeptical pairing vs curious+enthusiastic pairing. Decide Day 4 with a 30-second test generation.

2. **Audio length.** 3 minutes (tight, demo-focused) vs 8-10 minutes (product-focused, better shows off the feature). Default 5-7 minutes for balance.

3. **Audio format.** MP3 is safe. WAV is higher quality but larger. Default MP3.

4. **Integration with Managed Agents.** Does the podcast generation happen inside the Managed Agents session (so it appears in the live stream) or as a post-audit separate call? Leaning post-audit separate to keep Managed Agents focused on the research modules.

5. **Disclosure in audio.** Should the podcast explicitly say at the start "This is an AI-generated podcast"? I'd say yes, for ethical clarity. Adds 5 seconds to runtime.

---

*End of stretch goal doc. Core focus remains PRD and build guide. This is only implemented if Day 4 ends with core complete.*
