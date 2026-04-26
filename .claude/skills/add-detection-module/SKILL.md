---
name: add-detection-module
description: Scaffold a new Lucid detection module (Module I, J, ...). Templates the module file from module_g_attribution.py, adds the ModuleName enum entry, registers in the scoring loop AND the orchestrator dispatch table, and creates prompts/module_<letter>/v1.md. Use when the user asks to "add a new module" or "scaffold module X".
disable-model-invocation: true
---

# add-detection-module

Scaffolds the wiring for a new Lucid detection module. CLAUDE.md "Adding a new detection module" lists 7 mechanical steps; the dual registration in `lucid/run.py::_run_scoring_loop` and `lucid/orchestrator/tools.py::invoke_module_for_run` is easy to forget — this skill keeps them in sync.

## Args from the user

Ask for these if the user didn't supply them:

1. **Letter** — single uppercase letter, the next free one (currently used: A, B, C, D, E, F, G, H).
2. **Snake-case name** — short identifier, used in the file name and citation constant. Example: `framing_bias`.
3. **Citation** — the paper backing the module, formatted per the `CITATION_*` constants at the top of any existing module file.
4. **One-line purpose** — what behavior this module detects.

## Steps

For letter `X` and name `<name>`:

1. **Module file.** Copy `lucid/modules/module_g_attribution.py` → `lucid/modules/module_<x>_<name>.py`. Replace:
   - `CITATION_TIMELINE` → `CITATION_<X>_<NAME_UPPER>` and update the value to the new citation.
   - All references to `Module G` / attribution behaviors with the new module's behaviors. Leave a `# TODO(claude)` for the actual scoring logic.
   - Set `PROMPT_VERSION = "v1"`.
   - Update the module-level docstring with the citation link.

2. **Schema enum.** Edit `lucid/schemas.py`:
   - Locate the `ModuleName` `StrEnum`. Append `MODULE_<X> = "module_<x>"` in alphabetical position.
   - If the module produces a new behavior taxonomy, add a new `StrEnum` for those labels (follow the pattern of `SpiralBenchBehavior`, `MemorySupport`, etc.).

3. **Prompt scaffold.** Create `prompts/module_<x>/v1.md`. Use the same frontmatter shape as `prompts/module_a/v1.md`:
   ```yaml
   ---
   version: v1
   module: <X>
   model: claude-opus-4-7        # or claude-sonnet-4-6 if extraction-only
   thinking_mode: adaptive       # or 'disabled' for trivial classification
   effort: high                  # ladder: low | medium | high | xhigh | max
   citation: "<full citation>"
   purpose: "<one-line>"
   hash: "<sha256 placeholder — bump after writing body>"
   ---
   ```
   Body should include `## Output Schema` and `## Changelog` sections. After the body is finalised, run:
   ```
   python .claude/skills/bump-prompt-version/scripts/bump.py prompts/module_<x> 1 --rehash-only
   ```
   to seal the hash.

4. **Scoring-loop registration.** Edit `lucid/run.py`:
   - Find `_run_scoring_loop`. Add the new module to the enabled-modules list. Match the pattern used by Module G (or by Module D for default-on-with-flag-to-disable).
   - If the module needs a CLI flag, follow Module D's `include_module_d` pattern in `lucid/cli.py`.

5. **Orchestrator dispatch.** Edit `lucid/orchestrator/tools.py`:
   - Find `invoke_module_for_run` (or the dispatch table near it). Add a case for `ModuleName.MODULE_<X>` calling the new module's `run(...)` function.

6. **Cost profile.** Edit `lucid/cost.py`:
   - Find `MODULE_PROFILES`. Add an entry for the new module with `model`, `output_tokens_per_conv`, and `cache_hit_rate` (default 0.50 unless the system prompt is heavily cached).

7. **Test scaffold.** Create `tests/test_module_<x>_<name>.py`. Mirror the structure of `tests/test_module_g_attribution.py`. Tests must use a mocked Anthropic client — never make real LLM calls in pytest.

## After scaffolding

Tell the user (verbatim):

> Scaffolded Module <X> (`<name>`). Next:
>
> 1. Implement the actual scoring logic in `lucid/modules/module_<x>_<name>.py` (search for `# TODO(claude)`).
> 2. Write the prompt body in `prompts/module_<x>/v1.md`, then seal the hash with `bump.py --rehash-only`.
> 3. Add tests with mocked LLM responses; run `uv run pytest tests/test_module_<x>_<name>.py -v`.
> 4. Run `uv run mypy lucid/ --strict` and `uv run ruff check lucid/` cleanly.
> 5. Smoke-test integration: `uv run lucid audit --source claude-ai --path ./demo/corpus --sample 5 --dry-run` should show the new module in the cost breakdown.
> 6. If the module has a calibration target, build a fixture and add a `lucid calibrate --module <x>` path under `lucid/calibration/`.

## Notes

- The skeleton lives in `module_g_attribution.py` because it's the simplest (deterministic, no LLM). For LLM-backed modules, also copy patterns from `module_a_spiralbench.py` (prompt loading, judge ensemble, finding emission).
- Don't forget to update CLAUDE.md's "Architecture map" — the new module deserves a one-line entry.
- Citation constants live at the top of each module file. Use the `CITATION_*` naming convention; never inline a citation string into a `Finding(...)` call.
