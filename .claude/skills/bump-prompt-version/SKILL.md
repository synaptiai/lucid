---
name: bump-prompt-version
description: Iterate a Lucid prompt to its next version. Copies prompts/<name>/v<N>.md → v<N+1>.md, recomputes the sha256 hash field, bumps the PROMPT_VERSION constant in the corresponding module, and stages both files. Use when the user asks to "bump prompt", "iterate the module X prompt", or "ship a new version of the synthesis prompt".
disable-model-invocation: true
---

# bump-prompt-version

Mechanical workflow for iterating a Lucid prompt to its next version. Encapsulates the 5-step process documented in CLAUDE.md "Iterating a prompt" so the SHA never drifts and the module constant always matches the file on disk.

## Usage

User invokes via `/bump-prompt-version <prompt-name>` where `<prompt-name>` is one of:

- A module letter: `a`, `b`, `c`, `d`, `e`, `f`, `h` → operates on `prompts/module_<letter>/`
- A flat prompt name: `synthesis`, `synthesis_validator` → operates on `prompts/<name>/`

## Steps

1. **Resolve the prompt directory.** Try `prompts/module_<arg>/` first, fall back to `prompts/<arg>/`. If neither exists, exit with a list of valid options.
2. **Find the highest existing version.** `ls v*.md`, parse the integer suffix, pick max → call this `N`.
3. **Run the bump script:** `python .claude/skills/bump-prompt-version/scripts/bump.py <prompt-dir> <N>`. The script:
   - Copies `v<N>.md` → `v<N+1>.md` byte-for-byte.
   - Updates the new file's frontmatter: `version: v<N+1>` and recomputes `hash:` against the body (which is unchanged at this point — the user edits the body next).
   - For module prompts (`module_<letter>`), updates `PROMPT_VERSION = "v<N+1>"` in `lucid/modules/module_<letter>_*.py`.
   - For `synthesis`, updates `SYNTHESIS_PROMPT_VERSION` in `lucid/synthesis/__init__.py`.
   - Stages both files with `git add`.
4. **Tell the user what to do next** (verbatim):

   > Created `v<N+1>.md` from `v<N>.md`. Now:
   >
   > 1. Edit the body of `v<N+1>.md` with your changes.
   > 2. Update the `## Changelog` section explaining what changed.
   > 3. Re-run the hash bump: `python .claude/skills/bump-prompt-version/scripts/bump.py <prompt-dir> <N+1> --rehash-only`
   > 4. If the module has a calibration target, run `uv run lucid calibrate --module <letter>` and verify κ/α didn't regress.
   > 5. `git commit` with a message like `prompts(<module>): v<N+1> — <one-line summary>`.

## Notes

- **Never edit a published prompt version in place.** The `protect_files.py` PreToolUse hook enforces this; this skill is the sanctioned path.
- **The bump script does not edit the body.** That's your job — the script only sets up the next version's scaffolding and bumps the version field.
- **`--rehash-only` is the second pass.** After you edit the body, re-run with `--rehash-only` to update only the `hash:` field. Don't touch the hash by hand; sha256 typos are silent prompt-cache killers.
- The script is idempotent: re-running step 3 on an unchanged body is a no-op.
