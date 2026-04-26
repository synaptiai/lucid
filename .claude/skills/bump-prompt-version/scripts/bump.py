#!/usr/bin/env python3
"""Bump a Lucid prompt to its next version, or rehash an existing one.

Usage:
    bump.py <prompt-dir> <current-version-int>            # create v<N+1>
    bump.py <prompt-dir> <current-version-int> --rehash-only  # update hash on v<N>

Example:
    bump.py prompts/module_a 1                # creates v2.md, bumps PROMPT_VERSION
    bump.py prompts/module_a 2 --rehash-only  # recomputes hash on v2.md after body edit
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_text, body_text). Raises if no frontmatter."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    if not m:
        raise ValueError("prompt has no YAML frontmatter")
    return m.group(1), m.group(2)


def update_frontmatter_field(fm: str, key: str, value: str) -> str:
    """Replace `key: ...` in frontmatter. Adds the line if missing."""
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    replacement = f"{key}: {value}"
    if pattern.search(fm):
        return pattern.sub(replacement, fm)
    return fm.rstrip() + f"\n{replacement}"


def write_prompt(path: Path, fm: str, body: str) -> None:
    path.write_text(f"---\n{fm}\n---\n{body}", encoding="utf-8")


def bump_module_constant(prompt_dir: Path, new_version: str) -> Path | None:
    """Update PROMPT_VERSION (or SYNTHESIS_PROMPT_VERSION) in the calling module.

    Returns the touched module path or None if no module mapping applies.
    """
    name = prompt_dir.name
    if name.startswith("module_"):
        letter = name.removeprefix("module_")
        candidates = list((REPO_ROOT / "lucid" / "modules").glob(f"module_{letter}_*.py"))
        if not candidates:
            return None
        target = candidates[0]
        text = target.read_text(encoding="utf-8")
        new_text = re.sub(
            r'^PROMPT_VERSION\s*=\s*"v\d+"',
            f'PROMPT_VERSION = "{new_version}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if new_text == text:
            return None
        target.write_text(new_text, encoding="utf-8")
        return target

    if name == "synthesis":
        target = REPO_ROOT / "lucid" / "synthesis" / "__init__.py"
        text = target.read_text(encoding="utf-8")
        new_text = re.sub(
            r'^SYNTHESIS_PROMPT_VERSION\s*=\s*"v\d+"',
            f'SYNTHESIS_PROMPT_VERSION = "{new_version}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if new_text == text:
            return None
        target.write_text(new_text, encoding="utf-8")
        return target

    return None


def stage(*paths: Path) -> None:
    args = ["git", "-C", str(REPO_ROOT), "add", "--", *[str(p) for p in paths]]
    subprocess.run(args, check=False)


def cmd_bump(prompt_dir: Path, current: int) -> None:
    src = prompt_dir / f"v{current}.md"
    dst = prompt_dir / f"v{current + 1}.md"
    if not src.is_file():
        sys.exit(f"source not found: {src}")
    if dst.exists():
        sys.exit(f"target already exists: {dst} (delete it first if intentional)")

    shutil.copyfile(src, dst)
    text = dst.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    new_version = f"v{current + 1}"
    fm = update_frontmatter_field(fm, "version", new_version)
    fm = update_frontmatter_field(fm, "hash", hashlib.sha256(body.encode("utf-8")).hexdigest())
    write_prompt(dst, fm, body)

    touched_module = bump_module_constant(prompt_dir, new_version)

    print(f"created: {dst.relative_to(REPO_ROOT)}")
    if touched_module:
        print(
            f"updated: {touched_module.relative_to(REPO_ROOT)} → version constant = {new_version}"
        )
        stage(dst, touched_module)
    else:
        print("note: no module-constant mapping for this prompt directory; staged file only")
        stage(dst)


def cmd_rehash(prompt_dir: Path, current: int) -> None:
    target = prompt_dir / f"v{current}.md"
    if not target.is_file():
        sys.exit(f"prompt not found: {target}")
    text = target.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    new_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    fm = update_frontmatter_field(fm, "hash", new_hash)
    write_prompt(target, fm, body)
    print(f"rehashed: {target.relative_to(REPO_ROOT)} → {new_hash[:16]}...")
    stage(target)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    prompt_dir = Path(args[0]).resolve()
    if not prompt_dir.is_dir():
        sys.exit(f"not a directory: {prompt_dir}")
    try:
        current = int(args[1])
    except ValueError:
        sys.exit(f"current-version must be an integer, got {args[1]!r}")

    if "--rehash-only" in args[2:]:
        cmd_rehash(prompt_dir, current)
    else:
        cmd_bump(prompt_dir, current)


if __name__ == "__main__":
    main()
