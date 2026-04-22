#!/usr/bin/env bash
# Build a PyInstaller one-file Lucid binary for the current platform.
#
# macOS arm64 is the primary target — this is what Lucid was developed on.
# Linux x64 and Windows x64 builds require running this script on those
# platforms (PyInstaller doesn't cross-compile). The binary is unsigned
# on macOS; users will need to run
#
#   xattr -d com.apple.quarantine ./lucid
#
# to clear the Gatekeeper quarantine flag after downloading.
#
# Usage:
#   bash scripts/build_binary.sh
#
# Output:
#   dist/lucid (platform-specific executable)
#
# Build time: ~45s on M-series macOS. ~60 MB binary.

set -euo pipefail

cd "$(dirname "$0")/.."

# Clean prior build artifacts so PyInstaller's cache doesn't serve stale
# module code after a source change.
rm -rf build dist

uv run pyinstaller \
  --onefile \
  --name lucid \
  --add-data "prompts:prompts" \
  --add-data "lucid/report/templates:lucid/report/templates" \
  --collect-submodules scipy.stats \
  --collect-submodules lucid \
  --noconfirm \
  --clean \
  --distpath dist \
  --workpath build \
  lucid/cli.py

echo ""
echo "Binary built: $(pwd)/dist/lucid"
ls -lh dist/lucid

echo ""
echo "Smoke test:"
./dist/lucid version
