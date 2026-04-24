"""Synthesis session: Managed Agents narrative writer.

Runs *after* the deterministic scoring phase. Reads the findings table,
spot-reads the corpus via read-only custom tools, writes narrative
sections of the HTML report with inline ``[F:finding_id]`` citation
tokens. A Sonnet 4.6 post-processor structures the prose into
validated blocks; uncited or invalid prose is dropped at render time.

See docs/plans/2026-04-24-synthesis-agent-refactor.md for the design.
"""

from lucid.synthesis.handler import HeartbeatMonitor, dispatch_tool_call
from lucid.synthesis.session import (
    MANAGED_AGENTS_BETA_HEADER,
    SynthesisConfig,
    SynthesisHandles,
    SynthesisOutcome,
    SynthesisSession,
)
from lucid.synthesis.tools import build_synthesis_registry

# Writer-prompt version key. Keyed to ``prompts/synthesis/<version>.md``
# *and* to the Managed Agents lifecycle name
# (``lucid-synthesis-v<SYNTHESIS_PROMPT_VERSION>``) so a prompt bump
# forces a new agent. Bump in lockstep with the prompt file's
# ``version`` frontmatter when shipping a prompt revision.
SYNTHESIS_PROMPT_VERSION = "v1"

__all__ = [
    "MANAGED_AGENTS_BETA_HEADER",
    "SYNTHESIS_PROMPT_VERSION",
    "HeartbeatMonitor",
    "SynthesisConfig",
    "SynthesisHandles",
    "SynthesisOutcome",
    "SynthesisSession",
    "build_synthesis_registry",
    "dispatch_tool_call",
]
