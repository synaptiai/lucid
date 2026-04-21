"""Lucid detection modules.

Every module follows the contract defined in :mod:`lucid.modules.base`:

- Produces :class:`~lucid.schemas.Finding` objects with full provenance.
- Cites its source paper as a module-level constant (``CITATION_*``).
- Loads prompts from ``prompts/module_<letter>/<version>.md`` — never from
  hard-coded strings. Version is exposed as ``PROMPT_VERSION``.
- Handles per-conversation errors gracefully: one bad conversation returns a
  :class:`~lucid.modules.base.ModuleError`, it does not crash the module.
"""
