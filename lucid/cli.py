"""Lucid CLI entry point.

`audit` and `calibrate` are stubs in Phase 1 — wired in Phase 4 / Phase 6.
The stubs are registered now so typer stays in multi-command mode and
`lucid --help` lists both subcommands.
"""

from __future__ import annotations

import typer

from lucid import __version__

app = typer.Typer(
    name="lucid",
    help=f"Lucid {__version__} — epistemic audit for personal AI conversation history.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print Lucid's version and exit."""
    typer.echo(f"lucid {__version__}")


@app.command()
def audit() -> None:
    """Run an audit on a conversation corpus. (Stub — wired in Phase 4.)"""
    typer.secho(
        "lucid audit is not yet implemented (Phase 4 wires this command).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(2)


@app.command()
def calibrate() -> None:
    """Run calibration against labeled data. (Stub — wired in Phase 6.)"""
    typer.secho(
        "lucid calibrate is not yet implemented (Phase 6 wires this command).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(2)


if __name__ == "__main__":
    app()
