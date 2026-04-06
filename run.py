#!/usr/bin/env python3
"""Run the CLI from a repo clone without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _bootstrap()
    from depth2normal.cli import cli

    cli()


if __name__ == "__main__":
    main()
