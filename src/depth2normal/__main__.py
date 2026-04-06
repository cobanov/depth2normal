"""Allow ``python -m depth2normal`` when the package is importable."""

from depth2normal.cli import cli

if __name__ == "__main__":
    cli()
