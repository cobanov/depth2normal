"""Command-line interface for depth2normal."""

from __future__ import annotations

import click

from depth2normal.converter import METHODS, convert


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    default="normal_map.png",
    show_default=True,
    help="Output path for the generated normal map.",
)
@click.option(
    "-s",
    "--strength",
    default=1.0,
    show_default=True,
    type=float,
    help="Gradient multiplier controlling normal intensity.",
)
@click.option(
    "-m",
    "--method",
    default="gaussian",
    show_default=True,
    type=click.Choice(METHODS, case_sensitive=False),
    help="Gradient algorithm.",
)
@click.option(
    "--sigma",
    default=1.0,
    show_default=True,
    type=float,
    help="Gaussian kernel sigma (only for --method gaussian).",
)
@click.version_option(package_name="depth2normal")
def cli(
    input_path: str,
    output: str,
    strength: float,
    method: str,
    sigma: float,
) -> None:
    """Convert a depth map image to a normal map image."""
    convert(input_path, output, strength=strength, method=method, sigma=sigma)
    click.echo(f"Normal map saved to {output}")
