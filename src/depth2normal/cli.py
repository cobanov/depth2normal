"""Command-line interface for depth2normal."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from PIL import Image

from depth2normal.converter import METHODS, RANGES, depth_to_normal, load_depth
from depth2normal.filters import Method
from depth2normal.onnx_export import DEFAULT_OPSET, export_onnx


class _DefaultGroup(click.Group):
    """Group that treats a bare path as an argument to ``convert``.

    ``depth2normal depth.png`` keeps working while the subcommands are there
    for anyone who wants them.
    """

    default_command = "convert"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


def _estimate_depth(image_path: str) -> np.ndarray:
    """Run depth estimation, reporting the one-off model download."""
    from depth2normal.estimate import MODEL_BYTES, estimate_depth, model_path

    if not model_path().exists():
        click.echo(f"Downloading the depth model once to {model_path().parent}")
        with click.progressbar(length=MODEL_BYTES, label="depth model") as bar:
            seen = 0

            def progress(downloaded: int, total: int) -> None:
                nonlocal seen
                bar.update(downloaded - seen)
                seen = downloaded

            return estimate_depth(image_path, progress=progress)
    return estimate_depth(image_path)


@click.group(cls=_DefaultGroup)
@click.version_option(package_name="depth2normal")
def cli() -> None:
    """Convert depth maps to normal maps."""


@cli.command()
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
@click.option(
    "--range",
    "depth_range",
    default="auto",
    show_default=True,
    type=click.Choice(RANGES, case_sensitive=False),
    help="How the depth values are scaled before the gradients.",
)
@click.option(
    "--invert",
    is_flag=True,
    help="Flip near and far, for depth maps where bright means far.",
)
@click.option(
    "--estimate",
    "estimate",
    is_flag=True,
    help="Treat the input as a photo and estimate its depth map first.",
)
@click.option(
    "--save-depth",
    type=click.Path(dir_okay=False),
    help="Also write the estimated depth map here (with --estimate).",
)
def convert(
    input_path: str,
    output: str,
    strength: float,
    method: Method,
    sigma: float,
    depth_range: str,
    invert: bool,
    estimate: bool,
    save_depth: str | None,
) -> None:
    """Convert a depth map image to a normal map image."""
    if save_depth and not estimate:
        raise click.UsageError("--save-depth only applies together with --estimate")

    if estimate:
        depth = _estimate_depth(input_path)
        if save_depth:
            Image.fromarray(depth.astype(np.uint8), mode="L").save(save_depth)
            click.echo(f"Depth map saved to {save_depth}")
    else:
        depth = load_depth(input_path)

    normal = depth_to_normal(
        depth,
        strength=strength,
        method=method,
        sigma=sigma,
        depth_range=depth_range,
        invert=invert,
    )
    Image.fromarray(normal, mode="RGB").save(output)
    click.echo(f"Normal map saved to {output}")


@cli.command("estimate")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    default="depth_map.png",
    show_default=True,
    help="Output path for the estimated depth map.",
)
def estimate_command(input_path: str, output: str) -> None:
    """Estimate a depth map from a photo, without converting it."""
    depth = _estimate_depth(input_path)
    Image.fromarray(depth.astype(np.uint8), mode="L").save(output)
    click.echo(f"Depth map saved to {output}")


@cli.command("export")
@click.option(
    "-o",
    "--output",
    default="depth2normal.onnx",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Output path for the ONNX model.",
)
@click.option(
    "-m",
    "--method",
    default="gaussian",
    show_default=True,
    type=click.Choice(METHODS, case_sensitive=False),
    help="Gradient algorithm to bake into the graph.",
)
@click.option(
    "--sigma",
    default=1.0,
    show_default=True,
    type=float,
    help="Gaussian kernel sigma to bake into the graph.",
)
@click.option(
    "--opset",
    default=DEFAULT_OPSET,
    show_default=True,
    type=int,
    help="ONNX opset to target.",
)
def export_command(output: str, method: Method, sigma: float, opset: int) -> None:
    """Export the conversion as a standalone ONNX model."""
    path = export_onnx(output, method=method, sigma=sigma, opset=opset)
    size_kb = Path(path).stat().st_size / 1024
    click.echo(f"ONNX model saved to {path} ({size_kb:.1f} KB)")
    click.echo("Inputs: depth float32 [1,1,H,W] on 0-255, strength float32 scalar")
    click.echo("Output: normal_map uint8 [1,H,W,3] RGB")
