<p align="center">
  <img src="assets/hero.webp" alt="A Mars photograph next to the normal map depth2normal produced from it" width="700">
</p>

<p align="center">
  A depth map goes in, a normal map comes out.<br>
  If you do not have a depth map, it estimates one.
</p>

<p align="center">
  <a href="https://pypi.org/project/depth2normal/"><img alt="pypi" src="https://img.shields.io/pypi/v/depth2normal?color=8c8cff&labelColor=1a1a1a"></a>
  <a href="https://github.com/cobanov/depth2normal/actions/workflows/ci.yml"><img alt="ci" src="https://img.shields.io/github/actions/workflow/status/cobanov/depth2normal/ci.yml?branch=main&color=8c8cff&labelColor=1a1a1a"></a>
  <img alt="tests" src="https://img.shields.io/badge/tests-71-8c8cff?labelColor=1a1a1a">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-8c8cff?labelColor=1a1a1a">
  <a href="LICENSE"><img alt="licence" src="https://img.shields.io/badge/licence-MIT-8c8cff?labelColor=1a1a1a"></a>
</p>

---

Turning depth into normals is a small piece of arithmetic: two gradients, a
vector, one normalisation. The parts that are usually missing sit on either
side of it. You need a depth map before you can start, and you need Python at
the other end to run the result. This package covers both.

```sh
pip install depth2normal
depth2normal depth.png -o normal.png
```

- **No depth map, no problem.** `--estimate` runs Depth Anything V2 Small on
  the photo first, so a photograph is a valid input.
- **The conversion exports to ONNX**, in about 1.7 KB, so it runs in a browser,
  in C#, in Unity, in a ComfyUI node, anywhere ONNX Runtime goes.
- **The same input gives the same answer.** An 8-bit and a 16-bit copy of one
  depth map now produce identical normals. Before 2.0 they did not, and the
  16-bit one was wrong.
- **Three runtime dependencies**: NumPy, Pillow and Click. No SciPy, no OpenCV.
- **A browser demo in `web/`**, no build step, nothing uploaded.

## Install

```sh
pip install depth2normal              # the converter
pip install 'depth2normal[all]'       # plus ONNX export and depth estimation
```

`uv add depth2normal` works the same way. The extras are separable:
`[onnx]` pulls in `onnx` for the export, `[estimate]` pulls in `onnxruntime`
for depth estimation. The converter itself runs on Python 3.10, but ONNX
Runtime stopped publishing 3.10 wheels, so depth estimation needs 3.11 or
newer.

## Use

```sh
depth2normal depth.png -o normal.png              # the common case
depth2normal depth.png -s 3 -m scharr             # stronger relief, sharper filter
depth2normal depth.png --invert                   # for maps where bright means far
depth2normal photo.jpg --estimate -o normal.png   # no depth map needed
depth2normal estimate photo.jpg -o depth.png      # just the depth map
depth2normal export -o depth2normal.onnx          # just the graph
```

| Option | Default | What it does |
| --- | --- | --- |
| `-o`, `--output` | `normal_map.png` | Where the normal map goes |
| `-s`, `--strength` | `1.0` | Gradient multiplier, so how pronounced the relief is |
| `-m`, `--method` | `gaussian` | `gaussian`, `sobel` or `scharr` |
| `--sigma` | `1.0` | Gaussian kernel width, smoothness against detail |
| `--range` | `auto` | `auto`, `minmax` or `raw`, see [How it works](#how-it-works) |
| `--invert` | off | Flip near and far |
| `--estimate` | off | Treat the input as a photo and estimate its depth first |
| `--save-depth` | | Keep the estimated depth map as well |

```python
import depth2normal

depth2normal.convert("depth.png", "normal.png", strength=2.0, method="scharr")

depth = depth2normal.estimate_depth("photo.jpg")     # needs [estimate]
normal = depth2normal.depth_to_normal(depth, strength=3.0)

depth2normal.export_onnx("depth2normal.onnx")        # needs [onnx]
```

## From a photo

<p align="center">
  <img src="assets/pipeline.webp" alt="A photograph, the depth map estimated from it, and the resulting normal map" width="700">
</p>

`--estimate` runs [Depth Anything V2 Small](https://huggingface.co/onnx-community/depth-anything-v2-small)
(Apache-2.0, 99 MB) through ONNX Runtime. The model is downloaded once,
checksummed, and cached in `~/.cache/depth2normal`, or wherever
`DEPTH2NORMAL_CACHE` points. A 640x480 photo takes about 230 ms on an M4 Pro,
on the CPU, with no GPU involved.

The estimate is relative depth, not metres. That is exactly what a normal map
needs, and it is not what a measurement needs.

## Anywhere ONNX Runtime goes

```sh
depth2normal export -o depth2normal.onnx --method gaussian --sigma 1
```

| | |
| --- | --- |
| Input `depth` | float32 `[1, 1, H, W]`, on the 0-255 range, height and width dynamic |
| Input `strength` | float32 scalar, so relief is tunable without re-exporting |
| Output `normal_map` | uint8 `[1, H, W, 3]`, RGB, ready for a canvas or an image file |
| Size | 1.1 KB for `sobel`, 6.1 KB for `gaussian` at sigma 3 |

The method and sigma are baked into the convolution weights at export time,
because they are the kernel. Strength is not, so one exported file covers every
strength.

The graph and the NumPy path agree to within **one level out of 255**, on at
most 3% of pixels, which is float32 landing on the other side of a rounding
boundary from float64. `tests/test_onnx_export.py` asserts exactly that.

## In the browser

`web/` is a single page that loads the exported graph and runs the whole thing
client side. No build step, no bundler:

```sh
cd web && python3 -m http.server
```

Drop a depth map in and the conversion is a few milliseconds of WebAssembly.
Drop a photograph in and press **Estimate depth**: the quantised 27 MB model is
fetched from Hugging Face and runs in the tab too, about 6.5 seconds for a
640x480 photo. Nothing is uploaded in either case.

`tools/build_web_models.py` regenerates the six models the page offers.

## How it works

1. Bring the depth values onto a fixed 0-255 range (see the table below).
2. Estimate `dx` and `dy` with a separable derivative filter, reflecting at the
   border.
3. Build `(-dx * strength, -dy * strength, 1)`, normalise it to unit length,
   and map it to 8-bit RGB.

Step 1 is the one that matters, and it is why the same depth map at different
bit depths used to give different answers:

| `--range` | What it does | When |
| --- | --- | --- |
| `auto` | Divides by the range the values look like they came from: 0-1 float, 8-bit, or 16-bit | The default. Consistent across the frames of a sequence |
| `minmax` | Stretches this image's own min and max to 0-255 | Low contrast maps, and metric depth in metres |
| `raw` | Leaves the values alone | Reproducing pre-2.0 output |

The gradient filters:

| Method | Quality | 2048x1152 | Notes |
| --- | --- | --- | --- |
| `gaussian` | Best | 60 ms | Gaussian derivative. `--sigma` trades smoothness against detail |
| `sobel` | Good | 41 ms | Classic 3x3. Sharp, but staircases on quantised depth |
| `scharr` | Good | 42 ms | Better rotational accuracy than Sobel, same speed |

All three are separable, which is what lets the package drop SciPy and still
match it: the same measurement on SciPy's `gaussian_filter` is 57 ms, and its
wheel is 19.5 MB you no longer download.

## What changed in 2.0

**Bit depth no longer changes the result.** Gradients were computed on the raw
pixel values, so a 16-bit depth map produced gradients 256 times larger than
the 8-bit copy of the same surface, and `strength` meant something different in
each file. Measured on `assets/depth.png`, converted at the defaults:

| | z average | Pixels tilted more than 30 degrees |
| --- | --- | --- |
| 1.0, 8-bit | 249.5 | 6.5% |
| 1.0, 16-bit | 197.9 | 48.5% |
| 2.0, either | 249.5 | 6.5% |

The two 2.0 rows are not merely close. They are the same bytes.

**Also in 2.0:** ONNX export, depth estimation from a photo, `--invert`,
`--range`, the browser demo, and SciPy removed. The CLI grew subcommands
(`estimate`, `export`) while `depth2normal depth.png` keeps working exactly as
it did. Borders now reflect rather than repeat the edge pixel, which is what
makes the exported graph match the NumPy path exactly.

## Measured

Apple M4 Pro, Python 3.14, NumPy 2.4, ONNX Runtime 1.29, on a 2048x1152 depth
map, median of seven runs.

| | NumPy | ONNX Runtime |
| --- | --- | --- |
| `gaussian`, sigma 1 | 60 ms | 13 ms |
| `gaussian`, sigma 3 | 113 ms | 70 ms |
| `sobel` | 41 ms | 5 ms |
| `scharr` | 42 ms | 5 ms |

The exported graph is the faster way to run this, by roughly 5x on the small
kernels. ONNX Runtime is doing the convolution the way convolutions are meant
to be done, and NumPy is making thirty-six shifted passes over a 2.4 megapixel
array.

## Development

```sh
uv sync --extra all
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python tools/build_web_models.py
```

## Licence

MIT. The depth model is Apache-2.0 and is downloaded, not vendored. Image
credits are in [assets/CREDITS.md](assets/CREDITS.md).
