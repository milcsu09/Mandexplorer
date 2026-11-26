# Mandexplorer

Mandexplorer is an application for exploring the [Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set).

## Prerequisites

- [SDL2](https://www.libsdl.org/)
- [SDL2_ttf](https://github.com/libsdl-org/SDL_ttf)
- [GNU MPFR](https://www.mpfr.org/)

## Build

```bash
make
```

## Usage

```bash
./a default.sce
```

OR

```bash
./a --help
```

Press `F1` inside the running program for overlay help.

## Reverse video (offline zoom)

No real-time capture needed: zoom to the target view, save it (`s`), and let
the renderer generate the whole MP4 from two scenes:

```bash
# zoom out from the saved deep scene to default.sce
./a default.sce --reverse-video target.sce --dir out --frames 900

# zoom in from default.sce to the saved deep scene
./a default.sce --reverse-video target.sce --dir in

# continue from a scene saved mid-zoom
./a saved_mid.sce --from saved_mid.sce --reverse-video target.sce --dir in
```

`--from` defaults to `default.sce`, `--dir` to `out`, `--frames` to 900 (30 s
at 30 fps). Shift+R does the same using the **current view** as the target and
`--from` (or `default.sce`) as the start. The generated trajectory keeps the
target point anchored on screen; iteration count grows with log-depth (17M
only on the deepest frames) and a probe raises it if pixels are still moving.
Frames are rendered full-size with SSAA=2, same as recorded video.

## Deep zoom acceleration (BLA)

For `width <= 1e-12` the renderer automatically enables a second-order
bilinear approximation of the reference orbit. Coefficients use a double
mantissa plus an explicit binary exponent, so deep-zoom frames render about
18–20x faster while staying byte-identical to the classic perturbation path
(see `bla_results.txt`). Resolution, SSAA and iteration counts are unchanged.

- `MANDEX_BLA=0` — force the classic perturbation path.
- `MANDEX_BLA_MARGIN=<n>` — log2 safety margin (default 16).
- `MANDEX_NO_AUTO=1`, `MANDEX_AUTO_FRAMES=<n>`, `MANDEX_DUMP=<prefix>`,
  `MANDEX_DUMP_LIMIT=<n>`, `MANDEX_DIFF_REUSE=<n>` — A/B measurement hooks.

The GPU/OpenCL path is skipped while BLA is active: PoCL on the development
host cannot compile the extended-exponent kernel, so rendering uses the CPU
worker pool.

## Neural preview (experimental)

`tools/nn_preview.py` upscales a dumped frame x2 with a tiny FSRCNN ONNX
model (`models/FSRCNN_x2.onnx`, 37 KB). It is intended **only** for a fast
interactive preview, never for final output: a neural network invents detail
that was not computed, which is unacceptable for a mathematical fractal.

```bash
python3 -m venv .venv-preview
.venv-preview/bin/pip install numpy opencv-contrib-python-headless
.venv-preview/bin/python tools/nn_preview.py in.ppm out.ppm
```

In-app preview uses the same model:

```bash
make
./a --preview --fullscreen my_scene.sce
# or pick the size explicitly:
./a --preview -w 1600 -h 900 my_scene.sce
```

The window stays at the requested size; the fractal is computed at half
resolution and upscaled with FSRCNN before display. Upscaling runs on its own
thread, so navigation never freezes: a cheap bilinear preview of the half-res
buffer is shown until the neural result arrives (~1 s at 800x600, ~3.5 s at
1600x900). This is for navigation only.
The model comes from `opencv_extra` (converted from the Apache-2.0 FSRCNN
TensorFlow model) and is included as `models/FSRCNN_x2.onnx`.

