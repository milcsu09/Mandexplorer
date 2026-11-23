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

