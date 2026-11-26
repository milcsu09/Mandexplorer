# Mandexplorer

C/SDL2 Mandelbrot explorer. Single binary built from `src/*.c`.

## Build
- `make` (output `a`). Requires `gcc`, SDL2, SDL2_ttf, mpfr.
- No tests configured.

## Source layout
- `src/main.c` — event loop, rendering pipeline, period detection, UI overlay (Alt toggles info).
- `src/scene.c`/`scene.h` — scene state, iter_max handling.
- `src/color.c`, `src/image.c`, `src/palette.c`, `src/common.c`, `src/thread_pool.c`.

## Key bindings (see src/main.c)
- Alt: toggle info overlay. F1/h: help. v: vibe zoom. i: adaptive iterations.
- b: interior color. m/o: Mandelbrot/Omnibrot. [ / ]: omni a.
- x: reset view. c: center on mouse.
- Insert/Delete: bailout ±. Home/End: iter_max ±. PgUp/PgDn: iter per color.
- s: save .sce. 1/2/4/8: SSAA.
- r: record video (press to start, press again to stop and render mp4).
- `.` / `,`: period detection tolerance up/down (1e-61 disables → "off").

## Period detection
- Threshold in `main.c` (~line 904): `period_tol > 1e-61 && ...`. Default 1e-40. Tunable via global `period_tol`.

## Vibe zoom
- The display thread only presents the last complete texture; orbit and GPU/CPU rendering run outside it.
- A render is never killed by a frame counter. When it completes, `PHASE_COMPUTE_IMAGE` sets `redraw` and commits the accumulated view.
- Zoom keeps the legacy 1.02 per frame; final quality (`iter_max`/SSAA/resolution) is unchanged.
- `bake_display()` (ported from `mandexplorer.goose`) resamples the image that
  is actually shown with the pending transform at every rebase, so the CPU
  pixels and the SDL canvas agree with the rebased scene. Sampling coords
  match the SDL `vr` transform (no Goose half-pixel offset).
- `canvas_frozen` is set when the SDL texture transform is baked into
  `texture_canvas` at a rebase and cleared when the next full frame is ready.
  While frozen the stale `s.image` is **not** uploaded (otherwise it overwrites
  the baked pixels and the view snaps back by the accumulated zoom). Vibe
  always shows the live bilinear half-res stream, never the late neural result.
- `gpu_job_active` suppresses texture uploads while a GPU frame is being written.
- `orbit_ready` stays true after a render, so queueing `compute_orbit` must
  clear it first; otherwise the next tick starts render workers while
  `compute_orbit` rebuilds `bla_table` (use-after-free in `bla_try_skip`).
- `compute_orbit()` uses a **dynamic MPFR precision** (`orbit_precision()`):
  `-log2(width) + log2(iter_max) + 64`, clamped to [160, FLOAT_PREC].  Fixed
  2048 bits made the orbit loop dominate (2.5 s per 262144 iterations vs
  0.4 s at 182 bits); view centres, recording and period detection keep
  `FLOAT_PREC` so they are unaffected.
- The reference-orbit delta is computed from the exact MPFR centres, not as a
  double subtraction (which quantises to ~2e-16 and is useless at deep zoom).
  In vibe only, the orbit is reused while the cached centre stays within half
  a view width (`ORBIT_REUSE_DEEP_EPS`) and BLA is rebuilt for the new delta;
  final renders keep the strict check so they stay pixel-identical.

## BLA acceleration
- `src/bla.c`/`bla.h`: second-order bilinear approximation of the reference
  orbit, with double mantissa + explicit binary exponent for coefficients.
  Coefficients overflow plain `double` at depth (~10^700).
- Enabled automatically for `width <= 1e-12`; disable with `MANDEX_BLA=0`,
  tune the log2 safety margin with `MANDEX_BLA_MARGIN` (default 16).
- Uses the CPU renderer: `gpu_init()` is skipped while BLA is enabled because
  PoCL cannot compile the extended-exponent kernel on this host.
- Debug hooks: `MANDEX_AUTO_FRAMES`, `MANDEX_DUMP`, `MANDEX_DUMP_LIMIT`,
  `MANDEX_DIFF_REUSE`, `MANDEX_NO_AUTO`.

## Recording
- `r` starts/stops capturing views; the confirmation `r` triggers the mp4 render.
- Duplicate detection compares against the previous view with a **sub-pixel**
  tolerance (`width / W * 1e-3`). A fixed absolute tolerance used to freeze
  recording at ~1300 frames once deep zoom made 2% steps smaller than it.
- Capture buffer grows 256 -> 65536 views; a full buffer prints a message and
  renders immediately rather than silently dropping frames.
- Each captured view stores its centre as **MPFR**, not a double. A double
  centre near -1.75 has ~2e-16 resolution; once the zoom is deeper than that
  (~frame 1500 from width 0.01 at 1.02/frame) the recorded target quantises to
  neighbouring ulps, so the video zooms into the wrong point and keyframe
  warps oscillate. `view_offset_px()` also works from the exact centres, and
  `MANDEX_DUMP_VIEWS=1` dumps all recorded centres to
  `/tmp/mandexplorer_views.txt` when rendering starts.
- Preview is suspended for the whole recording/video render: frames are
  rendered at full `s.w x s.h` into `s.image`, never through FSRCNN.

## Neural preview
- `--preview`: fractal is computed into `preview_render_image` at half of
  `s.w x s.h`, then a dedicated worker thread pipes it through
  `tools/nn_preview.py` (FSRCNN x2, `models/FSRCNN_x2.onnx`) into `s.image`.
- `s.image` stays full-size for the texture, video and UI; the GPU path is
  disabled in preview mode while BLA still accelerates the half-res render.
- Display never blocks: `preview_display_image()` shows a live bilinear upscale
  of the half-res buffer until the worker publishes the neural result.
  At most one upscale job is in flight; newest completed frame wins.
  After the first neural result the last complete frame stays on screen until
  the next one arrives (no flicker back to the partial bilinear image), and
  the half-res buffer is cleared at every commit so vibe frames do not leave
  stale pixels. `preview_active()` is false while recording/video rendering.
- `--fullscreen` sizes the window to the desktop display mode (useful with
  `--preview` so the half-res render still has detail).
- Never use preview output as final output: it invents detail.

## Notes
- `terminate()` called before state changes requiring recompute.
- `apply_iter_max()` reallocates orbit buffers.
