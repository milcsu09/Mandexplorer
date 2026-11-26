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
- Zoom defaults to 1.02 per frame (`--vibe-step <float>` overrides it; lower =
  smoother preview, slower zoom). Final quality (`iter_max`/SSAA/resolution)
  is unchanged.
- The view shown by the current frame is kept in MPFR (`disp_base_*`), as is
  the view its successor is rendered from (`render_base_*`). The scene is
  rebased at render start, and pan/scale are recomputed from the exact views
  when a frame completes, so the visible zoom never snaps backwards.
- `gpu_job_active` suppresses texture uploads while a GPU frame is being written.
- `orbit_ready` stays true after a render, so queueing `compute_orbit` must
  clear it first; otherwise the next tick starts render workers while
  `compute_orbit` rebuilds `bla_table` (use-after-free in `bla_try_skip`).

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
  During vibe the FSRCNN frame is deliberately **not** displayed: it arrives
  ~1/s while the camera commits ~13/s, which snapped the view backwards at
  every commit. Instead the freshly completed half-res frame is bilinearly
  upscaled (`preview_render_gen`) immediately and the display's pan/scale
  transform zooms it smoothly between frames. `preview_active()` is false
  while recording/video rendering.
  Preview texture uploads are skipped when the displayed pixels did not change
  (`preview_upload_gen`), removing a full-screen 60 Hz copy.
- `--fullscreen` sizes the window to the desktop display mode (useful with
  `--preview` so the half-res render still has detail).
- Never use preview output as final output: it invents detail.

## Notes
- `terminate()` called before state changes requiring recompute.
- `apply_iter_max()` reallocates orbit buffers.
