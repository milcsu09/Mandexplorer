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
  While frozen the stale `s.image` is not pasted over the whole canvas (that
  overwrites the baked pixels and snaps the view back by the accumulated zoom).
  It is uploaded, but composited only inside the growing render circle
  (horizontal clip slices, radius from `render_items_done`), so the expanding
  circle stays visible. Vibe always shows the live bilinear half-res stream,
  never the late neural result.
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

## Checkpointed orbit (stage 1 of the deep-zoom port)
- `src/orbit.c`/`orbit.h` replace the two `f64[N+1]` arrays. Exact MPFR `z`
  is stored every `ORBIT_STRIDE` (4096) iterations; doubles are regenerated on
  demand by a per-thread window using the **same MPFR ops and precision**, so
  every orbit value is bit-identical to the old representation.
- Scenes with `N <= 2^25` also get an f64 mirror filled during the same MPFR
  pass (zero extra cost), keeping old speed and the GPU path. The 640 MiB
  budget is deliberate: without the mirror `bla_build()` regenerates every
  orbit value through MPFR on **each vibe rebase** (~33.5M MPFR steps per
  frame at 2^25). Above 2^25 only checkpoints are kept (~64 bytes per 4096
  iterations instead of 16 bytes per iteration).
- If the flat mirror is absent the GPU is disabled (its kernel indexes an
  arbitrary orbit position). The flat buffers are reused across regenerations
  so an in-flight async GPU job never reads freed memory.
- `MANDEX_ORBIT_FLAT=0` forces checkpoint-only mode at any N;
  `MANDEX_ORBIT_STRIDE=N` overrides the checkpoint interval (32..1M).
- Stage 2 (FPG): record-min `|z|^2` period candidates are always collected in
  `orbit->periods` (one comparison/iteration). The expensive extended-exponent
  `an`/`bn` guard is opt-in via `MANDEX_FPG=1`; `MANDEX_DUMP_PERIODS=N` prints
  candidates and `fpg_limit`. Groundwork for stage 3 PA, no render change.
- Pixel-identity regression (`--reverse-video` + `MANDEX_DUMP`, 64x48,
  `palettes/big.pal`): default.sce `3759b222515d7334179e8d76fbb53266`,
  deep scene `20261126_071203.sce` `96ac31aeb421f2737a2289edd38d19e3`,
  mid-deep `20261126_002836.sce` `63d350482e6c3d1ba52689a0ef075154`
  (the last two also pass with `MANDEX_ORBIT_FLAT=0`). The 2^25 scene
  `20261126_091423.sce` at 32x24 gives `d520021bc6b466f38ac395d27ca621e9`,
  45 s after the budget fix vs 47 s for the pre-stage-1 binary (>180 s broken).
- This stage removes the memory wall but NOT the O(N) time wall: generating
  checkpoints still walks every iteration, so 10^16 needs stages 3-4
  (rebase + MPA skips) as well.

## First-order PA (stage 3)
- `src/pa.c`/`pa.h`: RFF-style first-order PA, `eps' = an*eps + bn*dc`,
  coefficients as mantissa + binary exponent, with the `PAGenerator` radius
  guard `(epsilon*|2Z| - |bn|*dcMax)/|an|`. Opt-in via `MANDEX_PA=1`,
  `MANDEX_PA_EPS=N` (bits, default 30), `MANDEX_PA_STATS=1`.
- RFF's `MPAView::uLookup()` rejects `refIteration == 0`: PA coefficients from
  `Z_0 = 0` are invalid. Our entries therefore start at BLA-aligned iterations
  >= 32, and store the longest guard-valid length (not a fixed horizon) so
  BLA alignment survives the skip.
- Pixel identity verified on default, mid-deep and deep BLA scenes (MD5s
  unchanged). Enabling PA is currently SLOWER on deep scenes (16M: 143 s vs
  22 s): the linearized eps changes BLA's radius decisions, and the naive
  table is O(entries × skip) (capped at `PA_MAX_ENTRIES` 8192). It stays
  default-off groundwork until stage 4's hierarchical MPA table replaces it.

## Multi-periodic approximation (stage 4)
- `src/pa.c` now also hosts RFF's uncompressed MPA: levels built from the
  record-minimum period candidates (`orbit->periods`), artificial periods
  inserted so neighbours differ by at most `MPA_MAX_MULT^2`, mixed-radix
  `skippable`/`elements` counts, and a closed-form `iteration -> table offset`
  mapper. Table memory is O(levels × periods), not O(N).
- `mpa_build()` walks the reference up to the last period candidate, with the
  same amortized merge loop as RFF's `uGenerateTable` (single-threaded); the
  mapper is recomputed instead of stored, so no per-iteration array is needed.
  `MANDEX_MPA_MAPCHECK=1` verifies it against RFF's running layout.
- Runtime `mpa_try_skip()` runs before PA/BLA in the pixel loop and applies the
  longest guard-valid entry. `radius == -INFINITY` is the "invalid entry"
  sentinel in log2 space (unlike the naive PA, which uses `0`).
- Opt-in: `MANDEX_MPA=1`, `MANDEX_MPA_EPS=N` (bits, default 46),
  `MANDEX_MPA_STATS=1`, `MANDEX_MPA_SELFTEST=1` (compares every readable entry
  with exact quadratic perturbation; worst relative error ~1e-10).
- Validation: deep scene `20261126_071203.sce` is byte-identical to BLA
  (`96ac31...`) and ~6× faster (12 s vs 78 s at 64×48); the boundary test scene
  at 1e-14 differs in <0.1% of pixels at 2^-46. Default-off until the deep
  representation (stage 5) and reference rebasing (stage 6) land.
- `MANDEX_MPA` with `MANDEX_BLA=0` also works on some scenes, but BLA acts as a
  bootstrap on others (BLA moves the perturbation to a phase where MPA entries
  become valid), so keep both enabled for now.

## Iteration ceiling
- `MAX_ITER_MAX` is **33554432** (2^25, was 2^24). It is a hard clamp in
  `apply_iter_max()`, the adaptive controller and the reverse-video generator;
  the `.sce` format and `--iter-max` itself have no such limit.
- The orbit is stored as two `f64` arrays (`16 bytes/iteration`), so 2^25 needs
  ~536 MB just for the orbit plus ~600 MB for BLA. Larger values require
  checkpointing or a disk-backed orbit rather than just raising the constant.

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
- A keyframe is warped instead of a full render for up to 10 consecutive
  frames (`keyframe_max_reuse`, was 5); override with `MANDEX_DIFF_REUSE=N`.

## Reverse video
- `Shift+R` or `--reverse-video TARGET.sce` generates the zoom trajectory
  offline from `--from START.sce` (default `default.sce`) to TARGET (or the
  current view for the key). `--dir in|out` (default out), `--frames N`
  (default 900). No real-time recording; CLI mode exits after the MP4.
- The centre is interpolated in MPFR with the target point anchored:
  `c(t) = c_target - (c_target - c_start) * w(t)/w_start`, width geometric.
- Per-frame `iter_max` follows log-depth (deep frames only get the full N),
  with a bounded probe raising it if pixels are still changing.

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
