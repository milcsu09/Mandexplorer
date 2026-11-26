# Mandexplorer → Goose: technical documentation and port plan (no C code retained)

Goal: re-implement Mandexplorer entirely in Goose, replacing all C/SDL2/OpenCL
source. The only allowed native dependencies are system libraries reached through
Goose `extern fn` (e.g. libmpfr) if the precision upgrade path is taken; no
Mandexplorer C source is kept.

## 0. Baseline decisions

1. Language/runtime: Goose, AOT build (`goose build`) because threads are required.
2. Graphics: Goose `gfx` module (SDL3 GPU) instead of SDL2.
3. Text/UI: Goose `ui` module (Nuklear) instead of SDL_ttf.
4. Concurrency: Goose `thread_fn` + typed queues; no shared mutable memory.
5. Precision: **f64-first port**. Original uses 2048-bit mpfr only for the
   reference orbit; porting that machinery to Goose would require either a
   custom big-float implementation or an extern binding to libmpfr. f64-first
   keeps the port pure Goose and matches Goose's native math. Deep-zoom
   precision becomes the known limitation (see §10).
6. GPU: optional compute shader via `gfx` compute pipelines; CPU path is the
   correctness reference. OpenCL is dropped.
7. File formats: preserve `.sce` (v1) and `.pal` (v1) text formats for
   compatibility with existing scenes/palettes.

## 1. Subsystem mapping (original → Goose)

| # | Original subsystem | Goose implementation |
|---|---|---|
| 1 | Build (`make`, gcc, SDL2, TTF, mpfr, OpenCL) | `goose build mandexplorer.goose` (AOT). Link via `goose --gfx-link`. No C files in repo. Dev runs may use `goose run` without threads. |
| 2 | `struct scene` (mpfr center, doubles, iter counts, palette path) | `struct Scene { c_re: f64, c_im: f64, width: f64, bailout: f64, lx,ly,lz: f64, iter_max: u64, iter_per_color: u64, palette_path: u8[..] }`. Root-owned inside `AppState`. |
| 3 | `struct palette` (flexible array) | `Palette { size: usize, data: u32[] }` (frozen array). Load/save via os. |
| 4 | `struct image` (atomic pixels) | `Image { w: i32, h: i32, pixels: u32[] }`. No atomics needed (single writer assembly); workers return rows. |
| 5 | Thread pool (SDL mutex/cond, function pointers) | One `thread_fn` worker per CPU (fixed count, default 8). Two typed queues: `JobQueue: Job[]`? Goose queues are typed per type; use `qput<Job>`/`qget<Job>` on a global queue variable declared in worker program. Jobs are flat structs. Results via `qput<Result>` on a result queue. |
| 6 | Mandelbrot perturbation render | Pure Goose functions `mandelbrot(...)`, `mandelbrot_color`, `mandelbrot_shade`. Same formulas, f64. |
| 7 | Omnibrot | Pure Goose `omnibrot(...)` with `complex_exp` Taylor path. |
| 8 | Reference orbit (mpfr) | f64 orbit loop (or extern mpfr handle layer, optional). Cache fields in AppState. |
| 9 | Orbit reuse/cache | Same heuristic (exact equality or small relative delta). |
| 10 | Period detection | Same comparison with `period_tol`; `period_tol` becomes Scene/AppState field (persisted? add to .sce? keep global default for compat). |
| 11 | Adaptive iterations (probe + controller) | Same logic in Goose; probe runs on main thread or a single worker with deadline via os clock. |
| 12 | GPU/OpenCL | Optional `gfx` compute pipeline; CPU fallback. See §9. |
| 13 | Vibe zoom | Same zoom-toward-cursor, refresh every N frames, SSAA disable. |
| 14 | Video recording | os subprocess? Goose has no subprocess API (omission). Alternative: write raw frames to a FIFO/pipe? Goose os has file I/O; can spawn ffmpeg via shell? No process API. Options: (a) write raw rgb24 to a file and instruct user to run ffmpeg; (b) add a tiny extern to `popen` (C library) — still no repo C, but FFI to libc. Recommend (b) via `extern fn` to libc `popen`/`pclose`/`fwrite`? Or implement in Goose using file writes + external command documented. We'll use extern to libc (allowed system lib) for pipe. |
| 15 | UI overlays (info/help/REC/rendering) | `ui` module immediate-mode widgets + `gfx` text; or `gfx` direct text? ui provides text. Draw lines as labels. |
| 16 | Input & view transform | `gfx` input API; pan/zoom math identical. |
| 17 | Window/resize/textures | `gfx` window + CPU image buffer; no textures needed for display? Use `gfx::update_texture` or `gfx::screen()`? Simpler: render to an RGBA texture each frame via `update_texture` from u32 buffer, or draw fullscreen quad. `gfx` screen texture is RGBA8; we can update a texture. |
| 18 | Save scene | os write_file_atomic with formatted text. |

## 2. Proposed Goose module layout

Single program file `mandexplorer.goose` (or split via `import`):
- `mandexplorer.goose` — main, app state, event loop, rendering phases.
- `mathx.goose` — mandelbrot/omnibrot/probe (optional package).
- `iofmt.goose` — scene/palette parse/format.
- `worker.goose` — `thread_fn worker()` and job/result types.

Goose whole-program compile allows multiple files via `import` from local dir.

## 3. Data structures (Goose)

```goose
struct Point { x: i32, y: i32, r2: f64 }

struct Job {
  job_id: u64
  // flat render parameters
  w: i32, h: i32, ss: i32
  center_re: f64, center_im: f64
  width: f64, bailout: f64
  iter_max: u64, iter_per_color: u64
  formula: i32   // 0 mandelbrot, 1 omnibrot
  omni_a: f64
  lx: f64, ly: f64, lz: f64
  color_inside: i32
  // tile assignment
  offset: u64, stride: u64
  // orbit reuse deltas
  delta_c_re: f64, delta_c_im: f64
  period_tol: f64
  // mouse origin for ring ordering? Worker computes fixed rows instead; see §5
}

struct RowResult {
  job_id: u64
  y: i32
  pixels: u32[]   // width entries, frozen array passed by value (flat)
}
```

`AppState` as root struct:
```goose
struct AppState {
  scene: Scene
  palette: Palette
  w: i32, h: i32
  image: u32[]          // w*h
  orbit_re: f64[]
  orbit_im: f64[]
  // render pixel ordering (optional; see §5)
  render_pixels: Point[]
  render_counts: u64[]
  // cache
  orbit_cache_valid: bool
  orbit_cache_iter_max: u64
  orbit_cache_c_re: f64
  orbit_cache_c_im: f64
  delta_c_re: f64, delta_c_im: f64
  // keyframe
  keyframe_image: u32[]
  has_keyframe: bool
  // recording
  ...
}
```

Memory ownership: AppState is a global root variable; arrays grow on the global
data stack. Scope exits in main are top-level, so buffers persist. Workers have
their own local arrays.

## 4. Threading design (no shared memory)

Workers cannot see AppState. Pattern:
1. Main thread prepares `Job` flat struct (all params by value).
2. Enqueue `job` to `jobs` queue: `jobs.put(job)`.
3. Worker: `let j = jobs.get();` compute assigned pixels; build `u32[]` row
   buffer; enqueue `RowResult { job_id, y, pixels }` to `results`.
4. Main thread each frame: while results available, copy into `image`.

Tile assignment: original sorts all pixels by distance from mouse for ring fill.
Without shared memory, worker can still compute any subset. Options:
- (A) Keep ring ordering on main thread: main precomputes `render_pixels` list
  (Point[]) and passes index ranges; worker maps indices to (x,y) via a
  precomputed array? Worker cannot access main's array. Could pass the slice of
  indices? Slices are not allowed in queue values. Could pass start/end indices
  into a deterministic ordering formula? Ring order depends on mouse; not simple.
- (B) Simpler: assign by rows (y ranges). Loses ring-fill visual but correctness
  intact. Recommended for port (visual minor). Each Job covers a contiguous row
  range; worker writes rows and returns RowResult per row.

Given Goose constraints, row-based jobs are the pragmatic choice. Ring-fill can
be approximated by updating texture progressively; not essential.

Number of jobs: `workers` (8) or more for progress granularity. Use one job per
worker covering row stripes; for progress, split into smaller row chunks (e.g. 16
rows). Enqueue many jobs; workers pull.

Termination/generation: original bumps `generation` and clears queue. In Goose,
queues cannot be "cleared" easily (no clear API). We can use a monotonic
`job_id`/`generation` field in Job and Result; main thread ignores results whose
generation mismatches. Workers check `j.generation != current_generation`? But
workers don't share current_generation. Pass `generation` in Job; worker echoes
it in Result; main filters. To stop work, main stops enqueuing and drains/ignores
stale results. Queue may backlog but bounded by job count. This replaces
`thread_pool_clear/wait`.

## 5. Rendering core (f64)

Port functions directly:
- `probe_known_interior`, `probe_mandelbrot`, `probe_omnibrot`, `probe_color`,
  `probe_colors_differ`.
- `mandelbrot` with interior tests, perturbation, period detection, renormalise,
  smooth count, derivative normal.
- `omnibrot` with complex_exp Taylor path.
- `mandelbrot_color`, `mandelbrot_shade`.
- SSAA loops in worker.

Interior color rule: if `color_inside && width <= 1e-3`, add (24,20,48).

Period detection: `if period_tol > 1e-61 && dist2 < period_tol*(1+z_mag2)`.

Adaptive controller identical, with `apply_iter_max` resizing `orbit_re`/`orbit_im`
arrays (`resize` or rebuild). Goose arrays: `orbit_re.resize(n+1, 0.0)`.

## 6. Orbit computation (f64)

Replace mpfr loop:
```goose
fn compute_orbit(state: AppState&, ...) {
  if formula == OMNIBROT { orbit_ready = true; return }
  // z = 0; for i in 0..iter_max+1 store; z = z*z + c
  state.orbit_re.resize(state.scene.iter_max+1, 0.0)
  state.orbit_im.resize(state.scene.iter_max+1, 0.0)
  var zr = 0.0; var zi = 0.0
  for i in 0..state.scene.iter_max+1 {
    state.orbit_re[i] = zr
    state.orbit_im[i] = zi
    let nzr = zr*zr - zi*zi + scene.c_re
    zi = 2.0*zr*zi + scene.c_im
    zr = nzr
  }
  // cache fields
}
```
Note: f64 precision limits deep zoom; acceptable for port baseline.

Precision upgrade path (optional, no C in repo):
- Declare `extern fn` to libmpfr symbols and manage `u64` handles. This is a
  large API surface and bypasses Goose ownership; keep behind a build flag
  `PRECISION=mpfr`. Not required for first complete port. Alternatively keep
  orbit computation in a small separate executable? Out of scope.

## 7. Adaptive iterations & probe

- Probe grid 64×48, deadline 200ms, min 512 samples.
- Use os clock (`os::clock_ms()` or equivalent) for deadline.
- Probe runs on main thread between frames or in a worker; main thread simpler
  and avoids queues. But probe can be expensive; running on main thread blocks
  UI. Acceptable (original also blocks render pipeline). Could run in a worker
  with result queue; but controller needs result before next adaptive decision.
  Keep main-thread probe like original (phase PROBE).
- Deep zoom detection: `width <= 1e-12` → AUTO_DEEP.

## 8. Vibe zoom

Same as original: disable SSAA (save ss_prev), zoom factor 1.02 toward mouse
every frame, terminate/redraw every 8th frame. `s.vibe` flag.

## 9. GPU / compute shader (optional)

Goose gfx supports compute pipelines and storage buffers/textures. Could port
Mandelbrot kernel to GLSL 450 compute shader:
- Limitations: GLSL 450 doubles require extension not guaranteed; use `double`
  only if device supports. Original OpenCL required `cl_khr_fp64`.
- Compute shader can write `rgba8` image via storage image; but smooth coloring
  needs f64. If unsupported, skip GPU (CPU fallback).
- Plan: implement CPU first; add compute shader as optional enhancement behind
  `--gpu` flag and runtime capability check. Not blocking for port.

## 10. Precision limitations and solutions

- Goose has f32/f64 only. Original deep zoom relies on mpfr orbit.
- Solution A (recommended): f64-only port; document max reliable zoom ~1e-13.
  Adaptive controller still works; interior tests remain.
- Solution B: extern binding to libmpfr (system library) with opaque u64
  handles; no C code in repo, but complex and unsafe. Defer.
- Solution C: implement big-float in Goose (huge effort). Not recommended.

No C code is retained in any case; B uses only system libmpfr via FFI.

## 11. Video recording (no subprocess API)

Goose os lacks process spawning. Options:
1. Use `extern fn` to libc `popen`/`pclose`/`fwrite`/`fflush` to drive ffmpeg.
   This is FFI to a system library, not project C code.
2. If avoiding all C ABI, write raw frames to `recording.raw` and print an
   ffmpeg command for the user to run manually.

Recommend option 1 for feature parity, with a fallback to option 2. Keyframe
warp/fit logic ports directly (uses image arrays).

## 12. UI / text

- Use `gfx` window with `WINDOW_RESIZABLE`.
- Display image by updating a texture each frame from `image.pixels`.
- Overlays: use `ui` labels/draw_text. Info lines exactly as original.
- Help overlay: list of controls.
- REC indicator and rendering progress.

## 13. File I/O and formats

Scene (.sce v1): read whole file via os, parse lines. Write via os write.
Palette (.pal v1): same.

Parsing approach: split content into lines (Goose string helpers `split`), parse
numbers with std parse functions. mpfr strings become f64 parse (`strtod` via
Goose float parsing or `os`? Goose supports float literals/parsing via std).
For compatibility, f64 parse of high-precision strings is fine.

## 14. Input mapping

Use `gfx` input:
- `key_pressed`/`key_down` for toggles and actions.
- `mouse_pos`, `mouse_delta` for pan.
- `mouse_wheel` for zoom.
- Resize: detect `screen_size()` change; rebuild image/arrays.

Actions mirror original keys (see reconstruction doc §8.11).

## 15. Build & run (Goose)

```bash
# dev (single-threaded, JIT)
goose run mandexplorer.goose
# release (threads)
goose build mandexplorer.goose
./mandexplorer default.sce
```
- Threads require AOT; dev mode can set `const USE_THREADS = false` to run
  single-threaded for debugging.
- gfx linking: `goose --gfx-link cc`.

## 16. Milestones

M0. Skeleton: window, load scene/palette, display static image buffer.
M1. f64 Mandelbrot render (CPU, single thread) + pan/zoom + overlays.
M2. SSAA + interior color + shading.
M3. Threaded rendering (queues) + progress.
M4. Orbit cache + period detection.
M5. Omnibrot + a control.
M6. Adaptive iterations (probe + controller).
M7. Vibe zoom.
M8. Save scene + palette reload.
M9. Video recording (ffmpeg pipe + keyframes + fit).
M10. Resize handling + help UI.
M11. Optional GPU compute shader.
M12. Optional mpfr precision upgrade.

## 17. Risks / open items

- Goose queue API details: need verify `qput`/`qget` global queue syntax and
  AOT thread compilation. Prototype early (M3).
- Thread count discovery: Goose os may not expose CPU count; use constant 8.
- No process API: ffmpeg integration needs libc FFI or manual fallback.
- `ui` module text capacity and font scaling; verify overlays readable.
- Memory ownership in root struct with many arrays; verify `resize`/`copy`.
- Frozen array passing through queues: ensure `u32[]` is flat enough for queue.
  If not, use `u32[..N]` fixed-capacity or index-based results.
- Performance: Goose generated C with -O3 should be okay; bounds checks may add
  cost; use `as!` where safe.

## 18. Verification plan

- Visual comparison against original at same scenes/params.
- Frame timing comparison.
- Scene/palette round-trip (save then load).
- Video output frame count and playback.
- Threaded vs single-threaded pixel equality.
- Resize stability.