# Mandexplorer — full reconstruction of the existing C/SDL2/mpfr project

Purpose: this document fixes the *actual* behaviour of the original Mandexplorer
program as implemented in `src/*.c`. It is the reference for the future Goose
port. No port code exists yet; this is study only.

## 1. Project identity

- Name: Mandexplorer, an interactive Mandelbrot-set explorer.
- Author: Pető Milán (MIT license).
- Language: C11, single binary built from `src/*.c` via `make` (output `a`).
- Windowing: SDL2 (window, renderer, textures, events, threads, mutex/cond).
- Text: SDL2_ttf (`res/DMSans-Regular.ttf`, 18 px).
- Math: GNU MPFR at 2048-bit precision (`FLOAT_PREC`), for the reference orbit
  and scene coordinates; `double` for perturbation deltas and Omnibrot.
- Optional: OpenCL (`HAVE_OPENCL`) for GPU rendering; falls back to CPU.
- Threading: custom thread pool (SDL threads) for tile/row rendering.

## 2. Build

`Makefile`:
- `gcc -std=c11 -Wall -Wextra -Wpedantic -O3 $(EXTRA_CFLAGS) -oa $(wildcard src/*.c) -lm -lSDL2 -lSDL2_ttf -lmpfr $(EXTRA_LIBS)`
- OpenCL detected via pkg-config; if present defines `HAVE_OPENCL=1` and links
  OpenCL, else `HAVE_OPENCL=0`.
- No tests.

Run: `./a <scene.sce>` or `./a --help`. `default.sce` shipped. GPU info:
`./a --gpu-info` (initialises GPU, prints availability/device, exits).

## 3. Source layout

| File | Role |
|---|---|
| `src/common.h/.c` | typedefs (u8..u64, i8..i64, f32, f64, usize), `malloc0`, `imin`/`imax`, `enum formula` (`FORMULA_MANDELBROT`, `FORMULA_OMNIBROT`). |
| `src/config.h` | `FLOAT_PREC 2048` (bits), `TILE_SIZE 20` (unused). |
| `src/scene.h/.c` | `struct scene` (mpfr center, width, bailout, light lx/ly/lz, iter_max, iter_per_color, palette_path); create/destroy/save/load. File version 1. |
| `src/color.h/.c` | ARGB u32 helpers: rgb/argb/a/r/g/b, lerp, bilinear lerp. |
| `src/palette.h/.c` | `struct palette` (size + flexible `u32 data[]`); create/from/destroy/get (mod size), save/load version 1. |
| `src/image.h/.c` | `struct image` (w,h, `_Atomic u32 pixels[]`); create/destroy/set/get (atomic relaxed)/fill/copy. |
| `src/thread_pool.h/.c` | Mutex/cond queue of function pointers; create/destroy/clear/enqueue/wait/is_done/threads_active. |
| `src/gpu.h/.c` | OpenCL init/shutdown/availability/fp64/should_use/render; embedded kernel string; binary cache `~/.cache/mandexplorer/kernel_<hash>.bin`. |
| `src/main.c` | Everything else: args, init, event loop, rendering phases, algorithms, UI, video. |

## 4. Configuration constants (main.c)

- `MAX_ITER_MAX 16777216`
- `OMNIBROT_A_DEFAULT 0.005`
- Orbit reuse: `ORBIT_REUSE_EPS 1e-3`, `ORBIT_REUSE_MIN_WIDTH 1e-10`
- Video fit: `VIDEO_FIT_MAX_ATTEMPTS 3`, `VIDEO_FIT_SHRINK_FACTOR 4`,
  `VIDEO_FIT_MIN_SHRINK 64`, `VIDEO_FIT_PROBE_DEADLINE 500`,
  `VIDEO_FIT_PROBE_MIN_SAMP 128`, `VIDEO_FIT_SENS_LO 0.005`
- Keyframe: `KEYFRAME_MAX_REUSE 5`, `KEYFRAME_MAX_MAG 1.5`
- Adaptive: `AUTO_BUDGET_MS 6000`, `AUTO_TARGET_MS 4000`, `AUTO_HARD_MS 60000`,
  `AUTO_SENSITIVITY_LO 0.005`, `AUTO_SENSITIVITY_HI 0.020`, `AUTO_DEEP_WIDTH 1e-12`
- Probe: `PROBE_COLS 64`, `PROBE_ROWS 48`, `PROBE_DEADLINE_MS 200`,
  `PROBE_MIN_SAMPLES 512`
- Period tolerance default `period_tol = 1e-40`; disable threshold `> 1e-61`.
- Interior color default on; `color_inside`.

## 5. Global state `s`

Single static struct `s`:
- `scene*`, `palette*`, `w,h`, `sectors` (CPU count, min 4).
- `mouse_cx,cy`, `vibe`, window/renderer, textures (texture, texture_canvas,
  texture_helper), font, running.
- `phase`: `PHASE_IDLE / COMPUTE_ORBIT / COMPUTE_IMAGE / PROBE`.
- `pool`, `image`, `orbit_re`, `orbit_im` (double arrays length iter_max+1).
- `ss`, `ss_prev`, pan_x, pan_y, scale, redraw.
- `formula`, `omni_a`, `center_re`, `center_im` (double viewport center).
- `generation` (atomic), `work_generation`.
- `orbit_ready` (atomic), `tiles_done` (atomic).

Plus module globals: render pixel list/count/cap, render counts/cap, probe
snapshot fields, adaptive state, period_tol, color_inside, orbit_cache, delta_c,
keyframe state, recording/video state, ffmpeg pipe, recorded_views array.

## 6. Scene file format (.sce), version 1

Lines:
1. version (u32)
2. c_re (mpfr string)
3. c_im (mpfr string)
4. width (%.17e)
5. bailout (%.17e)
6. lx (%.17e)
7. ly (%.17e)
8. lz (%.17e)
9. iter_max (u64)
10. iter_per_color (u64)
11. palette_path (string up to 4095 chars)

`scene_create`: mpfr_init2 at FLOAT_PREC for c_re/c_im.

## 7. Palette file format (.pal), version 1

1. version
2. size (u64)
3. for each: `a r g b` (u32 each)

`palette_get` returns `data[i % size]`.

## 8. Algorithms

### 8.1 Reference orbit (Mandelbrot, mpfr)

`compute_orbit` (runs in thread pool or cache-hit skip):
- If Omnibrot: mark orbit_ready, return (no orbit).
- Else iterate z_{n+1}=z_n^2+c in mpfr at 2048 bits, starting z=0, for
  0..iter_max, storing `orbit_re[iter]`, `orbit_im[iter]` as double.
- Fill `orbit_cache`: mpfr copies of c_re/c_im, double copies, iter_max, valid.

Orbit reuse (`PHASE_IDLE` block): if Mandelbrot and cache valid and same iter_max:
- Compute cur_re/cur_im doubles and dc_re/dc_im.
- If exact mpfr equality and dc==0: delta=0, cache_hit.
- Else if width>1e-10 and |dc_re|/width<1e-3 and |dc_im|/width<1e-3:
  delta_c_re=dc_re, delta_c_im=dc_im, cache_hit.
- Else recompute orbit.

### 8.2 Per-pixel Mandelbrot (perturbation)

`mandelbrot(delta_re,delta_im,&nx,&ny)`:
- Interior tests: period-2 circle `(x+1)^2+y^2 <= 1/16`; main cardioid
  `q*(q+qx) <= y^2/4` with qx=x-0.25. If inside, return iter_max (interior).
- Apply delta adjustment: `delta_re += delta_c_re`, `delta_im += delta_c_im`.
- Perturbation iteration:
  - r,z from orbit[iter_orbit]; epsilon perturbation.
  - deriv (for distance estimate) D_{n+1}=2*z_n*D_n+1.
  - epsilon_{n+1}=2*z_n*epsilon_n + epsilon_n^2 + delta.
  - z = orbit[iter_orbit+1]+epsilon.
  - Period detection: if have_saved, compute dr,dip; if period_tol>1e-61 and
    dr^2+di^2 < period_tol*(1+z_mag2): return iter_max (period).
  - Save z at iter+1==period, then period*=2 (cap iter_max).
  - If z_mag2 > bailout^2: compute normal vector nx,ny from deriv; smooth
    iteration count `nu = iter+1 - log(log(z_mag2)*0.5)/ln2`; return nu.
  - iter++, iter_orbit++.
  - If |epsilon|^2 > z_mag2/2: reset epsilon=z, iter_orbit=0 (renormalise).
- If never escapes: return iter_max.

`mandelbrot_color(nu)`: `x=nu/iter_per_color`; i=floor(x); f=frac; lerp palette[i],palette[i+1].

`mandelbrot_shade(nx,ny)`: lighting `(nx*lx+ny*ly+lz)/sqrt(nx^2+ny^2+1)`, clamp to [0.3? actually max(0, shade*0.7+0.3)].

Interior coloring: if color_inside and width<=1e-3, add (24,20,48) per sample.

### 8.3 Omnibrot

Formula: z <- z^2 * exp(a/z) + c, a in [1e-6,1] (default 0.005), critical point z0=a/2.
`omnibrot` plain double, no perturbation (zoom limited ~1e-13). Uses Taylor exp
for small |re|+|im|<0.25, else exp/cos/sin. Bailout on mag2>bailout^2 or non-finite.
Returns smooth? Actually returns `fmax(0, iter+1 - log(log(mag2)*0.5)/ln2)` on escape; iter_max if never.

`probe_omnibrot` simpler (no deriv/nx).

### 8.4 SSAA

`s.ss` in {1,2,4,8}. Worker loops sx,sy in [0,ss), averages r/g/b over ss*ss samples
into one pixel. GPU kernel does same. `mandelbrot_color` per sample then shading.

### 8.5 Period detection tolerance

Global `period_tol`, default 1e-40. Keys `.` increases ×10, `,` decreases /10
(min 1e-61). If `period_tol <= 1e-61` ("off"), condition `period_tol > 1e-61`
disables detection. Info overlay shows on/off and value.

### 8.6 GPU (OpenCL)

- Init: find platform/device with `cl_khr_fp64`, prefer GPU over CPU. Create context/queue, build program from kernel_src (or cached binary).
- Kernel `mandelbrot_kernel` mirrors CPU perturbation (double), including period
  detection, omnibrot? No—GPU returns early for Omnibrot (cpu fallback). Also
  skips if width small? `gpu_should_use`: available, not omnibrot, w*h>=128*128,
  and not (vibe && iter_max>5e5 && width<1e-13) / (iter_max>5e5 && width<1e-10).
- Orbital buffers uploaded; palette uploaded; output read back into image.
- Device name printed; `--no-gpu` disables.

### 8.7 Threading

Thread pool: `SDL_GetCPUCount()` threads (min? sectors min 4 but pool threads count CPU count), queue capacity 1920*1080. Jobs are `struct work {offset,stride}`; `n_tiles()` = sectors (CPU count, min 4). Each worker processes pixels `i=offset; i<total; i+=stride` in render_pixels order (sorted by distance from mouse for circular fill). Increments tiles_done on finish. `terminate()` clears queue, waits, resets phase/idle, orbit_ready false, tiles_done 0, bumps generation.

CPU count via `SDL_GetCPUCount()`; sectors same.

### 8.8 Adaptive iterations

- `auto_iter` bool (default true), toggled by `i`.
- States: AUTO_OFF/HOLD/GROW/CONVERGED/BUDGET/RESCUE/DEEP/VIBE.
- Probe: coarse 64×48 grid (coprime row stride), compares mandelbrot at N vs 4N
  (or omnibrot), counts pixels whose color changes (escaped flag change or RGB diff≥2). Deadline 200ms / min 512 samples. Skips known interior.
- `auto_iter_update(render_ms, vibe)`:
  - If !auto_iter -> OFF.
  - If vibe -> VIBE.
  - If width<=1e-12 -> DEEP.
  - If probe_total==0 -> HOLD.
  - sensitivity = changed/total.
  - If <0.005 -> CONVERGED.
  - factor = clamp(TARGET/render_ms, 1.25, 2.0); if sens<0.02 factor=min(factor,1.5).
  - next = iter_max*factor (at least +1), cap MAX.
  - If next==iter_max -> BUDGET.
  - estimate = render_ms*(next/iter_max)*1.1.
  - If estimate>6000: rescue mode: next=iter_max*2, estimate; if estimate>60000 -> BUDGET return.
  - apply_iter_max(next); if changed -> GROW/RESCUE, redraw; else HOLD.
- Only grows N, never lowers. Manual Home/End override and disable auto.

### 8.9 Vibe zoom

Toggle `v`: disables SSAA (saves ss_prev), sets ss=1; every 8th idle frame (counter %8==0) terminates+redraw; each frame while phase idle zooms 1.02 toward mouse and pans? Actually zoom toward mouse (sx,sy) anchor; s.pan accumulates? Code: computes world point, scale*=1.02, pan_x=world_x-sx/scale. Wait in vibe block it sets pan_x directly (not accumulates) using sx. Then main loop applies pan. So smooth zoom toward cursor. Vibe also suppresses adaptive probe? In compute_image completion, can_probe requires !vibe, so auto_iter_update called without probe; state VIBE.

### 8.10 Video recording

Toggle `r`:
- Start: clear recorded_views, recording=true, vibe=true (smooth trajectory).
- While recording: each frame (if not dup within 1e-12) append view_state (cap 4096; if full stop recording and start render).
- Stop: recording=false, vibe=false; if recorded_count>0 start_render_phase.

`start_render_phase`: rendering_video=true, render_index=0, video_frame_pending=false, has_keyframe=false; open ffmpeg pipe:
`ffmpeg -y ... -f rawvideo -pix_fmt rgb24 -s WxH -r 30 -i - -c:v libx264 -pix_fmt yuv444p -preset fast "YYYYMMDD_HHMMSS.mp4"`.

Render scheduling (when rendering_video && !pending && index<count && phase idle):
- If has_keyframe && can_reuse_keyframe(kv,v): warp_keyframe (bilinear scale/translate) into image, pending=true, keyframe_pending=false. (Fast path.)
- Else: terminate; set formula/omni_a; fit_video_n (adaptive shrink iter_max until probe sensitivity<0.005 or min); apply_iter_max if changed; set scene params; ss=2; color_inside; pan 0 scale 1; center; auto off; redraw; pending=true; keyframe_pending=true.

`fit_video_n`: starts n=recorded iter_max; tries n, n/4, n/16 (min floor iter_per_color*4 or 1) using probe_job with deadline 500ms/min 128; stops when sensitivity>=0.005 or too few samples.

When frame completes (phase idle & pending): write image as rgb24 to ffmpeg; if keyframe_pending copy image to keyframe_image, store keyframe_view, has_keyframe, index; pending=false; index++. If index>=count: close pipe, rendering_video=false.

`can_reuse_keyframe`: same formula, recorded_iter_max, iter_per_color, ss, color_inside, omni_a, bailout, lx,ly,lz; geometric containment of scaled viewport; scale S>=1/1.5; reuse count <5. `warp_keyframe`: bilinear sample from keyframe image with scale S and offset cx,cy.

### 8.11 Input controls

Keys:
- Esc: terminate.
- Alt: toggle info overlay.
- F1/h: toggle help overlay.
- v: vibe zoom.
- i: adaptive iterations toggle.
- b: interior color toggle.
- m: Mandelbrot formula (terminate, invalidate orbit cache).
- o: Omnibrot formula.
- [ / ]: omni_a /2 or *2 (clamp 1e-6..1), terminate.
- x: reset view (width 8, center -0.75+0i, pan/scale reset), terminate.
- c: center on mouse (adjust pan by mouse offset/scale), terminate.
- Insert/Delete: bailout *2 or /2 (min 2), terminate.
- Home/End: iter_max *2 or /2 (manual, disables auto), apply_iter_max.
- PgUp/PgDn: iter_per_color *2 or /2 (min1), terminate.
- . / ,: period_tol *10 or /10 (min 1e-61), terminate.
- s: save scene as timestamp .sce (uses scene_save).
- r: recording toggle / render.
- 1/2/4/8: ss.
- Mouse left drag: pan (pan_x -= xrel/scale, pan_y -= yrel/scale).
- Left button up: terminate+redraw.
- Right button down: terminate+redraw.
- Wheel: zoom toward mouse (or center if Ctrl); factor 1.4 (1.1 if Shift); up zoom in. terminate.
- Window resize: recreate textures/image, works array, has_keyframe false, terminate? Actually terminate called? In resize handler no terminate call? It calls image destroy/create, textures recreate, works realloc, redraw=true. Does not bump generation? It doesn't call terminate. Hmm. But redraw triggers recompute using current generation; old threads may still run? Possibly not cleared. Actual behavior as coded.

CLI args: `-w/-h` (initial size), `--re/--im/--width/--bailout/--light-angle/--iter-max/--iter-per-color/--formula (mandelbrot|m|omnibrot|o)/--a/--palette/--no-gpu/--gpu-info`, scene path required.

Light angle: converts to lx=sin, ly=cos, lz=1, normalized.

### 8.12 UI overlays

Info overlay (Alt): top-left lines: C_re, C_im, Width, formula, omni_a, bailout, period status, iterations, iter per color, adaptive state/sensitivity/frame, interior color, SSAA. Bottom-left: phase status with timing estimates. Help overlay (F1/h): centered list of keys. REC indicator (red) if recording. "Rendering video: i/n" if rendering.

Text rendering: `write_text` uses TTF_RenderText_Blended, creates texture, draws black bg rect, copies. Called many times per frame.

### 8.13 Main loop phases

PHASE_IDLE:
- If redraw: apply pan/zoom to scene (mpfr_add/sub, width/=scale, reset pan/scale), capture mouse, fill image, set work_generation, snapshot probe params, set center doubles, orbit cache check/enqueue compute_orbit, then if not cache hit stays IDLE? Actually sets phase=PHASE_COMPUTE_ORBIT after enqueue; if cache_hit orbit_ready true but phase still set to PHASE_COMPUTE_ORBIT (checked next? It sets phase before cache check? Let's see: `s.phase = PHASE_COMPUTE_ORBIT;` then cache logic sets orbit_ready. So phase is COMPUTE_ORBIT; next loop sees orbit_ready true -> transitions. For cache hit, immediate transition next frame.)

PHASE_COMPUTE_ORBIT: if orbit_ready -> set work_generation, start timer, phase COMPUTE_IMAGE, build render_pixels sorted by mouse distance, then GPU or enqueue workers.

PHASE_COMPUTE_IMAGE: if tiles_done==n_tiles -> compute total ms; if can_probe enqueue probe_job -> PROBE else auto_iter_update and IDLE.

PHASE_PROBE: if probe_done -> auto_iter_update and IDLE.

Each frame: upload image to SDL texture, render with current pan/scale (but pan usually 0 after idle apply), overlays, present. During compute, image partially filled; UI shows progress.

Texture upload: lock streaming texture, copy image pixels, unlock. Then render to canvas texture, then to screen with pan/scale transform (usually identity).

### 8.14 Component relationships

- `main.c` owns `s` and orchestrates.
- scene provides high-precision center/width; main converts to double center for perturbation/GPU.
- palette provides colors; image holds pixels.
- thread_pool executes compute_orbit and workers and probe_job.
- gpu optional fast path; uses orbit/palette/image.
- color helpers.
- common types.

Data flow per frame (interactive):
1. Input mutates pan/scale/scene params or terminates.
2. Redraw triggers phase idle transform -> orbit compute (or cache) -> image compute (CPU/GPU) -> optional probe -> adaptive N update.
3. Render image to screen.

## 9. Actual behavior notes / quirks

- `TILE_SIZE` defined but unused.
- `render_counts`/`render_pixels` allocated dynamically; if allocation fails, render_pixels null -> no CPU workers enqueued (GPU path still possible).
- `thread_pool_create (SDL_GetCPUCount(), 1920*1080)`; if CPU count 0? sectors min 4 but pool threads CPU count (could be 0 on weird). Not guarded for pool.
- `apply_iter_max` allocates buffers before terminate; frees old after terminate. Race safety relies on terminate clearing queue/waiting.
- Omnibrot disables GPU and orbit; adaptive probe uses omnibrot probe (no interior skip? It calls probe_omnibrot directly without known_interior).
- Interior coloring only if width<=1e-3 (both CPU/GPU/probe? probe doesn't color; adaptive uses sensitivity not color).
- Probe uses `palette_get(s.palette,...)` global; okay.
- `period_tol` global not in scene; not saved/loaded. Resets to 1e-40 each run.
- `color_inside` global not saved in scene? Scene doesn't store it. Not persisted.
- `auto_iter` resets? Not persisted.
- Resize doesn't terminate workers (potential race) but sets redraw.
- `s.vibe` assigned from local `vibe` each loop.
- Recording cap 4096 frames; if exceeded stops and renders.
- Video render uses ss=2 regardless of original.
- `fit_video_n` mutates probe globals and restores deadline/min_samples.
- Keyframe reuse requires recorded_iter_max equality; fit may change v->iter_max during render, but recorded_iter_max stored separately? view_state has both iter_max and recorded_iter_max. In recording, only iter_max set; recorded_iter_max left 0/uninitialized! Wait struct view_state has `recorded_iter_max`. In recording capture, `vs.iter_max = ...;` but `vs.recorded_iter_max` not set. So it's 0 (calloc? recorded_views realloc not zeroed? realloc doesn't zero; but first allocation calloc? They use realloc on NULL? `recorded_views = NULL` then realloc(NULL,...) acts like malloc (uninitialized). Hmm. Actually `recorded_views` initially NULL; realloc(NULL, size) = malloc (uninitialized). So recorded_iter_max garbage. But can_reuse_keyframe compares `kv->recorded_iter_max != v->recorded_iter_max`. This may be unreliable. Actual behavior as coded. Note for porting: fix.
- In video render, `v->iter_max` may be overwritten by fitted_n (`v->iter_max = fitted_n`), but recorded_iter_max unchanged (garbage/0). Fine.

- `center_re/center_im` doubles derived from scene mpfr; used for perturbation delta and omnibrot.

- `work_generation` bumped on redraw and on phase transitions.

- `generation` atomic bumped by terminate().

## 10. Dependencies

- SDL2, SDL2_ttf, mpfr, (optional OpenCL), libc, ffmpeg (runtime for video), gcc.
- Font file path relative to CWD: `res/DMSans-Regular.ttf`.
- Scenes/palettes relative paths.

## 11. Porting implications (summary)

- Core compute is perturbation Mandelbrot using mpfr orbit + double epsilon; high precision only in orbit. Port can keep mpfr via FFI or reimplement orbit in f64 for shallow zoom.
- All state is global mutable; port should structure as app state.
- Threading model: thread pool with function pointers; Goose uses queues of flat values.
- GPU path optional; OpenCL kernel duplicates CPU logic. Port may skip GPU initially.
- Video recording depends on ffmpeg subprocess and raw rgb24 pipe.
- UI text via TTF; Goose ui module or extern TTF.
- File formats must be preserved for compatibility (.sce v1, .pal v1).

---

This reconstruction reflects code as of commit `200eb70`.<|end|><|start|>advisor to=write<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="path">
/home/et/Applications/Mandexplorer/goose-port/knowledge/00_INDEX.md