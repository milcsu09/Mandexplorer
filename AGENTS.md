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
- `.` / `,`: period detection tolerance up/down (1e-61 disables → "off").

## Period detection
- Threshold in `main.c` (~line 904): `period_tol > 1e-61 && ...`. Default 1e-40. Tunable via global `period_tol`.

## Notes
- `terminate()` called before state changes requiring recompute.
- `apply_iter_max()` reallocates orbit buffers.
EOF
path="/home/et/Applications/Mandexplorer/AGENTS.md"