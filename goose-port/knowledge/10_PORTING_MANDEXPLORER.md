# Porting Mandexplorer (C/SDL2/mpfr) to Goose — notes

## Original project
- C/SDL2 Mandelbrot explorer, single binary from src/*.c.
- Uses mpfr for high precision Mandelbrot iteration.
- SDL2 renderer + textures, TTF overlay (Alt toggles info).
- Thread pool for parallel rendering.
- Period detection, adaptive iterations, SSAA, video recording, scene save/load.

## Key bindings / features to preserve
- Alt info overlay; F1/h help; v vibe zoom; i adaptive iterations.
- b interior color; m/o Mandelbrot/Omnibrot; [ / ] omni a.
- x reset view; c center on mouse.
- Insert/Delete bailout ±; Home/End iter_max ±; PgUp/PgDn iter per color.
- s save .sce; 1/2/4/8 SSAA.
- r record video.
- ./, period detection tolerance.

## Mapping to Goose

### Precision
- Goose has only f32/f64 math. Mandexplorer uses mpfr (arbitrary precision).
- Option A: port to f64 (lower precision, simpler) — loses deep zoom.
- Option B: keep mpfr via `extern fn` to libmpfr (C). Goose supports extern fn and C interop. Need declare signatures and manage mpfr_t (opaque pointers as u64?).
- mpfr_t is a struct; can pass as pointer (u64) across FFI. Could write C shim functions operating on mpfr_t pointers and call from Goose.
- Better: keep core iteration in C shim, called from Goose, or port UI/rendering to Goose and keep compute in C.

### Graphics
- Original uses SDL2 renderer + streaming texture (ARGB8888).
- Goose gfx uses SDL3 GPU; can create texture and update_texture with pixel buffer. Or use extern SDL2 calls directly.
- Text overlay: original uses SDL_ttf; Goose can use ui module (Nuklear text) or keep TTF via extern.

### Threading
- Original uses thread_pool (C).
- Goose: `thread_fn` workers + typed queues. Jobs/rows must be flat. Mandelbrot row can be flat struct with index + pixel buffer (u8[]). Sample 23 shows pattern.

### State/rendering
- Original renders to SDL_Texture; Goose can maintain pixel buffer (u8[>..] or fixed) and update gfx texture each frame.
- SSAA: render at higher resolution then downsample; can do in pixel buffer.

### File I/O
- os module covers read/write files, dirs.

### Period detection
- Keep algorithm; uses orbit buffers. Goose arrays can hold f64 orbits (or mpfr via C).

### Scene save/load (.sce)
- Use os read/write + parsing; could use text format.

## Recommended port strategy
1. First build a minimal Goose program: window via gfx, render Mandelbrot at f64, display texture.
2. Add input/zoom/pan.
3. Add threading via queues.
4. Add UI overlay (ui module or text).
5. Decide precision: f64 first, then mpfr shim if needed.
6. Add save/load, recording, SSAA, period detection.

## Constraints to remember
- Whole-program compile; threads need external C compiler (`goose build`, not JIT).
- No heap; pixel buffers as growable/frozen arrays.
- Flat values for queues.
- Bounds checks: use asserts for performance.
EOF