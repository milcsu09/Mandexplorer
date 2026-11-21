# Goose knowledge base (for porting Mandexplorer)

Source: cloned `goose-port/goose` (aardappel/goose). Docs read: README, tutorial, samples README, stdlib (math/os/gfx), Mandexplorer quick scan.

## Files
- `01_OVERVIEW.md` — what Goose is, build/run, limitations.
- `02_MEMORY_MODEL.md` — data stacks, scopes, arrays, strings.
- `03_ARRAYS_STRINGS.md` — array family, strings, slices.
- `04_REFS_LIFETIMES.md` — references, optionals, relative links, pools.
- `05_ENUMS.md` — ADTs, match, case functions.
- `06_CONTROL_ERRORS_GENERICS_THREADS.md` — control flow, errors, generics, threads.
- `07_STDLIB_CORE.md` — std, exception, dictionary, vec/mat/quat, math, os, binary, ffi.
- `08_GFX.md` — SDL3 GPU graphics module.
- `09_AUDIO_UI_PHYSICS.md` — audio/ui/physics (brief).
- `10_PORTING_MANDEXPLORER.md` — mapping C/SDL2/mpfr Mandexplorer to Goose.
- `11_BUILD_RUN.md` — compiler commands, packages, C interop.
- `12_MANDEXPLORER_RECONSTRUCTION.md` — full reconstruction of the existing Mandexplorer C/SDL2/mpfr program.
- `13_PORT_PLAN_GOOSE.md` — detailed Mandexplorer → Goose port plan, subsystem mapping, limitations and solutions (no C code retained).


## Key facts for porting
- Goose is whole-program compiled to C; supports `extern fn` to C (mpfr, SDL).
- Graphics via `gfx` (SDL3 GPU) or extern SDL2.
- Threading via `thread_fn` + typed queues, flat values only.
- High precision: no native mpfr; use extern calls to libmpfr (C).
- No heap; all dynamic values owned by variables/scopes.
- Whole program compile; no separate compilation.
