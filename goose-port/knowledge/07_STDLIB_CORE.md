# Standard library core

Modules: std, exception, dictionary, vec, mat, quat, math, os, ffi, binary, base64, csv, json, regex, audio, gfx, physics, ui.

## std
Scalars/bits, hashing, rng, arrays (read/search/transform/in-place/length), byte sets, strings.

## exception
Packages return...from as `exception::try<E>(body) { error => ... }` and `exception::throw(value)`.

## dictionary
Keyed by slices (e.g. `const u8[:]`), no copies.

## vec/mat/quat
Vector/matrix/quaternion helpers.

## math
libm at both widths (sin etc pick by arg type). Constants PI, TAU, E, INF, NAN. radians/degrees, is_nan/inf/finite, fract, inverse_lerp, smoothstep, wrap_angle. mul_add (FMA semantics). checkfree-friendly.

## os
File I/O, dirs, stdin/stdout, args, env, time/clock, sleep, random_seed/bytes, resource_dir, trim_memory, exit/abort. Paths UTF-8. read_file appends whole file; write_file_atomic. list_dir sorted. resource_dir for shipped assets.

## ffi
cstr/bytes to copy C-owned memory into Goose storage.

## binary
Checked reads/writes external formats (LE/BE), cursor Reader.

## json/csv/regex/base64
Parsing/serialization helpers.

## audio
Procedural mono/stereo PCM via SDL3, no graphics window.

## physics
Rigid body physics on Box3D, up to 8 worker threads.

## ui
Fonts, text rendering, game HUDs, windows/widgets on Nuklear drawn through gfx.
EOF