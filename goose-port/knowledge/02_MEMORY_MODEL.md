# Memory model details

## Data stacks
- Program has native call stack, static data, N data stacks.
- Data stack = multi-GB address-space reservation with bump pointer.
- Only memory system: no malloc, free list, GC, refcount, destructors.
- Values pushed in lifetime order; popped en masse when owning scope exits.
- At most one resizable value live per stack, topmost for entire lifetime.
- Growth is pointer bump, no capacity check; value never moves.
- Compiler proves all statically; runtime tracks bump pointers only.

## Leaving scope frees data
- Closing brace resets stack top (single store).
- No destructor runs.

## Nothing moves, references survive growth
- `push` returns reference to new element.
- References into growing array stay valid.

## Copies visible
- Fixed-size values (scalars, structs, fixed arrays, refs, slices) copy silently on assignment (C-like).
- Non-fixed values never copy implicitly; use `copy(a)` for explicit O(n) copy.
- `var b = a;` for non-fixed binds by reference.
- Call `f(xs)` passes growable array by reference without copying elements.

## In-place construction
- Constructed value always built at final location, passed down through calls.
- Returning built-up value costs same as writing through out-param.

## Size classes (reference)
- Fixed: scalars, structs, fixed arrays, refs, slices.
- Resizable: `[>..]` grow-only, `[>..<]` grow-shrink.
- Frozen: `[]`, `[varint]`.
- Limited: `[..k]`, `[..]`.
EOF