# Goose overview

## Identity
- Memory-safe systems language, no heap allocator, GC, refcount, destructors.
- Compiles whole program to one C file; runs where C compiler exists.
- In-process backend: TinyCC (JIT) for `goose file.goose` (no separate C compiler).
- Threads require external C compiler (TinyCC cannot place TLS).

## Syntax
- Familiar C/Rust: braces, `let`/`var`/`const`, `fn`, `struct`, `match`.
- `let` immutable binding, `var` mutable; `const` read-only contents.
- Types after name; inference common.
- `if`/`match`/blocks are expressions.
- No methods/impl; UFCS: `x.f(a)` calls `f(x,a)`.
- Multiple return values (calling convention, not tuple type).
- Trailing params with defaults.

## Arithmetic
- Arithmetic at operands' own width; no silent promotion to 64-bit.
- Implicit int->float; float->int only via `as`.
- Float literal has no fixed width; adapts to context.
- `%` Euclidean (result in [0, |b|)).
- Unsigned wraps always; signed overflow aborts debug, wraps release.
- Casts: `as` range-checked debug; `as!` never checked.

## Memory model
- Native call stack + static data + N data stacks (compiler computes N).
- Data stack = large address-space reservation, bump pointer.
- No malloc/free list/GC/refcount/destructors.
- Every dynamic value owned by variable; scope exit resets pointer.
- At most one resizable value live per stack, always topmost; growth = pointer bump; never moves.
- Compiler proves statically.

## Limitations
- Need think about ownership/scope.
- Recursive functions cannot keep growable data across recursive call.
- Arrays of variable-size elements cannot be indexed (unless index of refs).
- Fixed-mode enum cannot be pointed into; variable-mode cannot be overwritten.
- No escaping closures, function pointers, dynamic dispatch beyond ADT tags.
- Whole-program compilation only.
- Pool reuse: stale slot reads replacement value (logic bug, not memory corruption).
- v1 omissions: no move for resizables, no nested owning root structs/arrays, no labeled break, no namespace privacy, no subprocess/networking in library.

## Status
- New language; compiler ~30k lines C++.
- CI on Windows/macOS/Linux + sanitizer.
EOF