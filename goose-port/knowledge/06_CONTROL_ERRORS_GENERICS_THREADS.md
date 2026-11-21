# Control flow, errors, generics, threads

## Control flow
- `if`, `match`, `block`, `{}` are expressions; block value = trailing expression.
- No ternary.
- `guard c;` / `guard c else { ... }`.
- `block { ... }` gives break target with value.
- `defer` runs statement/block when enclosing block ends (any exit).
- Loops: `for x in range_or_array`, `while`, `loop`.
- `for i in 3`, `for i in 10..13`, `for i: i32 in 0..n`.
- `for x in xs` copies (fixed-size elements); `for &x in xs` references.
- `for x, i in xs` with index.
- Multiple results: `let q, ok = divide(7,2);` assign existing with `=`.
- Trailing params defaults.

## Errors
- No exceptions/Result convention; app chooses.
- Trailing bool for two outcomes; optional for found/not.
- `return E from f`: returns E as result of innermost active call of f, unwinding frames. Checked statically. Hidden discriminant implementation.
- `assert`, `abort(msg)`, `exit(code)`.

## Generics
- Untyped param is generic; `<T>` names type.
- Monomorphized; type args inferred from args.
- Overloads resolve on arg types; slice param takes any array kind.
- Function values are compile-time entities; passed as generic params; no closures/pointers; cannot escape.
- Trailing block sugar: `xs.filter() { it % 2 == 0 }`.
- Blocks may `return` from lexically enclosing function.
- Nested functions can use enclosing locals.
- Recursion opt-in: `recursive fn`; explicit signatures; no growable data across recursive call.

## Threads
- `thread_fn worker()` compiled as separate program with own memory/globals copies.
- Values pass through typed queues, one per type.
- Values must be flat (no refs/slices at any depth); passing is memcpy.
- `thread_spawn`, `thread_spawn_contained` (failure contained, message to waiter).
- `thread_wait(id)` / `thread_wait(id, msg)`.
- `qput<T>`, `qget<T>`.
- No shared mutable memory; no locks/atomics/ordering in Goose code.
- Standard library thread-safe for stateful OS functionality.
- Threads need external C compiler (TinyCC limitation).

## Calling C
- `extern fn` binds Goose signature to C symbol.
- Types crossing: scalars/bool, flat fixed structs/arrays (by value or via ref pointer), slices (as `{data,len}`), `u8[>..]&` builder.
- Compiler emits one C file.
- `--include header.h`.
- `export fn` makes Goose function callable from C.

## Calling Goose from C
- `export fn` ordinary Goose fn; compiler adds C wrapper.
- Optional string picks C symbol.
- `--standalone --header hello.h -o hello.c`.
- Host calls `goose_init` once; `goose_exit` releases.
EOF