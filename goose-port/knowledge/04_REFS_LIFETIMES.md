# References, optionals, relative links, pools

## References
- `T&` machine address, 8 bytes, no arithmetic/casts, never dangles.
- Transparent: no `*`/`->`; `.=` binds/rebinds reference; `==` compares targets, `.==` identity.
- `var cur .= pool[0];` binds; `cur .= l;` retargets.

## Optionals
- `T?` nullable reference; null = address 0, no extra space.
- Must be narrowed by `if`/`while`/`guard`/`assert` before use.
- Inside narrowed region it is `T&`.

## Relative references
- `T&<u32>` self-relative: offset from field to target in same enclosing array; position-independent.
- `T&<u32 in pool>` pool-relative: offset from named global pool base; other arrays can link into pool.
- Widths: u8/u16/u32/u64/varint.
- Loading gives ordinary `T&`; storing `T&` re-encodes.
- `T&<u32>?` uses offset 0 as null.
- Self-relative structures can be saved/loaded as bytes (`to_bytes`/`from_bytes`).

## self sentinel
- `self` means value being constructed; legal only as whole initializer of field in non-optional relative ref.
- Used for circular list sentinel.

## Lifetimes
- Every ref/slice has static root: variable bounding target lifetime.
- Rule: ref must not outlive owner variable; must not observe target at wrong type.
- No lifetime annotations; roots inferred; functions specialized per root.
- No aliasing/exclusivity rules.
- Lifetime errors name root and destination.

## Pools
- `slot_pool<T>`: root struct bundling `items` array + freelist. Declare owner local/global, pass as `slot_pool<T>&`.
- `alloc_index`/`alloc_ref` reuse free slot or push new; `free(i)` records slot for reuse.
- Elements remain valid always; freed slot reads replacement value (logic bug).
- `slice_pool<T>`: runs of elements: `alloc_slice(n)`, `realloc_slice(s,n)`, `free_slice(s)` with freelist merging neighbors.
- Backing array exposed as `pool.items`; length includes freed storage.
- `reusable`/`reusable[]` deprecated in favor of these.

## Root structs
- Ordinary struct at most one resizable tail.
- Root struct bundles state with independently stored fields, including multiple resizable arrays.
- Declare owner as local/global, pass by reference.
EOF