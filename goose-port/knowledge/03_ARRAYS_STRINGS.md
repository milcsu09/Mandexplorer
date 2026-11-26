# Arrays, strings, slices

## Array family
| Type | Meaning | Metadata |
|---|---|---|
| `T[k]` | fixed size compile-time | none |
| `T[]` / `T[varint]` | sized at construction, frozen | length (u32 or varint) |
| `T[..k]` | capacity k inline | length (smallest unsigned for k) |
| `T[..]` | capacity chosen at construction | capacity+length |
| `T[>..]` | grow-only resizable | length (owner) |
| `T[>..<]` | grow-shrink resizable | length (owner) |

- All non-resizable stored `[metadata][elements...]` inline, packed, no pointer.
- Resizable elements on data stack; owner keeps length.
- `.len` count; limited also `.cap`.
- No padding: unaligned loads/stores allowed.
- Finishing resizable: copy into `T[]`: `let done: i64[] = copy(grow);`.

## Which to use
- `[>..]` default: arenas, pools, builders, scratch.
- `[>..<]` stacks/queues/heaps.
- `[..k]`/`[..]` inline small arrays.
- `[]`/`[varint]` finished data/strings.

## Strings
- No string type; string is u8 array-family.
- `u8[>..]` builder; `u8[]`/`u8[varint]` finished; `u8[..16]` small inline; `u8[:]` view.
- String literal is `const u8[:]` (read-only static).
- `append`, `push`, `format`, `str(...)`.
- `"""` raw multiline strings, indentation of closing `"""` removed.
- Slicing copies nothing.
- `split`, `words`, `trim`, `find`.

## Slices
- `T[:]` reference + count; accepts any array kind.
- Passing array to slice param creates view (no copy).
- `const` qualifier for writability; inferred.
- Slice can be key in dictionary (views, no copies).

## Indexing
- Fixed-size arrays indexable.
- Variable-size elements sequential: iterable but not indexable unless index of refs.
- `^k` counts from end in slicing.

## Operations
- `push`, `pop`, `pop_n`, `clear`, `resize`, `append`.
- `copy(a)` explicit copy.
EOF