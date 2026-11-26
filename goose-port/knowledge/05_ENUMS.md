# Enums (ADTs)

## Declaration
```goose
enum Shape { Circle { r: f64 }, Rect { w: f64, h: f64 }, Dot }
```
- Only dynamic polymorphism; no inheritance/traits/vtables.
- Case functions: one overload per variant; dispatch via jump table; exhaustiveness checked.

## Two modes
- Fixed mode `Shape`: tag + room for largest payload; indexable; can overwrite element with different variant. Cannot point into payload.
- Variable mode `Shape..`: each value exactly size of variant; packed sequential (not indexable unless index); cannot change variant; may take references into payload.

## match
```goose
match s {
  Circle c => ...,
  Rect r => ...,
  _ => ...,
}
```
- Arm may list several variants: `Circle, Dot => "round"` (no payload binder).
- Match on integers accepts literals, ranges, named constants.
- `&r` binder only on variable-mode payloads.

## Case functions
```goose
fn area(s: Shape.Circle) -> f64 { ... }
fn area(s: Shape.Rect) -> f64 { ... }
```
- Variant types (`Shape.Circle`) are ordinary types.

## Memory
- Variable-mode enums reduce memory/cache (records benchmark: 4x less memory than Rust enum, 2x speed).
EOF