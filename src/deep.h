#ifndef DEEP_H
#define DEEP_H


#include "common.h"


// Stage 5 of the deep-zoom port (docs/RFF_2_ANALYSIS.md).
//
// A mantissa+exponent real and its complex counterpart for values whose
// magnitude leaves the f64 range: views deeper than ~1e-308 and the
// perturbations / derivatives they produce.  `m` is a double normalised to
// [0.5, 1) (or exactly 0, which is the zero value); the represented number is
// `m * 2^e` with `e` a full-width i32.  All arithmetic keeps the mantissa
// normalised, so a value such as 1e-400 (m ~ 0.93, e ~ -1329) is exact in the
// exponent and carries a full double mantissa.
//
// This mirrors the extended-exponent coefficients that `bla.c` / `pa.c`
// already use internally, promoted to a shared type for the pixel loop.

struct dxr
{
  f64 m;
  i32 e;
};

struct dxr dxr_from_d (f64 v);
struct dxr dxr_set (f64 m, i32 e);
struct dxr dxr_mul (struct dxr a, struct dxr b);
struct dxr dxr_div (struct dxr a, struct dxr b);
f64        dxr_log2 (struct dxr a);
f64        dxr_to_d (struct dxr a);
bool       dxr_is_zero (struct dxr a);


// Complex value (x + i y) * 2^e.
struct dcx
{
  f64 x, y;
  i32 e;
};

struct dcx dcx_from_d (f64 x, f64 y);
struct dcx dcx_norm (f64 x, f64 y, i32 e);
struct dcx dcx_add (struct dcx a, struct dcx b);
struct dcx dcx_sub (struct dcx a, struct dcx b);
struct dcx dcx_neg (struct dcx a);
struct dcx dcx_conj (struct dcx a);
struct dcx dcx_mul (struct dcx a, struct dcx b);
struct dcx dcx_mul_d (struct dcx a, f64 x, f64 y);
struct dcx dcx_add_d (struct dcx a, f64 x, f64 y);
struct dcx dcx_scale2 (struct dcx a);
f64        dcx_log2_abs (struct dcx a);
f64        dcx_re (struct dcx a);
f64        dcx_im (struct dcx a);
bool       dcx_is_zero (struct dcx a);


#endif // DEEP_H
