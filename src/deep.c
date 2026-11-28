#include "deep.h"


// Exponent gap past which the smaller addend cannot change the larger one at
// double precision; matching bla.c / pa.c so all extended types agree.
#define DEEP_ADD_GAP 80


struct dxr
dxr_set (f64 m, i32 e)
{
  if (m == 0.0 || !isfinite (m))
    return (struct dxr) { 0.0, 0 };

  i32 shift;
  frexp (m, &shift);

  return (struct dxr) { ldexp (m, -shift), e + shift };
}


struct dxr
dxr_from_d (f64 v)
{
  return dxr_set (v, 0);
}


struct dxr
dxr_mul (struct dxr a, struct dxr b)
{
  return dxr_set (a.m * b.m, a.e + b.e);
}


struct dxr
dxr_div (struct dxr a, struct dxr b)
{
  if (b.m == 0.0)
    return (struct dxr) { 0.0, 0 };

  return dxr_set (a.m / b.m, a.e - b.e);
}


f64
dxr_log2 (struct dxr a)
{
  if (a.m == 0.0)
    return -INFINITY;

  return log2 (fabs (a.m)) + (f64) a.e;
}


f64
dxr_to_d (struct dxr a)
{
  return ldexp (a.m, a.e);
}


bool
dxr_is_zero (struct dxr a)
{
  return a.m == 0.0;
}


struct dcx
dcx_norm (f64 x, f64 y, i32 e)
{
  const f64 m = fmax (fabs (x), fabs (y));

  if (m == 0.0 || !isfinite (m))
    return (struct dcx) { 0.0, 0.0, 0 };

  i32 shift;
  frexp (m, &shift);

  const f64 sc = ldexp (1.0, -shift);

  return (struct dcx) { x * sc, y * sc, e + shift };
}


struct dcx
dcx_from_d (f64 x, f64 y)
{
  return dcx_norm (x, y, 0);
}


struct dcx
dcx_add (struct dcx a, struct dcx b)
{
  if (a.x == 0.0 && a.y == 0.0)
    return b;
  if (b.x == 0.0 && b.y == 0.0)
    return a;

  const i32 d = a.e - b.e;

  if (d > DEEP_ADD_GAP)
    return a;
  if (d < -DEEP_ADD_GAP)
    return b;

  if (d >= 0)
    {
      const f64 sc = ldexp (1.0, -d);
      return dcx_norm (a.x + b.x * sc, a.y + b.y * sc, a.e);
    }

  const f64 sc = ldexp (1.0, d);
  return dcx_norm (a.x * sc + b.x, a.y * sc + b.y, b.e);
}


struct dcx
dcx_neg (struct dcx a)
{
  if (a.x == 0.0 && a.y == 0.0)
    return a;

  return (struct dcx) { -a.x, -a.y, a.e };
}


struct dcx
dcx_conj (struct dcx a)
{
  return (struct dcx) { a.x, -a.y, a.e };
}


struct dcx
dcx_sub (struct dcx a, struct dcx b)
{
  return dcx_add (a, dcx_neg (b));
}


struct dcx
dcx_mul (struct dcx a, struct dcx b)
{
  return dcx_norm (a.x * b.x - a.y * b.y,
                   a.x * b.y + a.y * b.x,
                   a.e + b.e);
}


struct dcx
dcx_mul_d (struct dcx a, f64 x, f64 y)
{
  return dcx_norm (a.x * x - a.y * y, a.x * y + a.y * x, a.e);
}


struct dcx
dcx_add_d (struct dcx a, f64 x, f64 y)
{
  return dcx_add (a, dcx_from_d (x, y));
}


struct dcx
dcx_scale2 (struct dcx a)
{
  if (a.x == 0.0 && a.y == 0.0)
    return a;

  // Doubling only bumps the exponent; the mantissa stays in [0.5, 1).
  return (struct dcx) { a.x, a.y, a.e + 1 };
}


f64
dcx_log2_abs (struct dcx a)
{
  const f64 m = fmax (fabs (a.x), fabs (a.y));

  if (m == 0.0)
    return -INFINITY;

  return log2 (m) + (f64) a.e;
}


f64
dcx_re (struct dcx a)
{
  return ldexp (a.x, a.e);
}


f64
dcx_im (struct dcx a)
{
  return ldexp (a.y, a.e);
}


bool
dcx_is_zero (struct dcx a)
{
  return a.x == 0.0 && a.y == 0.0;
}
