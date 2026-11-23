#include "bla.h"

// H = mantissa (double, normalised to [0.5,1)) times 2^e.
struct h
{
  f64 x, y;
  i32 e;
};

static struct h
hnorm (f64 x, f64 y, i32 e)
{
  const f64 m = fmax (fabs (x), fabs (y));

  if (m == 0.0 || !isfinite (m))
    return (struct h) { 0.0, 0.0, 0 };

  i32 shift;

  frexp (m, &shift);

  const f64 sc = ldexp (1.0, -shift);

  return (struct h) { x * sc, y * sc, e + shift };
}

static struct h
hmul (struct h a, struct h b)
{
  return hnorm (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x, a.e + b.e);
}

static struct h
hadd (struct h a, struct h b)
{
  if (a.x == 0.0 && a.y == 0.0)
    return b;

  if (b.x == 0.0 && b.y == 0.0)
    return a;

  const i32 d = a.e - b.e;

  if (d > 80)
    return a;

  if (d < -80)
    return b;

  if (d >= 0)
    {
      const f64 sc = ldexp (1.0, -d);
      return hnorm (a.x + b.x * sc, a.y + b.y * sc, a.e);
    }

  const f64 sc = ldexp (1.0, d);
  return hnorm (a.x * sc + b.x, a.y * sc + b.y, b.e);
}

static struct h
hscale (struct h a, f64 f)
{
  return hnorm (a.x * f, a.y * f, a.e);
}

static f64
hlog2 (struct h v)
{
  const f64 m = v.x * v.x + v.y * v.y;
  return m == 0.0 ? -INFINITY : (f64) v.e + 0.5 * log2 (m);
}

struct step
{
  struct h a, b, c, d, e;
  f64 radius;
  i32 start;
  i32 span;
};

static struct step
compose (struct step f, struct step s)
{
  struct step r = { 0 };

  r.a = hmul (s.a, f.a);
  r.b = hadd (hmul (s.a, f.b), s.b);
  r.c = hadd (hmul (s.a, f.c), hmul (s.c, hmul (f.a, f.a)));
  r.d = hadd (hadd (hmul (s.a, f.d), hscale (hmul (s.c, hmul (f.a, f.b)), 2.0)),
              hmul (s.d, f.a));
  r.e = hadd (hadd (hmul (s.a, f.e), hmul (s.c, hmul (f.b, f.b))),
              hadd (hmul (s.d, f.b), s.e));

  r.start = f.start;
  r.span = f.span + s.span;
  return r;
}

void
bla_clear (struct bla_table *t)
{
  free (t->data);
  free (t->offsets);
  free (t->counts);

  t->data = NULL;
  t->offsets = NULL;
  t->counts = NULL;
  t->levels = 0;
  t->entries = 0;
  t->valid = false;
}

void
bla_build (struct bla_table *t, const f64 *orbit_re, const f64 *orbit_im,
           u64 n, f64 max_delta, f64 margin)
{
  bla_clear (t);

  if (n < (u64) BLA_BASE_STEP + 2)
    return;

  const i32 count = (i32) n - 1;
  const i32 l0 = count / BLA_BASE_STEP;

  if (l0 < 1)
    return;

  const usize cap = (usize) l0 * 2 + 32;

  struct step *st = calloc (cap, sizeof *st);

  t->offsets = calloc (BLA_MAX_LEVELS, sizeof (i32));
  t->counts = calloc (BLA_MAX_LEVELS, sizeof (i32));

  if (!st || !t->offsets || !t->counts)
    {
      free (st);
      bla_clear (t);
      return;
    }

  // TOLERANCE = 2^-16; the first neglected term is cubic because the E/C/D
  // terms are carried, so the per-iteration budget is half the log2 budget.
  const f64 tol_log = -16.0;
  const f64 md_log = max_delta > 0.0 ? log2 (max_delta) : -INFINITY;

  const struct h one = { 1.0, 0.0, 0 };
  const struct h zero = { 0.0, 0.0, 0 };

  i32 used = 0;

  t->offsets[0] = 0;

  for (i32 s = 0; s < l0; ++s)
    {
      const i32 start = s * BLA_BASE_STEP;

      struct h a = one, b = zero, c = zero, d = zero, e = zero;
      f64 radius = INFINITY;

      for (i32 k = 0; k < BLA_BASE_STEP; ++k)
        {
          const f64 zx = 2.0 * orbit_re[start + k];
          const f64 zy = 2.0 * orbit_im[start + k];
          const f64 mag = hypot (zx, zy);

          if (mag == 0.0)
            {
              radius = BLA_NEVER;
              break;
            }

          const f64 budget = (tol_log + log2 (mag)) * 0.5;
          const f64 injected = hlog2 (b) + md_log;

          if (!(injected < budget))
            {
              radius = BLA_NEVER;
              break;
            }

          const f64 headroom = injected == -INFINITY
                                 ? budget
                                 : budget + log2 (1.0 - exp2 (injected - budget));
          const f64 r = headroom - hlog2 (a);

          if (r < radius)
            radius = r;

          const struct h tx = hnorm (zx, zy, 0);
          const struct h nc = hadd (hmul (tx, c), hmul (a, a));
          const struct h nd = hadd (hmul (tx, d), hscale (hmul (a, b), 2.0));
          const struct h ne = hadd (hmul (tx, e), hmul (b, b));

          a = hmul (tx, a);
          b = hadd (hmul (tx, b), one);
          c = nc;
          d = nd;
          e = ne;
        }

      st[used].a = a;
      st[used].b = b;
      st[used].c = c;
      st[used].d = d;
      st[used].e = e;
      st[used].radius = isfinite (radius) ? radius : BLA_NEVER;
      st[used].start = start;
      st[used].span = BLA_BASE_STEP;
      ++used;
    }

  t->counts[0] = used;

  i32 prev_off = 0;
  i32 prev_cnt = used;
  i32 level = 1;

  for (; level < BLA_MAX_LEVELS; ++level)
    {
      const i32 merged = prev_cnt / 2;

      if (merged < 1)
        break;

      if ((usize) used + (usize) merged > cap)
        {
          struct step *bigger = realloc (st, ((usize) used + (usize) merged + 16) * sizeof *st);
          if (!bigger)
            break;
          st = bigger;
        }

      const i32 off = used;

      for (i32 s = 0; s < merged; ++s)
        {
          struct step r = compose (st[prev_off + 2 * s], st[prev_off + 2 * s + 1]);

          const f64 b1d = hlog2 (st[prev_off + 2 * s].b) + md_log;

          if (b1d < st[prev_off + 2 * s + 1].radius)
            {
              const f64 headroom = st[prev_off + 2 * s + 1].radius
                                     + log2 (1.0 - exp2 (b1d - st[prev_off + 2 * s + 1].radius));
              const f64 mapped = headroom - hlog2 (st[prev_off + 2 * s].a);
              const f64 rad = fmin (st[prev_off + 2 * s].radius, mapped) - 1.0;
              r.radius = isfinite (rad) ? rad : BLA_NEVER;
            }
          else
            r.radius = BLA_NEVER;

          st[used++] = r;
        }

      t->offsets[level] = off;
      t->counts[level] = used - off;
      prev_off = off;
      prev_cnt = merged;
    }

  t->levels = level;
  t->entries = used;
  t->margin = margin;

  t->data = calloc ((usize) used * BLA_ENTRY_DOUBLES, sizeof (f64));

  if (!t->data)
    {
      free (st);
      bla_clear (t);
      return;
    }

  for (i32 i = 0; i < used; ++i)
    {
      const usize base = (usize) i * BLA_ENTRY_DOUBLES;
      struct h *co[5] = { &st[i].a, &st[i].b, &st[i].c, &st[i].d, &st[i].e };

      for (i32 k = 0; k < 5; ++k)
        {
          t->data[base + (usize) k * 3 + 0] = co[k]->x;
          t->data[base + (usize) k * 3 + 1] = co[k]->y;
          t->data[base + (usize) k * 3 + 2] = (f64) co[k]->e;
        }

      t->data[base + 15] = st[i].radius;
    }

  free (st);
  t->valid = true;
}

bool
bla_try_skip (const struct bla_table *t, u64 ref_iter,
              f64 *eps_re, f64 *eps_im, f64 *der_re, f64 *der_im,
              f64 dc_re, f64 dc_im, u64 *span)
{
  if (!t->valid || ref_iter % BLA_BASE_STEP != 0)
    return false;

  const u64 unit = ref_iter / BLA_BASE_STEP;

  // Cheap precheck first: converting to extended-exponent form and evaluating
  // the polynomial is only worth doing for a level that actually qualifies.
  // The exponent of |dz|^2 gives log2|dz| with < 0.5 error, which is plenty
  // next to the 16-bit safety margin, and avoids a libm call per attempt.
  const f64 m2 = *eps_re * *eps_re + *eps_im * *eps_im;
  union { f64 d; u64 u; } bits;
  bits.d = m2;
  const f64 dz_log = m2 > 0.0
                       ? 0.5 * (f64) (((i64) ((bits.u >> 52) & 0x7ff) - 1023))
                       : -INFINITY;

  i32 level = t->levels - 1;

  if (unit != 0)
    {
      const i32 zeros = __builtin_ctzll (unit);
      if (zeros < level)
        level = zeros;
    }

  for (; level >= 0; --level)
    {
      const u64 idx = unit >> level;

      if (idx >= (u64) t->counts[level])
        continue;

      const usize base = ((usize) t->offsets[level] + (usize) idx) * BLA_ENTRY_DOUBLES;
      const f64 *d = t->data + base;

      if (!(d[15] > BLA_NEVER) || !(dz_log + t->margin <= d[15]))
        continue;

      struct h dz = hnorm (*eps_re, *eps_im, 0);
      struct h der = hnorm (*der_re, *der_im, 0);
      const struct h delta0 = hnorm (dc_re, dc_im, 0);

      struct h a = { d[0], d[1], (i32) d[2] };
      struct h b = { d[3], d[4], (i32) d[5] };
      struct h c = { d[6], d[7], (i32) d[8] };
      struct h dd = { d[9], d[10], (i32) d[11] };
      struct h e = { d[12], d[13], (i32) d[14] };

      struct h out = hadd (hmul (a, dz), hmul (b, delta0));
      out = hadd (out, hmul (c, hmul (dz, dz)));
      out = hadd (out, hmul (dd, hmul (dz, delta0)));
      out = hadd (out, hmul (e, hmul (delta0, delta0)));

      der = hadd (hmul (a, der), b);
      dz = out;

      *eps_re = ldexp (dz.x, dz.e);
      *eps_im = ldexp (dz.y, dz.e);
      *der_re = ldexp (der.x, der.e);
      *der_im = ldexp (der.y, der.e);
      *span = (u64) BLA_BASE_STEP << level;

      return true;
    }

  return false;
}
