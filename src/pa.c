#include "pa.h"

static struct pa_z
pa_norm (f64 x, f64 y, i32 e)
{
  const f64 m = fmax (fabs (x), fabs (y));

  if (m == 0.0 || !isfinite (m))
    return (struct pa_z) { 0.0, 0.0, 0 };

  i32 shift;
  frexp (m, &shift);

  const f64 sc = ldexp (1.0, -shift);
  return (struct pa_z) { x * sc, y * sc, e + shift };
}


static struct pa_z
pa_mul_complex (struct pa_z a, f64 x, f64 y)
{
  return pa_norm (a.x * x - a.y * y, a.x * y + a.y * x, a.e);
}


static struct pa_z
pa_add (struct pa_z a, struct pa_z b)
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
      return pa_norm (a.x + b.x * sc, a.y + b.y * sc, a.e);
    }

  const f64 sc = ldexp (1.0, d);
  return pa_norm (a.x * sc + b.x, a.y * sc + b.y, b.e);
}


static struct pa_z
pa_add_one (struct pa_z a)
{
  const struct pa_z one = { 1.0, 0.0, 0 };
  return pa_add (a, one);
}


static f64
pa_log2 (struct pa_z v)
{
  const f64 m = v.x * v.x + v.y * v.y;
  return m == 0.0 ? -INFINITY : (f64) v.e + 0.5 * log2 (m);
}


static void
pa_apply (struct pa_z a, f64 x, f64 y, f64 *out_re, f64 *out_im)
{
  const struct pa_z r = pa_mul_complex (a, x, y);
  *out_re = ldexp (r.x, r.e);
  *out_im = ldexp (r.y, r.e);
}


void
pa_init (struct pa_table *t)
{
  memset (t, 0, sizeof *t);
}


void
pa_clear (struct pa_table *t)
{
  free (t->entries);
  memset (t, 0, sizeof *t);
}


bool
pa_build (struct pa_table *t, struct orbit *o, f64 dc_max, f64 epsilon)
{
  pa_clear (t);

  const u64 n = o->n;

  if (!(epsilon > 0.0) || n < (u64) PA_ALIGN * 2)
    return false;

  // RFF never looks up PA at refIteration 0.  We build a bounded lookahead PA
  // for EVERY BLA-aligned start, so a BLA skip always lands on a usable entry;
  // the finite horizon keeps build cost O(N) and the radius guard keeps it
  // safe.  Period candidates are still used elsewhere (stage 4 MPA).
  usize count = (usize) (n / PA_ALIGN);

  if (count > PA_MAX_ENTRIES)
    count = PA_MAX_ENTRIES;

  t->entries = calloc (count, sizeof *t->entries);

  if (!t->entries)
    return false;

  t->count = count;

  const f64 dc_log2 = dc_max > 0.0 ? log2 (dc_max) : -INFINITY;
  const f64 eps_log2 = log2 (epsilon);
  const u64 aligned_n = n - n % PA_ALIGN;

  // idx 0 would start at iteration 0, which RFF explicitly rejects.
  for (usize idx = 1; idx < count; ++idx)
    {
      const u64 start = (u64) idx * PA_ALIGN;
      u64 end = start + PA_SKIP_CAP;

      if (end > aligned_n)
        end = aligned_n;

      struct pa_entry *q = &t->entries[idx];

      struct pa_z an = { 1.0, 0.0, 0 };
      struct pa_z bn = { 0.0, 0.0, 0 };
      struct pa_z best_an = an;
      struct pa_z best_bn = bn;
      f64 radius = INFINITY;
      f64 best_radius = 0.0;
      u64 best_len = 0;

      for (u64 j = start; j < end; ++j)
        {
          f64 rr, ri;

          if (!orbit_at (o, j, &rr, &ri))
            {
              pa_clear (t);
              return false;
            }

          const f64 z2r = 2.0 * rr;
          const f64 z2i = 2.0 * ri;

          if (z2r != 0.0 || z2i != 0.0)
            {
              const f64 b = pa_log2 (bn) + dc_log2;
              const f64 a = eps_log2 + log2 (hypot (z2r, z2i));
              const f64 an_log2 = pa_log2 (an);

              if (b >= a || an_log2 == -INFINITY)
                radius = 0.0;
              else
                {
                  const f64 corr =
                    a - b > 80.0 ? 0.0 : log2 (1.0 - exp2 (b - a));
                  radius = fmin (radius, a + corr - an_log2);
                }
            }

          // The guard only tightens, so the first failure ends this entry.
          if (radius == 0.0)
            break;

          an = pa_mul_complex (an, z2r, z2i);
          bn = pa_add_one (pa_mul_complex (bn, z2r, z2i));

          // Only keep snapshots on the BLA grid, so a skip never breaks BLA
          // alignment and the stored coefficients match the stored length.
          if ((j - start + 1) % PA_ALIGN == 0)
            {
              best_len = j - start + 1;
              best_radius = radius;
              best_an = an;
              best_bn = bn;
            }
        }

      q->start = start;
      q->skip = best_len;
      q->an = best_an;
      q->bn = best_bn;
      q->radius = best_len ? best_radius : 0.0;
    }

  t->valid = true;
  return true;
}


bool
pa_try_skip (const struct pa_table *t, u64 iter_orbit,
             f64 eps_re, f64 eps_im, f64 dc_re, f64 dc_im,
             f64 deriv_re, f64 deriv_im,
             f64 *new_eps_re, f64 *new_eps_im,
             f64 *new_deriv_re, f64 *new_deriv_im,
             u64 *skip)
{
  if (!t || !t->valid || t->count == 0 || iter_orbit < PA_ALIGN
      || iter_orbit % PA_ALIGN != 0)
    return false;

  const usize idx = (usize) (iter_orbit / PA_ALIGN);

  if (idx >= t->count)
    return false;

  const struct pa_entry *q = &t->entries[idx];

  const f64 emag = hypot (eps_re, eps_im);
  const f64 elog = emag > 0.0 ? log2 (emag) : -INFINITY;

  if (q->start != iter_orbit || q->radius == 0.0 || !(elog < q->radius))
    return false;

  *skip = q->skip;

  pa_apply (q->an, eps_re, eps_im, new_eps_re, new_eps_im);
  pa_apply (q->an, deriv_re, deriv_im, new_deriv_re, new_deriv_im);

  // eps' = an*eps + bn*dc, D' = an*D + bn (bn is the derivative
  // contribution of an eps of 1).
  f64 er, ei, br, bi;
  pa_apply (q->bn, dc_re, dc_im, &er, &ei);
  pa_apply (q->bn, 1.0, 0.0, &br, &bi);
  *new_eps_re += er;
  *new_eps_im += ei;
  *new_deriv_re += br;
  *new_deriv_im += bi;

  return true;
}
