#include "orbit.h"
#include "config.h"

// One step of z <- z^2 + c.  Generation and segment regeneration MUST share
// this function: MPFR is deterministic, so identical operations at identical
// precision reproduce bit-identical doubles.
static inline void
orbit_step (mpfr_t z_re, mpfr_t z_im, mpfr_t two, mpfr_t t, mpfr_t z_re_new,
            const mpfr_t c_re, const mpfr_t c_im)
{
  mpfr_sqr (t, z_im, MPFR_RNDN);
  mpfr_fms (z_re_new, z_re, z_re, t, MPFR_RNDN);
  mpfr_mul (t, z_re, z_im, MPFR_RNDN);
  mpfr_fma (z_im, t, two, c_im, MPFR_RNDN);
  mpfr_add (z_re, z_re_new, c_re, MPFR_RNDN);
}


// Complex value as mantissa pair times 2^e.  FPG products of 2z overflow a
// plain double at depth, exactly like the BLA coefficients in bla.c.
struct cex
{
  f64 x, y;
  i32 e;
};

static struct cex
cex_norm (f64 x, f64 y, i32 e)
{
  const f64 m = fmax (fabs (x), fabs (y));

  if (m == 0.0 || !isfinite (m))
    return (struct cex) { 0.0, 0.0, 0 };

  i32 shift;
  frexp (m, &shift);

  const f64 sc = ldexp (1.0, -shift);
  return (struct cex) { x * sc, y * sc, e + shift };
}


static struct cex
cex_mul_complex (struct cex a, f64 x, f64 y)
{
  return cex_norm (a.x * x - a.y * y, a.x * y + a.y * x, a.e);
}


static struct cex
cex_add_one (struct cex a)
{
  const i32 d = a.e;

  if (d > 80)
    return a;

  const f64 sc = ldexp (1.0, -d);
  return cex_norm (a.x + sc, a.y, a.e);
}


static f64
cex_log2 (struct cex v)
{
  const f64 m = v.x * v.x + v.y * v.y;
  return m == 0.0 ? -INFINITY : (f64) v.e + 0.5 * log2 (m);
}


static void
orbit_release_checkpoints (struct orbit *o)
{
  if (o->re)
    {
      for (usize i = 0; i < o->count; ++i)
        mpfr_clear (o->re[i]);

      free (o->re);
      o->re = NULL;
    }

  if (o->im)
    {
      for (usize i = 0; i < o->count; ++i)
        mpfr_clear (o->im[i]);

      free (o->im);
      o->im = NULL;
    }

  o->count = 0;
}


static void
orbit_release_buffers (struct orbit *o)
{
  orbit_release_checkpoints (o);

  free (o->periods);
  o->periods = NULL;
  o->period_count = 0;
  o->period_cap = 0;
  o->fpg_limit = 0;

  free (o->flat_re);
  free (o->flat_im);
  o->flat_re = NULL;
  o->flat_im = NULL;
  o->flat_cap = 0;
}


void
orbit_init (struct orbit *o)
{
  memset (o, 0, sizeof *o);

  o->stride = ORBIT_STRIDE;

  const char *env = getenv ("MANDEX_ORBIT_STRIDE");

  if (env)
    {
      const long v = strtol (env, NULL, 10);

      if (v >= 32 && v <= (1L << 20))
        o->stride = (u64) v;
    }

  mpfr_inits2 (FLOAT_PREC, o->c_re, o->c_im, (mpfr_ptr) NULL);
}


void
orbit_clear (struct orbit *o)
{
  orbit_release_buffers (o);
  mpfr_clears (o->c_re, o->c_im, (mpfr_ptr) NULL);
  memset (o, 0, sizeof *o);
}


void
orbit_invalidate (struct orbit *o)
{
  orbit_release_buffers (o);
  o->valid = false;
  ++o->version;
}


bool
orbit_generate (struct orbit *o, u64 n, const mpfr_t c_re, const mpfr_t c_im,
                mpfr_prec_t prec, f64 dc_max, bool allow_flat,
                orbit_cancel_fn cancel, void *cancel_arg)
{
  orbit_release_checkpoints (o);
  o->valid = false;
  o->n = n;
  o->prec = prec;
  o->count = n / o->stride + 1;

  mpfr_set (o->c_re, c_re, MPFR_RNDN);
  mpfr_set (o->c_im, c_im, MPFR_RNDN);

  const bool flat = allow_flat && (u64) (n + 1) * 16u <= (u64) ORBIT_FLAT_BUDGET;

  if (flat)
    {
      // Keep the old buffers when their size still fits.  An asynchronous GPU
      // job may still be reading them from the previous frame; the old full
      // arrays were overwritten in place for exactly this reason.
      if (!o->flat_re || o->flat_cap < n + 1)
        {
          free (o->flat_re);
          free (o->flat_im);

          o->flat_re = calloc ((usize) n + 1, sizeof (f64));
          o->flat_im = calloc ((usize) n + 1, sizeof (f64));
          o->flat_cap = n + 1;
        }

      if (!o->flat_re || !o->flat_im)
        {
          orbit_release_buffers (o);
          return false;
        }
    }
  else
    {
      free (o->flat_re);
      free (o->flat_im);
      o->flat_re = NULL;
      o->flat_im = NULL;
      o->flat_cap = 0;
    }

  o->re = malloc (o->count * sizeof (mpfr_t));
  o->im = malloc (o->count * sizeof (mpfr_t));

  if (!o->re || !o->im)
    {
      orbit_release_buffers (o);
      return false;
    }

  for (usize k = 0; k < o->count; ++k)
    {
      mpfr_init2 (o->re[k], prec);
      mpfr_init2 (o->im[k], prec);
    }

  mpfr_t two, z_re, z_im, z_re_new, t;

  mpfr_inits2 (prec, two, z_re, z_im, z_re_new, t, (mpfr_ptr) NULL);

  mpfr_set_d (two, 2.0, MPFR_RNDN);
  mpfr_set_d (z_re, 0.0, MPFR_RNDN);
  mpfr_set_d (z_im, 0.0, MPFR_RNDN);

  // FPG state (RFF MB2ReferenceGenerator::tryUpdateAnBn).  an = Π(2z_k),
  // bn = bn*2z_k + 1 is the first-order perturbation w.r.t. c.  The guard
  // |bn|*dcMax > |z|^2 marks where a linear skip from this reference stops
  // being valid for the tile.  Nothing here changes the orbit itself: the
  // stage-2 port only records the data stage 3 will consume.
  // The extended-exponent products cost ~2 frexp/ldexp/log2 per iteration,
  // which is noticeable at 2^25.  Stage 2 is groundwork for PA, so keep it
  // opt-in until stage 3 actually consumes the coefficients.
  const bool track_fpg = getenv ("MANDEX_FPG") != NULL;
  const f64 dc_log2 = track_fpg && dc_max > 0.0 ? log2 (dc_max) : -INFINITY;
  struct cex an = { 1.0, 0.0, 0 };
  struct cex bn = { 0.0, 0.0, 0 };
  f64 min_z2 = 1.0;

  for (u64 iter = 0; iter <= n; ++iter)
    {
      if (cancel && cancel (cancel_arg))
        {
          mpfr_clears (two, z_re, z_im, z_re_new, t, (mpfr_ptr) NULL);
          orbit_release_buffers (o);
          return false;
        }

      const f64 zr = mpfr_get_d (z_re, MPFR_RNDN);
      const f64 zi = mpfr_get_d (z_im, MPFR_RNDN);

      if (flat)
        {
          o->flat_re[iter] = zr;
          o->flat_im[iter] = zi;
        }

      if (iter % o->stride == 0)
        {
          const usize k = (usize) (iter / o->stride);
          mpfr_set (o->re[k], z_re, MPFR_RNDN);
          mpfr_set (o->im[k], z_im, MPFR_RNDN);
        }

      const f64 z2 = zr * zr + zi * zi;

      if (iter > 0)
        {
          f64 bn_log2 = 0.0;

          if (track_fpg)
            {
              const struct cex bn_next =
                cex_add_one (cex_mul_complex (bn, 2.0 * zr, 2.0 * zi));

              bn_log2 = cex_log2 (bn_next);

              const f64 z2_log2 = z2 > 0.0 ? log2 (z2) : -INFINITY;

              if (!o->fpg_limit && bn_log2 + dc_log2 > z2_log2)
                o->fpg_limit = iter;

              an = cex_mul_complex (an, 2.0 * zr, 2.0 * zi);
              bn = bn_next;
            }

          // record-minima period candidates (RFF tryAppendPeriodCandidate).
          // This is a single comparison per iteration, so it stays on.
          if (min_z2 > z2)
            {
              min_z2 = z2;

              if (o->period_count == o->period_cap)
                {
                  const usize cap = o->period_cap ? o->period_cap * 2 : 64;
                  struct orbit_period *p =
                    realloc (o->periods, cap * sizeof *p);

                  if (p)
                    {
                      o->periods = p;
                      o->period_cap = cap;
                    }
                }

              if (o->period_count < o->period_cap)
                {
                  struct orbit_period *q = &o->periods[o->period_count++];
                  q->iter = iter;
                  q->z2 = z2;
                  q->an_log2 = track_fpg ? cex_log2 (an) : 0.0;
                  q->bn_log2 = bn_log2;
                }
            }
        }

      orbit_step (z_re, z_im, two, t, z_re_new, c_re, c_im);
    }

  mpfr_clears (two, z_re, z_im, z_re_new, t, (mpfr_ptr) NULL);

  if (!o->fpg_limit)
    o->fpg_limit = n;

  const char *periods_env = getenv ("MANDEX_DUMP_PERIODS");

  if (periods_env)
    {
      long lim = strtol (periods_env, NULL, 10);

      if (lim <= 0)
        lim = 32;

      fprintf (stderr,
               "FPG: %zu candidates, fpg_limit=%llu of %llu iterations\n",
               o->period_count, (unsigned long long) o->fpg_limit,
               (unsigned long long) n);

      for (usize i = 0; i < o->period_count && (long) i < lim; ++i)
        {
          const struct orbit_period *q = &o->periods[i];
          fprintf (stderr,
                   "  #%zu iter=%llu z2=%.3e an=%.2f bn=%.2f\n", i,
                   (unsigned long long) q->iter, q->z2, q->an_log2,
                   q->bn_log2);
        }
    }

  ++o->version;
  o->valid = true;
  return true;
}


// Per-thread window.  A window covers [base, base+stride] inclusive so the
// renderer's consecutive orbit[j] / orbit[j+1] reads never cross a boundary.
static _Thread_local struct
{
  struct orbit *owner;
  u64 version;
  u64 base;
  u64 len;
  f64 *re;
  f64 *im;
  usize cap;
} cursor;


static bool
orbit_regen (struct orbit *o, u64 j)
{
  const u64 base = (j / o->stride) * o->stride;
  u64 end = base + o->stride;

  if (end > o->n)
    end = o->n;

  const u64 len = end - base + 1;

  if (len > cursor.cap)
    {
      f64 *re = realloc (cursor.re, (usize) len * sizeof (f64));

      if (!re)
        return false;

      cursor.re = re;

      f64 *im = realloc (cursor.im, (usize) len * sizeof (f64));

      if (!im)
        return false;

      cursor.im = im;
      cursor.cap = (usize) len;
    }

  mpfr_t two, z_re, z_im, z_re_new, t;

  mpfr_inits2 (o->prec, two, z_re, z_im, z_re_new, t, (mpfr_ptr) NULL);

  mpfr_set_d (two, 2.0, MPFR_RNDN);

  const usize k = (usize) (base / o->stride);

  mpfr_set (z_re, o->re[k], MPFR_RNDN);
  mpfr_set (z_im, o->im[k], MPFR_RNDN);

  for (u64 i = 0; i < len; ++i)
    {
      cursor.re[i] = mpfr_get_d (z_re, MPFR_RNDN);
      cursor.im[i] = mpfr_get_d (z_im, MPFR_RNDN);

      if (i + 1 < len)
        orbit_step (z_re, z_im, two, t, z_re_new, o->c_re, o->c_im);
    }

  mpfr_clears (two, z_re, z_im, z_re_new, t, (mpfr_ptr) NULL);

  cursor.owner = o;
  cursor.version = o->version;
  cursor.base = base;
  cursor.len = len;
  return true;
}


bool
orbit_at (struct orbit *o, u64 j, f64 *re, f64 *im)
{
  if (!o || !o->valid || j > o->n)
    return false;

  if (o->flat_re)
    {
      *re = o->flat_re[j];
      *im = o->flat_im[j];
      return true;
    }

  if (cursor.owner != o || cursor.version != o->version
      || j < cursor.base || j >= cursor.base + cursor.len)
    {
      if (!orbit_regen (o, j))
        return false;
    }

  *re = cursor.re[j - cursor.base];
  *im = cursor.im[j - cursor.base];
  return true;
}


bool
orbit_has_flat (const struct orbit *o)
{
  return o && o->flat_re && o->flat_im;
}


u64
orbit_stride (const struct orbit *o)
{
  return o ? o->stride : ORBIT_STRIDE;
}


usize
orbit_period_count (const struct orbit *o)
{
  return o ? o->period_count : 0;
}


bool
orbit_period_at (const struct orbit *o, usize i, struct orbit_period *out)
{
  if (!o || !out || i >= o->period_count)
    return false;

  *out = o->periods[i];
  return true;
}


u64
orbit_fpg_limit (const struct orbit *o)
{
  return o ? o->fpg_limit : 0;
}
