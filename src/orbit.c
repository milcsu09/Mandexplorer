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
                mpfr_prec_t prec, bool allow_flat, orbit_cancel_fn cancel,
                void *cancel_arg)
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

  for (u64 iter = 0; iter <= n; ++iter)
    {
      if (cancel && cancel (cancel_arg))
        {
          mpfr_clears (two, z_re, z_im, z_re_new, t, (mpfr_ptr) NULL);
          orbit_release_buffers (o);
          return false;
        }

      if (flat)
        {
          o->flat_re[iter] = mpfr_get_d (z_re, MPFR_RNDN);
          o->flat_im[iter] = mpfr_get_d (z_im, MPFR_RNDN);
        }

      if (iter % o->stride == 0)
        {
          const usize k = (usize) (iter / o->stride);
          mpfr_set (o->re[k], z_re, MPFR_RNDN);
          mpfr_set (o->im[k], z_im, MPFR_RNDN);
        }

      orbit_step (z_re, z_im, two, t, z_re_new, c_re, c_im);
    }

  mpfr_clears (two, z_re, z_im, z_re_new, t, (mpfr_ptr) NULL);

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
