#ifndef ORBIT_H
#define ORBIT_H

#include "common.h"
#include <mpfr.h>

// Stage 1 of the deep-zoom port (docs/RFF_2_ANALYSIS.md).
//
// The reference orbit is stored as exact MPFR states at every ORBIT_STRIDE
// iterations instead of as two f64 arrays of length N.  Doubles for the
// current segment are regenerated on demand with the very same MPFR operations
// and precision used during generation, so every orbit value is bit-identical
// to the old full-array representation.
//
// Small scenes additionally keep an f64 mirror (filled during the same MPFR
// pass, at zero extra cost) so current render speeds and the GPU path stay
// unchanged.  Above ORBIT_FLAT_BUDGET the mirror is omitted and access goes
// through a per-thread regenerating cursor; memory then no longer grows as
// 16 bytes/iteration.

#define ORBIT_STRIDE 4096
// 2^25 iterations need 512 MiB + 16 bytes for the mirror.  Without it BLA
// rebuilds regenerate the whole orbit through MPFR on every vibe frame, which
// is far more expensive than the mirror itself.
#define ORBIT_FLAT_BUDGET ((usize) 640u * 1024u * 1024u)

// A period candidate is an iteration where |z| reaches a new record minimum.
// an/bn are the first-order perturbation coefficients accumulated up to that
// iteration (log2 magnitudes); they are the raw material for stage 3 (PA).
struct orbit_period
{
  u64 iter;
  f64 z2;
  f64 an_log2;
  f64 bn_log2;
};

struct orbit
{
  bool valid;
  u64 n;                 // highest valid iteration index
  u64 stride;            // checkpoint interval
  mpfr_prec_t prec;      // precision of the generation pass
  usize count;           // number of checkpoints
  mpfr_t *re;            // exact z_re at checkpoint k*stride
  mpfr_t *im;            // exact z_im at checkpoint k*stride
  mpfr_t c_re;           // exact centre the orbit was generated for
  mpfr_t c_im;
  f64 *flat_re;          // optional full f64 mirror (NULL for large N)
  f64 *flat_im;
  u64 flat_cap;          // allocated iterations, reused across regenerations
  u64 version;           // bumped whenever the orbit is regenerated

  // Stage 2 (FPG): record-min period candidates and the first iteration where
  // the linear perturbation no longer covers dcMax.
  struct orbit_period *periods;
  usize period_count;
  usize period_cap;
  u64 fpg_limit;
};

// Return true to abandon generation (the caller changed generation).
typedef bool (*orbit_cancel_fn) (void *arg);

void orbit_init (struct orbit *o);
void orbit_clear (struct orbit *o);
void orbit_invalidate (struct orbit *o);

bool orbit_generate (struct orbit *o, u64 n, const mpfr_t c_re,
                     const mpfr_t c_im, mpfr_prec_t prec, f64 dc_max,
                     bool allow_flat, orbit_cancel_fn cancel, void *cancel_arg);

// Thread-safe: each calling thread owns its own regenerating window.
bool orbit_at (struct orbit *o, u64 j, f64 *re, f64 *im);

bool orbit_has_flat (const struct orbit *o);
u64 orbit_stride (const struct orbit *o);

usize orbit_period_count (const struct orbit *o);
bool orbit_period_at (const struct orbit *o, usize i, struct orbit_period *out);
u64 orbit_fpg_limit (const struct orbit *o);

#endif // ORBIT_H
