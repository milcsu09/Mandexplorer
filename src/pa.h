#ifndef PA_H
#define PA_H

#include "common.h"
#include "orbit.h"

// Stage 3 of the deep-zoom port: first-order periodic approximation, after
// RFF's PA / PAGenerator.  For a reference segment of length k,
//
//   eps_{i+k} ~= an * eps_i + bn * dc
//   D_{i+k}   ~= an * D_i   + bn
//
// with an = prod(2*Z_j) and bn = bn*2*Z_j + 1.  Coefficients use a double
// mantissa plus an explicit binary exponent because the products overflow a
// plain double at depth.  `radius` is a log2 bound on |eps| for which the
// discarded quadratic term stays below `epsilon` of |2*Z_j| at every step.

// BLA walks the reference in 32-step blocks; PA entry boundaries are aligned
// to the same grid so the renderer can actually land on them.
#define PA_ALIGN 32
#define PA_SKIP_CAP 4096
// Naive per-start construction is O(entries * skip); cap it so enabling PA
// never dominates a 2^25 render.  Stage 4 replaces this with amortized build.
#define PA_MAX_ENTRIES 8192

struct pa_z
{
  f64 x, y;
  i32 e;
};

struct pa_entry
{
  u64 start;       // reference iteration the coefficients were built from
  u64 skip;
  struct pa_z an;
  struct pa_z bn;
  f64 radius;      // log2 bound on |eps|
};

struct pa_table
{
  struct pa_entry *entries;
  usize count;
  bool valid;
};

void pa_init (struct pa_table *t);
void pa_clear (struct pa_table *t);

// Build one entry per record-min period candidate.  `epsilon` is the allowed
// relative error of the dropped quadratic term (e.g. 2^-30).
bool pa_build (struct pa_table *t, struct orbit *o, f64 dc_max, f64 epsilon);

// Try the longest valid entry.  Entries start at iteration 0, so this is only
// applicable when the perturbation has been rebased to the reference start.
bool pa_try_skip (const struct pa_table *t, u64 iter_orbit,
                  f64 eps_re, f64 eps_im, f64 dc_re, f64 dc_im,
                  f64 deriv_re, f64 deriv_im,
                  f64 *new_eps_re, f64 *new_eps_im,
                  f64 *new_deriv_re, f64 *new_deriv_im,
                  u64 *skip);

#endif // PA_H
