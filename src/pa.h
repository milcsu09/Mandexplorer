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


// Stage 4: multi-periodic approximation (RFF MB2MPAGenerator, uncompressed
// path).  The table is built from the record-minima period candidates already
// collected by the orbit.  Levels are exponential (each within MPA_MAX_MULT of
// the next, real periods plus artificial ones), and a closed-form mixed-radix
// mapper (iteration -> table offset) is used both while building and at render
// time.  The table therefore costs O(levels x periods) memory instead of the
// O(N) of the naive PA table or BLA.

#define MPA_MAX_LEVELS 64
#define MPA_MAX_MULT 2
#define MPA_MIN_SKIP 4
#define MPA_PERTURB_REQ 2
#define MPA_MAX_TABLE (1u << 22)

struct mpa_entry
{
  u64 skip;
  struct pa_z an;
  struct pa_z bn;
  f64 radius;      // log2 bound on |eps|
};

struct mpa_table
{
  u64 *periods;        // one period per level
  bool *artificial;    // level is an inserted (non-reference) period
  u64 *skippable;      // iterations covered within one period of this level
  u64 *elements;       // cumulative table entry count through this level
  usize levels;
  struct mpa_entry *table;
  usize table_len;
  f64 epsilon;
  bool valid;
};

void mpa_init (struct mpa_table *t);
void mpa_clear (struct mpa_table *t);

// `dc_max` must bound |dc| over the whole image (same value BLA is built with).
bool mpa_build (struct mpa_table *t, struct orbit *o, f64 dc_max,
                f64 epsilon);

// Runtime lookup.  Only usable when `ref_iter` lands on a mixed-radix boundary;
// returns the longest guard-valid skip from `ref_iter`, updating eps and deriv.
bool mpa_try_skip (const struct mpa_table *t, u64 ref_iter,
                   f64 eps_re, f64 eps_im, f64 deriv_re, f64 deriv_im,
                   f64 dc_re, f64 dc_im,
                   f64 *new_eps_re, f64 *new_eps_im,
                   f64 *new_deriv_re, f64 *new_deriv_im,
                   u64 *skip);

usize mpa_level_count (const struct mpa_table *t);

// Debug: compare each readable table entry against the exact quadratic
// perturbation over the same reference range.  Prints the worst error.
void mpa_selftest (const struct mpa_table *t, struct orbit *o, f64 dc);

#endif // PA_H
