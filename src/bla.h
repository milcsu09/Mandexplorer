#ifndef BLA_H
#define BLA_H

#include "common.h"

// Second-order bilinear approximation ("BLA") of a reference-orbit range.
// A skip maps the travelling perturbation w and the pixel offset d by
//   w_out = A*w + B*d + C*w^2 + D*w*d + E*d^2
// and carries an entry radius (log2) for which the truncation stays small.
// Coefficients use a double mantissa with an explicit binary exponent:
// products of 2*Z over a range overflow plain double at depth.

#define BLA_BASE_STEP 32
#define BLA_MAX_LEVELS 16
#define BLA_ENTRY_DOUBLES 20
#define BLA_NEVER (-1e30)

struct bla_table
{
  f64 *data;          // BLA_ENTRY_DOUBLES per entry
  i32 *offsets;       // BLA_MAX_LEVELS
  i32 *counts;        // BLA_MAX_LEVELS
  i32 levels;
  i32 entries;
  f64 margin;         // log2 safety margin for the radius test
  bool valid;
};

void bla_build (struct bla_table *t, const f64 *orbit_re, const f64 *orbit_im,
                u64 n, f64 max_delta, f64 margin);
void bla_clear (struct bla_table *t);

bool bla_try_skip (const struct bla_table *t, u64 ref_iter,
                   f64 *eps_re, f64 *eps_im, f64 *der_re, f64 *der_im,
                   f64 dc_re, f64 dc_im, u64 *span);

#endif // BLA_H
