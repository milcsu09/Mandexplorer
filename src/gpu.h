#ifndef GPU_H
#define GPU_H

#include "common.h"
#include "image.h"
#include "palette.h"

struct gpu_view
{
  f64 c_re;
  f64 c_im;
  f64 width;

  u64 iter_max;
  u64 iter_per_color;

  const f64 *orbit_re;
  const f64 *orbit_im;

  f64 delta_c_re;
  f64 delta_c_im;
  f64 period_tol;

  i32 ss;

  enum formula formula;

  f64 omni_a;
  f64 bailout;

  f64 lx;
  f64 ly;
  f64 lz;

  bool color_inside;

  i32 w;
  i32 h;

  const struct palette *palette;
};

bool gpu_init (void);
void gpu_shutdown (void);
bool gpu_is_available (void);
bool gpu_supports_fp64 (void);
bool gpu_should_use (i32 w, i32 h, i32 ss, u64 iter_max, f64 width,
                     enum formula formula, bool vibe);
void gpu_render_mandelbrot (const struct gpu_view *v, struct image *out);
const char *gpu_device_name (void);

#endif // GPU_H