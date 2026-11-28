#ifndef SCENE_H
#define SCENE_H


#include "common.h"
#include <mpfr.h>


#define SCENE_FILE_VERSION 1


struct scene
{
  mpfr_t c_re;
  mpfr_t c_im;

  // Actual view width is `width * 2^width_exp`.  Shallow scenes (the f64
  // range) keep width_exp == 0 so every existing code path is unchanged;
  // a deep scene stores a normalised mantissa in [0.5, 1) plus the exponent,
  // which is what makes depths past ~1e-308 representable (stage 5).
  f64 width;
  i32 width_exp;

  f64 bailout;

  f64 lx;
  f64 ly;
  f64 lz;

  u64 iter_max;
  u64 iter_per_color;

  char palette_path[4096];
};


struct scene *
scene_create (void);

void
scene_destroy (struct scene *scene);

i32
scene_save (const char *path, struct scene *scene);

struct scene *
scene_load (const char *path);


// log2 of the actual view width.  Works for both representations: for a
// shallow scene width_exp == 0, so this is exactly log2(width).
static inline f64
scene_log2_width (const struct scene *scene)
{
  if (scene->width == 0.0)
    return -INFINITY;

  return (f64) scene->width_exp + log2 (scene->width);
}


// A deep scene stores its width as mantissa * 2^exp with exp != 0; shallow
// scenes (the whole f64 range) keep exp == 0 and the legacy f64 width.
static inline bool
scene_is_deep (const struct scene *scene)
{
  return scene->width_exp != 0;
}


#endif // SCENE_H

