#ifndef SCENE_H
#define SCENE_H


#include "common.h"
#include <mpfr.h>


#define SCENE_FILE_VERSION 1


struct scene
{
  mpfr_t c_re;
  mpfr_t c_im;

  f64 width;

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


#endif // SCENE_H

