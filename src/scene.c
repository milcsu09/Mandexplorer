#include "config.h"
#include "scene.h"
#include <stdio.h>


struct scene *
scene_create (void)
{
  struct scene *scene = malloc0 (sizeof (struct scene));

  mpfr_init2 (scene->c_re, FLOAT_PREC);
  mpfr_init2 (scene->c_im, FLOAT_PREC);

  return scene;
}


void
scene_destroy (struct scene *scene)
{
  mpfr_clear (scene->c_re);
  mpfr_clear (scene->c_im);

  free (scene);
}


i32
scene_save (const char *path, struct scene *scene)
{
  FILE *f = fopen (path, "w");

  if (!f)
    return 1;

  // VERSION
  fprintf (f, "%" PRIu32 "\n", SCENE_FILE_VERSION);

  // C_RE
  mpfr_out_str (f, 10, 0, scene->c_re, MPFR_RNDN);
  fputc ('\n', f);

  // C_IM
  mpfr_out_str (f, 10, 0, scene->c_im, MPFR_RNDN);
  fputc ('\n', f);

  // WIDTH
  fprintf (f, "%.17e\n", scene->width);

  // BAILOUT
  fprintf (f, "%.17e\n", scene->bailout);

  // LX
  fprintf (f, "%.17e\n", scene->lx);

  // LY
  fprintf (f, "%.17e\n", scene->ly);

  // LZ
  fprintf (f, "%.17e\n", scene->lz);

  // ITER_MAX
  fprintf (f, "%" SCNu64 "\n", scene->iter_max);

  // ITER_PER_COLOR
  fprintf (f, "%" SCNu64 "\n", scene->iter_per_color);

  // PALETTE_PATH
  fprintf (f, "%s\n", scene->palette_path);

  fclose (f);

  return 0;
}


struct scene *
scene_load (const char *path)
{
  FILE *f = fopen (path, "r");

  if (!f)
    return NULL;

  // VERSION

  u32 version;

  if (fscanf (f, "%" SCNu32 "\n", &version) != 1 || version != SCENE_FILE_VERSION)
    {
      fclose (f);

      return NULL;
    }

  struct scene *scene = scene_create ();

  // C_RE

  if (mpfr_inp_str (scene->c_re, f, 10, MPFR_RNDN) == 0)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // C_IM

  if (mpfr_inp_str (scene->c_im, f, 10, MPFR_RNDN) == 0)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // WIDTH

  if (fscanf (f, "%lf\n", &scene->width) != 1)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // BAILOUT

  if (fscanf (f, "%lf\n", &scene->bailout) != 1)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // LX

  if (fscanf (f, "%lf\n", &scene->lx) != 1)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // LY

  if (fscanf (f, "%lf\n", &scene->ly) != 1)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // LZ

  if (fscanf (f, "%lf\n", &scene->lz) != 1)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // ITER_MAX

  if (fscanf (f, "%" SCNu64 "\n", &scene->iter_max) != 1)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // ITER_PER_COLOR

  if (fscanf (f, "%" SCNu64 "\n", &scene->iter_per_color) != 1)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  // PALETTE_PATH

  if (fscanf (f, "%4095[^\n]", scene->palette_path) != 1)
    {
      scene_destroy (scene);

      fclose (f);

      return NULL;
    }

  fclose (f);

  return scene;
}

