#include "config.h"
#include "scene.h"
#include <stdio.h>


// Print the width.  Shallow scenes (width_exp == 0) keep the exact legacy
// `%.17e` form so saved files are unchanged; a deep width is reconstructed
// through MPFR, whose `%Re` prints a scientific exponent far outside the
// double range.
static void
scene_save_width (FILE *f, const struct scene *scene)
{
  if (scene->width_exp == 0)
    {
      fprintf (f, "%.17e\n", scene->width);
      return;
    }

  mpfr_t t;
  mpfr_init2 (t, 256);
  mpfr_set_d (t, scene->width, MPFR_RNDN);
  mpfr_mul_2si (t, t, scene->width_exp, MPFR_RNDN);
  mpfr_fprintf (f, "%.17Re\n", t);
  mpfr_clear (t);
}


// Parse the width line.  Normal doubles take the plain strtod path; a value
// below the f64 normal range (or a subnormal) is parsed through MPFR and
// stored as a normalised mantissa plus exponent.
static bool
scene_load_width (FILE *f, struct scene *scene)
{
  char buf[512];

  if (fscanf (f, "%511s", buf) != 1)
    return false;

  scene->width = 0.0;
  scene->width_exp = 0;

  char *end = NULL;
  const f64 v = strtod (buf, &end);

  if (end != buf && *end == '\0' && isfinite (v) && v >= 1e-300)
    {
      scene->width = v;
      return true;
    }

  if (v < 0.0 || (v == 0.0 && buf[0] == '-'))
    return false;

  mpfr_t t;
  mpfr_init2 (t, 256);

  if (mpfr_set_str (t, buf, 10, MPFR_RNDN) != 0)
    {
      mpfr_clear (t);
      return false;
    }

  if (mpfr_zero_p (t))
    {
      scene->width = 0.0;
      scene->width_exp = 0;
    }
  else if (mpfr_sgn (t) < 0)
    {
      mpfr_clear (t);
      return false;
    }
  else
    {
      mpfr_exp_t ex;
      const f64 m = mpfr_get_d_2exp (&ex, t, MPFR_RNDN);

      scene->width = m;
      scene->width_exp = (i32) ex;
    }

  mpfr_clear (t);
  return true;
}


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
  scene_save_width (f, scene);

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

  if (!scene_load_width (f, scene))
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

