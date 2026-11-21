#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <mpfr.h>
#include <stdatomic.h>
#include <time.h>

#include "color.h"
#include "common.h"
#include "config.h"
#include "image.h"
#include "palette.h"
#include "scene.h"
#include "thread_pool.h"


/// General configuration


#define FONT_PATH "res/DMSans-Regular.ttf"


// #define FONT_PATH "res/Iosevka-Regular.ttf"

#define MAX_ITER_MAX 16777216

// Omnibrot: z <- z^2 * exp (a / z) + c
#define OMNIBROT_A_DEFAULT 0.005

// Adaptive iteration control.
//
// The probe compares the image at N and 4N on a coarse grid and measures the
// fraction of pixels whose colour changes visibly.  The controller only ever
// grows N; it never lowers it automatically.  Growth is capped by a render
// time budget, so a slow frame can never trigger a zoom-quality regression.
#define AUTO_BUDGET_MS        6000
#define AUTO_TARGET_MS        4000
#define AUTO_HARD_MS          60000
#define AUTO_SENSITIVITY_LO   0.005
#define AUTO_SENSITIVITY_HI   0.020
#define AUTO_DEEP_WIDTH       1e-12

#define PROBE_COLS 64
#define PROBE_ROWS 48
#define PROBE_DEADLINE_MS 200
#define PROBE_MIN_SAMPLES 512

/// Global state


enum phase
{
  PHASE_IDLE,
  PHASE_COMPUTE_ORBIT,
  PHASE_COMPUTE_IMAGE,
  PHASE_PROBE,
};


enum auto_iter_state
{
  AUTO_OFF,
  AUTO_HOLD,
  AUTO_GROW,
  AUTO_CONVERGED,
  AUTO_BUDGET,
  AUTO_RESCUE,
  AUTO_DEEP,
  AUTO_VIBE,
};


enum formula
{
  FORMULA_MANDELBROT,
  FORMULA_OMNIBROT,
};


static struct
{
  struct scene *scene;

  struct palette *palette;

  i32 w;
  i32 h;

  SDL_Window *window;
  SDL_Renderer *renderer;

  SDL_Texture *texture;
  SDL_Texture *texture_canvas;
  SDL_Texture *texture_helper;

  TTF_Font *font;

  bool running;

  enum phase phase;

  struct thread_pool *pool;

  struct image *image;

  f64 *orbit_re;
  f64 *orbit_im;

  i32 ss;

  f64 pan_x;
  f64 pan_y;
  f64 scale;

  bool redraw;

  enum formula formula;

  f64 omni_a;

  // Viewport center as plain doubles (used by the naive omnibrot evaluator)
  f64 center_re;
  f64 center_im;

  // Current generation
  _Atomic i32 generation;

  // Current work's generation
  i32 work_generation;

  _Atomic bool orbit_ready;

  _Atomic i32 tiles_done;
} s;


// Probe snapshot.  It is filled when a render starts and is only read by the
// probe job after that render has finished.
static _Atomic bool probe_done;

static u64 probe_changed;
static u64 probe_total;

static f64 probe_center_re;
static f64 probe_center_im;
static f64 probe_width;
static f64 probe_bailout2;
static f64 probe_omni_a;

static u64 probe_iter;
static u64 probe_iter_per_color;

static i32 probe_screen_w;
static i32 probe_screen_h;

static enum formula probe_formula;


// Adaptive iteration state.
static bool auto_iter = true;
static u64 auto_render_ms;
static f64 auto_sensitivity;
static enum auto_iter_state auto_state = AUTO_HOLD;
static f64 period_tol = 1e-40;

// At deep zoom a minibrot interior is normally pure black, which looks like
// an unrendered hole.  Paint it dark blue so the small Mandelbrot silhouette
// is visible.  b toggles it.
static bool color_inside = true;


void
terminate (void)
{
  atomic_fetch_add (&s.generation, 1);

  thread_pool_clear (s.pool);

  thread_pool_wait (s.pool);

  s.phase = PHASE_IDLE;

  atomic_store (&s.orbit_ready, false);

  atomic_store (&s.tiles_done, 0);
}


i32
n_tiles (void)
{
  const i32 tiles_amount_x = (s.w + TILE_SIZE - 1) / TILE_SIZE;
  const i32 tiles_amount_y = (s.h + TILE_SIZE - 1) / TILE_SIZE;
  return tiles_amount_x * tiles_amount_y;
}


// Change iter_max and rebuild the reference-orbit buffers.  New buffers are
// allocated before the old ones are released; the caller must be in a state
// where no worker is using them (this function calls terminate()).
static bool
apply_iter_max (u64 iter_max)
{
  if (iter_max < 1)
    iter_max = 1;

  if (iter_max > MAX_ITER_MAX)
    iter_max = MAX_ITER_MAX;

  if (iter_max == s.scene->iter_max)
    return false;

  f64 *orbit_re = calloc (iter_max + 1, sizeof (f64));
  f64 *orbit_im = calloc (iter_max + 1, sizeof (f64));

  if (!orbit_re || !orbit_im)
    {
      free (orbit_re);
      free (orbit_im);

      fprintf (stderr, "Could not allocate orbit buffers for %lu iterations\n",
               (unsigned long)iter_max);

      return false;
    }

  terminate ();

  free (s.orbit_re);
  free (s.orbit_im);

  s.orbit_re = orbit_re;
  s.orbit_im = orbit_im;

  s.scene->iter_max = iter_max;

  return true;
}


void
compute_orbit (void *argument)
{
  (void)argument;

  if (s.formula == FORMULA_OMNIBROT)
    {
      // The omnibrot evaluator works directly in double precision and does
      // not need a reference orbit.
      atomic_store (&s.orbit_ready, true);
      return;
    }

  mpfr_t two, z_re, z_im, z_re_new, t;

  mpfr_inits2 (FLOAT_PREC, two, z_re, z_im, z_re_new, t, (mpfr_ptr)NULL);

  mpfr_set_d (two, 2.0, MPFR_RNDN);
  mpfr_set_d (z_re, 0.0, MPFR_RNDN);
  mpfr_set_d (z_im, 0.0, MPFR_RNDN);

  for (u64 iter = 0; iter <= s.scene->iter_max; iter++)
    {
      if (s.work_generation != atomic_load (&s.generation))
        return;

      s.orbit_re[iter] = mpfr_get_d (z_re, MPFR_RNDN);
      s.orbit_im[iter] = mpfr_get_d (z_im, MPFR_RNDN);

      // t = z_im * z_im
      mpfr_sqr (t, z_im, MPFR_RNDN);

      // z_re_new = z_re * z_re - t
      mpfr_fms (z_re_new, z_re, z_re, t, MPFR_RNDN);

      // t = z_re * z_im
      mpfr_mul (t, z_re, z_im, MPFR_RNDN);

      // z_im = t * 2 + c_im
      mpfr_fma (z_im, t, two, s.scene->c_im, MPFR_RNDN);

      // z_re = z_re_new + c_re
      mpfr_add (z_re, z_re_new, s.scene->c_re, MPFR_RNDN);
    }

  mpfr_clears (two, z_re, z_im, z_re_new, t, (mpfr_ptr)NULL);

  atomic_store (&s.orbit_ready, true);
}


// e^(re + i im)
static void
complex_exp (f64 re, f64 im, f64 *restrict out_re, f64 *restrict out_im)
{
  if (fabs (re) + fabs (im) < 0.25)
    {
      // Taylor series; the remainder is below 1e-9 here.
      const f64 w2_re = re * re - im * im;
      const f64 w2_im = 2.0 * re * im;

      const f64 w3_re = w2_re * re - w2_im * im;
      const f64 w3_im = w2_re * im + w2_im * re;

      const f64 w4_re = w3_re * re - w3_im * im;
      const f64 w4_im = w3_re * im + w3_im * re;

      const f64 w5_re = w4_re * re - w4_im * im;
      const f64 w5_im = w4_re * im + w4_im * re;

      const f64 w6_re = w5_re * re - w5_im * im;
      const f64 w6_im = w5_re * im + w5_im * re;

      *out_re = 1.0 + re + w2_re / 2.0 + w3_re / 6.0 + w4_re / 24.0
                + w5_re / 120.0 + w6_re / 720.0;

      *out_im = im + w2_im / 2.0 + w3_im / 6.0 + w4_im / 24.0
                + w5_im / 120.0 + w6_im / 720.0;
    }
  else
    {
      const f64 e = exp (re);

      *out_re = e * cos (im);
      *out_im = e * sin (im);
    }
}


// Omnibrot: z <- z^2 * exp (a / z) + c
//
// The critical point of z^2 * exp (a / z) is z = a / 2, so it is used as z_0.
// This is a plain double precision evaluator: it does not use the
// single-reference perturbation machinery, so the zoom depth is limited by
// double precision (roughly 1e-13 relative).
f64
omnibrot (f64 delta_re, f64 delta_im, f64 *restrict nx, f64 *restrict ny)
{
  const f64 a = s.omni_a;

  const f64 c_re = s.center_re + delta_re;
  const f64 c_im = s.center_im + delta_im;

  // Critical point: d/dz (z^2 exp (a / z)) = exp (a / z) * (2 z - a)
  f64 z_re = a / 2.0;
  f64 z_im = 0.0;

  f64 deriv_re = 0.0;
  f64 deriv_im = 0.0;

  const f64 bailout2 = s.scene->bailout * s.scene->bailout;

  for (u64 iter = 0; iter < s.scene->iter_max; ++iter)
    {
      const f64 den = z_re * z_re + z_im * z_im;

      if (!(den > 0.0))
        // Hit the essential singularity; treat it as an escape.
        return iter;

      // w = a / z
      const f64 w_re = a * z_re / den;
      const f64 w_im = -a * z_im / den;

      f64 e_re;
      f64 e_im;

      complex_exp (w_re, w_im, &e_re, &e_im);

      // g = z^2 * e^w
      const f64 z2_re = z_re * z_re - z_im * z_im;
      const f64 z2_im = 2.0 * z_re * z_im;

      const f64 g_re = z2_re * e_re - z2_im * e_im;
      const f64 g_im = z2_re * e_im + z2_im * e_re;

      // g' = e^w * (2 z - a)
      const f64 gp_re = e_re * (2.0 * z_re - a) - e_im * (2.0 * z_im);
      const f64 gp_im = e_re * (2.0 * z_im) + e_im * (2.0 * z_re - a);

      // D_{n+1} = g' * D_n + 1
      const f64 new_deriv_re = gp_re * deriv_re - gp_im * deriv_im + 1.0;
      const f64 new_deriv_im = gp_re * deriv_im + gp_im * deriv_re;

      deriv_re = new_deriv_re;
      deriv_im = new_deriv_im;

      z_re = g_re + c_re;
      z_im = g_im + c_im;

      const f64 z_mag2 = z_re * z_re + z_im * z_im;

      if (!isfinite (z_mag2) || z_mag2 > bailout2)
        {
          if (!isfinite (z_mag2))
            {
              if (nx)
                *nx = 0.0;

              if (ny)
                *ny = 0.0;

              return iter;
            }

          const f64 aa = z_re * deriv_re + z_im * deriv_im;
          const f64 bb = z_im * deriv_re - z_re * deriv_im;

          const f64 len = hypot (aa, bb);

          if (nx)
            *nx = len > 0.0 ? aa / len : 0.0;

          if (ny)
            *ny = len > 0.0 ? bb / len : 0.0;

          return fmax (0.0, iter + 1 - log (log (z_mag2) * 0.5) / F64_LN2);
        }
    }

  // Never escaped

  if (nx)
    *nx = 0.0;

  if (ny)
    *ny = 0.0;

  return s.scene->iter_max;
}


// Adaptive iteration probe


struct probe_value
{
  bool escaped;
  f64 nu;
};


static bool
probe_known_interior (f64 x, f64 y)
{
  // Period-2 circle: (x + 1)^2 + y^2 <= 1/16
  const f64 dx = x + 1.0;
  if (dx * dx + y * y <= 1.0 / 16.0)
    return true;

  // Main cardioid
  const f64 qx = x - 0.25;
  const f64 q = qx * qx + y * y;
  if (q * (q + qx) <= y * y / 4.0)
    return true;

  return false;
}


static struct probe_value
probe_mandelbrot (f64 c_re, f64 c_im, u64 max_iter, f64 bailout2)
{
  f64 z_re = 0.0;
  f64 z_im = 0.0;

  for (u64 iter = 0; iter < max_iter; ++iter)
    {
      const f64 z_re2 = z_re * z_re;
      const f64 z_im2 = z_im * z_im;

      z_im = 2.0 * z_re * z_im + c_im;
      z_re = z_re2 - z_im2 + c_re;

      const f64 mag2 = z_re * z_re + z_im * z_im;

      if (mag2 > bailout2)
        {
          const f64 nu = (f64)(iter + 1)
                         - log (log (mag2) * 0.5) / F64_LN2;

          return (struct probe_value) { true, fmax (0.0, nu) };
        }
    }

  return (struct probe_value) { false, (f64)max_iter };
}


static struct probe_value
probe_omnibrot (f64 c_re, f64 c_im, u64 max_iter, f64 bailout2, f64 a)
{
  f64 z_re = a / 2.0;
  f64 z_im = 0.0;

  for (u64 iter = 0; iter < max_iter; ++iter)
    {
      const f64 den = z_re * z_re + z_im * z_im;

      if (!(den > 0.0))
        return (struct probe_value) { true, (f64)iter };

      const f64 w_re = a * z_re / den;
      const f64 w_im = -a * z_im / den;

      f64 e_re;
      f64 e_im;

      complex_exp (w_re, w_im, &e_re, &e_im);

      const f64 z2_re = z_re * z_re - z_im * z_im;
      const f64 z2_im = 2.0 * z_re * z_im;

      const f64 g_re = z2_re * e_re - z2_im * e_im;
      const f64 g_im = z2_re * e_im + z2_im * e_re;

      z_re = g_re + c_re;
      z_im = g_im + c_im;

      const f64 mag2 = z_re * z_re + z_im * z_im;

      if (!isfinite (mag2))
        return (struct probe_value) { true, (f64)iter };

      if (mag2 > bailout2)
        return (struct probe_value) { true, (f64)(iter + 1) };
    }

  return (struct probe_value) { false, (f64)max_iter };
}


static u32
probe_color (f64 nu)
{
  const f64 ipc = (f64)(probe_iter_per_color > 0
                        ? probe_iter_per_color
                        : 1);
  const f64 x = nu / ipc;

  const i64 i = (i64)x;
  const f64 f = x - (f64)i;

  const u32 c1 = palette_get (s.palette, i);
  const u32 c2 = palette_get (s.palette, i + 1);

  return color_lerp (c1, c2, f);
}


static bool
probe_colors_differ (f64 a, f64 b)
{
  // Be deliberately sensitive here.  On a deep boundary a change of only a
  // couple of RGB levels can still mean that the filament has not resolved
  // yet.  It is better to refine once more than to freeze a half-black image.
  const u32 ca = probe_color (a);
  const u32 cb = probe_color (b);

  const i64 dr = imax ((i64)color_r (ca) - (i64)color_r (cb),
                       (i64)color_r (cb) - (i64)color_r (ca));
  const i64 dg = imax ((i64)color_g (ca) - (i64)color_g (cb),
                       (i64)color_g (cb) - (i64)color_g (ca));
  const i64 db = imax ((i64)color_b (ca) - (i64)color_b (cb),
                       (i64)color_b (cb) - (i64)color_b (ca));

  return imax (dr, imax (dg, db)) >= 2;
}


static void
probe_job (void *argument)
{
  (void)argument;

  u64 changed = 0;
  u64 total = 0;

  const u64 max_iter = probe_iter > 0 ? probe_iter : 1;
  const u64 max_iter2 = max_iter > MAX_ITER_MAX / 4
                        ? MAX_ITER_MAX
                        : max_iter * 4;

  const f64 aspect = probe_screen_w > 0
                     ? (f64)probe_screen_h / (f64)probe_screen_w
                     : 1.0;

  const u32 probe_start = SDL_GetTicks ();

  bool deadline_hit = false;

  // Visit rows in a coprime stride so a deadline-limited probe still covers
  // the whole viewport instead of only its top edge.
  for (i32 row = 0; row < PROBE_ROWS && !deadline_hit; ++row)
    {
      const i32 gy = (row * 17) % PROBE_ROWS;

      if (s.work_generation != atomic_load (&s.generation))
        return;

      for (i32 gx = 0; gx < PROBE_COLS; ++gx)
        {
          if (s.work_generation != atomic_load (&s.generation))
            return;

          if (SDL_GetTicks () - probe_start >= PROBE_DEADLINE_MS)
            {
              deadline_hit = true;
              break;
            }

          const f64 dx = ((gx + 0.5) / PROBE_COLS - 0.5) * probe_width;
          const f64 dy = (0.5 - (gy + 0.5) / PROBE_ROWS) * probe_width * aspect;

          const f64 c_re = probe_center_re + dx;
          const f64 c_im = probe_center_im + dy;

          struct probe_value a;
          struct probe_value b;

          if (probe_formula == FORMULA_OMNIBROT)
            {
              a = probe_omnibrot (c_re, c_im, max_iter, probe_bailout2, probe_omni_a);
              b = probe_omnibrot (c_re, c_im, max_iter2, probe_bailout2, probe_omni_a);
            }
          else
            {
              if (probe_known_interior (c_re, c_im))
                {
                  total++;
                  continue;
                }

              a = probe_mandelbrot (c_re, c_im, max_iter, probe_bailout2);
              b = probe_mandelbrot (c_re, c_im, max_iter2, probe_bailout2);
            }

          bool changed_pixel;

          if (a.escaped != b.escaped)
            changed_pixel = true;
          else if (!a.escaped)
            // Both samples are still inside (or unresolved).  Increasing N
            // does not change a black pixel, so it is not a reason to grow.
            changed_pixel = false;
          else
            changed_pixel = probe_colors_differ (a.nu, b.nu);

          total++;

          if (changed_pixel)
            changed++;
        }
    }

  if (total < PROBE_MIN_SAMPLES)
    {
      // Too few samples for a trustworthy decision (usually a very high N
      // combined with the deadline).  Hold the current iteration count.
      probe_changed = 0;
      probe_total = 0;
    }
  else
    {
      probe_changed = changed;
      probe_total = total;
    }

  atomic_store (&probe_done, true);
}


static const char *
auto_state_name (enum auto_iter_state state)
{
  switch (state)
    {
    case AUTO_OFF:       return "off";
    case AUTO_HOLD:      return "hold";
    case AUTO_GROW:      return "grow";
    case AUTO_CONVERGED: return "converged";
    case AUTO_BUDGET:    return "budget";
    case AUTO_RESCUE:    return "rescue";
    case AUTO_DEEP:      return "deep-zoom";
    case AUTO_VIBE:      return "vibe";
    }

  return "hold";
}


static void
auto_iter_update (u64 render_ms, bool vibe)
{
  auto_render_ms = render_ms;

  if (!auto_iter)
    {
      auto_state = AUTO_OFF;
      return;
    }

  // Vibe mode is a navigation mode: texture scaling plus a fresh render every
  // 15 steps.  Do not change N underneath it; refine after vibe is switched
  // off, when the user is actually looking at a still image.
  if (vibe)
    {
      auto_state = AUTO_VIBE;
      return;
    }

  if (!(probe_width > AUTO_DEEP_WIDTH) || !isfinite (probe_width))
    {
      auto_state = AUTO_DEEP;
      return;
    }

  if (probe_total == 0)
    {
      auto_state = AUTO_HOLD;
      return;
    }

  const f64 sensitivity = (f64)probe_changed / (f64)probe_total;

  auto_sensitivity = sensitivity;

  if (sensitivity < AUTO_SENSITIVITY_LO)
    {
      auto_state = AUTO_CONVERGED;
      return;
    }

  const f64 frame_ms = fmax ((f64)render_ms, 1.0);

  f64 factor = (f64)AUTO_TARGET_MS / frame_ms;

  factor = fmin (fmax (factor, 1.25), 2.0);

  if (sensitivity < AUTO_SENSITIVITY_HI)
    factor = fmin (factor, 1.5);

  u64 next = (u64)((f64)s.scene->iter_max * factor);

  if (next <= s.scene->iter_max)
    next = s.scene->iter_max + 1;

  if (next > MAX_ITER_MAX)
    next = MAX_ITER_MAX;

  if (next == s.scene->iter_max)
    {
      auto_state = AUTO_BUDGET;
      return;
    }

  f64 estimate = frame_ms
                 * ((f64)next / (f64)s.scene->iter_max)
                 * 1.10;

  bool rescue = false;

  // 6 s is the comfortable target, not an absolute veto.  If the probe still
  // sees real changes, refusing to grow is exactly how the old controller
  // produced a half-black deep zoom.  Allow a slower step, but only up to a
  // hard ceiling.  The controller still never lowers N.
  if (estimate > AUTO_BUDGET_MS)
    {
      rescue = true;

      // Climb by at most 2x.  A 4x jump on a 16-second frame turns into a
      // minute-long render and rarely buys anything over the next 2x step.
      const f64 rescue_factor = 2.0;

      next = (u64)((f64)s.scene->iter_max * rescue_factor);

      if (next <= s.scene->iter_max)
        next = s.scene->iter_max + 1;

      if (next > MAX_ITER_MAX)
        next = MAX_ITER_MAX;

      estimate = frame_ms
                 * ((f64)next / (f64)s.scene->iter_max)
                 * 1.10;

      if (estimate > AUTO_HARD_MS)
        {
          // Quality is still visibly off, but a bigger frame would be
          // unreasonable.  Leave N untouched for a manual Home.
          auto_state = AUTO_BUDGET;
          return;
        }
    }

  if (apply_iter_max (next))
    {
      s.redraw = true;
      auto_state = rescue ? AUTO_RESCUE : AUTO_GROW;

      printf ("auto iter_max -> %lu (%s, sens %.2f%%, frame %lums, next ~%lums)\n",
              (unsigned long)s.scene->iter_max,
              rescue ? "rescue" : "grow",
              sensitivity * 100.0,
              (unsigned long)render_ms,
              (unsigned long)estimate);
    }
  else
    auto_state = AUTO_HOLD;
}


f64
mandelbrot (f64 delta_re, f64 delta_im, f64 *restrict nx, f64 *restrict ny)
{
  if (s.formula == FORMULA_OMNIBROT)
    return omnibrot (delta_re, delta_im, nx, ny);

  // Analytical membership tests: cardioid + period-2 circle.
  const f64 x = s.center_re + delta_re;
  const f64 y = s.center_im + delta_im;

  // Period-2 circle: (x+1)^2 + y^2 <= 1/16
  const f64 dx2 = x + 1.0;
  if (dx2 * dx2 + y * y <= 1.0 / 16.0)
    {
      if (nx) *nx = 0.0;
      if (ny) *ny = 0.0;
      return s.scene->iter_max;
    }

  // Main cardioid
  const f64 qx = x - 0.25;
  const f64 q = qx * qx + y * y;
  if (q * (q + qx) <= y * y / 4.0)
    {
      if (nx) *nx = 0.0;
      if (ny) *ny = 0.0;
      return s.scene->iter_max;
    }

  u64 iter = 0;
  u64 iter_orbit = 0;

  // Period (cycle) detection state.
  u64 period = 1;
  f64 saved_re = 0.0;
  f64 saved_im = 0.0;
  bool have_saved = false;

  f64 epsilon_re = 0.0;
  f64 epsilon_im = 0.0;

  f64 deriv_re = 0.0;
  f64 deriv_im = 0.0;

  const f64 bailout2 = s.scene->bailout * s.scene->bailout;

  while (iter < s.scene->iter_max)
    {
      f64 r_re;
      f64 r_im;

      f64 z_re;
      f64 z_im;

      r_re = s.orbit_re[iter_orbit];
      r_im = s.orbit_im[iter_orbit];

      z_re = r_re + epsilon_re;
      z_im = r_im + epsilon_im;

      // D_{n+1} = 2 * z_n * D_n + 1
      const f64 new_deriv_re = 2.0 * (z_re * deriv_re - z_im * deriv_im) + 1.0;
      const f64 new_deriv_im = 2.0 * (z_re * deriv_im + z_im * deriv_re);

      deriv_re = new_deriv_re;
      deriv_im = new_deriv_im;

      // 2 * z_{n} * ε_{n}
      const f64 lin_re = 2.0 * (r_re * epsilon_re - r_im * epsilon_im);
      const f64 lin_im = 2.0 * (r_re * epsilon_im + r_im * epsilon_re);

      // ε_{n}^2
      const f64 quad_re = epsilon_re * epsilon_re - epsilon_im * epsilon_im;
      const f64 quad_im = epsilon_re * epsilon_im * 2.0;

      // ε_{n+1} = 2 * z_{n} * ε_{n} + ε_{n}^2 + δ_c
      epsilon_re = lin_re + quad_re + delta_re;
      epsilon_im = lin_im + quad_im + delta_im;

      r_re = s.orbit_re[iter_orbit + 1];
      r_im = s.orbit_im[iter_orbit + 1];

      z_re = r_re + epsilon_re;
      z_im = r_im + epsilon_im;

      const f64 z_mag2 = z_re * z_re + z_im * z_im;

      // Period (cycle) detection
      if (have_saved)
        {
          const f64 dr = z_re - saved_re;
          const f64 di = z_im - saved_im;
          if (period_tol > 1e-61 && dr * dr + di * di < period_tol * (1.0 + z_mag2))
            {
              if (nx) *nx = 0.0;
              if (ny) *ny = 0.0;
              return s.scene->iter_max;
            }
        }

      if (iter + 1 == period)
        {
          saved_re = z_re;
          saved_im = z_im;
          have_saved = true;
          period *= 2;
          if (period > s.scene->iter_max) period = s.scene->iter_max;
        }

      // TODO: Bailout
      if (z_mag2 > bailout2)
        {
          const f64 a = z_re * deriv_re + z_im * deriv_im;
          const f64 b = z_im * deriv_re - z_re * deriv_im;

          const f64 len = hypot (a, b);

          if (nx)
            *nx = a / len;

          if (ny)
            *ny = b / len;

          return fmax (0.0, iter + 1 - log (log (z_mag2) * 0.5) / F64_LN2);
        }

      iter++;
      iter_orbit++;

      if (epsilon_re * epsilon_re + epsilon_im * epsilon_im > z_mag2 / 2)
        {
          epsilon_re = z_re;
          epsilon_im = z_im;
          iter_orbit = 0;
        }
    }

  // Never escaped

  if (nx)
    *nx = 0.0;

  if (ny)
    *ny = 0.0;

  return s.scene->iter_max;
}


u32
mandelbrot_color (f64 nu)
{
  const f64 x = nu / s.scene->iter_per_color;

  const i64 i = x;
  const f64 f = x - i;

  const u32 c1 = palette_get (s.palette, i);
  const u32 c2 = palette_get (s.palette, i + 1);

  return color_lerp (c1, c2, f);
}


f64
mandelbrot_shade (f64 nx, f64 ny)
{
  const f64 lx = s.scene->lx;
  const f64 ly = s.scene->ly;
  const f64 lz = s.scene->lz;

  const f64 shade = (nx * lx + ny * ly + lz) / sqrt (nx * nx + ny * ny + 1.0);

  return fmax (0.0, shade * 0.7 + 0.3);
}


/// Work


struct work
{
  i32 x;
  i32 y;
};


void
worker (void *argument)
{
  struct work *work = argument;

  const f64 sample = 1.0 / s.ss;
  const i32 samples = s.ss * s.ss;

  const f64 scale = s.scene->width / s.w;

  for (i32 y = work->y; y < work->y + TILE_SIZE && y < s.h; ++y)
    for (i32 x = work->x; x < work->x + TILE_SIZE && x < s.w; ++x)
      {
        if (s.work_generation != atomic_load (&s.generation))
          return;

        u64 r = 0, g = 0, b = 0;

        for (i32 sy = 0; sy < s.ss; ++sy)
          for (i32 sx = 0; sx < s.ss; ++sx)
            {
              if (s.work_generation != atomic_load (&s.generation))
                return;

              const f64 dx = (x - s.w / 2.0 + (sx + 0.5) * sample) * scale;
              const f64 dy = (s.h / 2.0 - y - (sy + 0.5) * sample) * scale;

              f64 nx, ny;
              f64 nu = mandelbrot (dx, dy, &nx, &ny);

              if (nu >= s.scene->iter_max)
                {
                  if (color_inside && s.scene->width <= 1e-3)
                    {
                      // Dark blue interior: the minibrot silhouette is
                      // visible instead of turning into a black hole.
                      r += 24;
                      g += 20;
                      b += 48;
                    }

                  continue;
                }

              const u32 color = mandelbrot_color (nu);

              const f64 shade = mandelbrot_shade (nx, ny);

              r += color_r (color) * shade;
              g += color_g (color) * shade;
              b += color_b (color) * shade;
            }

        image_set2 (s.image, x, y, color_rgb (r / samples, g / samples, b / samples));
      }

  atomic_fetch_add (&s.tiles_done, 1);
}


// UI


void
write_text (i32 x, i32 y, u8 r, u8 g, u8 b, const char *format, ...)
{
  char buffer[1024];

  va_list va;
  va_start (va, format);

  vsnprintf (buffer, sizeof buffer, format, va);

  va_end (va);

  SDL_Color color = { .r = r, .g = g, .b = b, .a = 0xFF };

  SDL_Surface *surface = TTF_RenderText_Blended (s.font, buffer, color);

  if (!surface)
    return;

  SDL_Texture *texture = SDL_CreateTextureFromSurface (s.renderer, surface);

  if (!texture)
    {
      SDL_FreeSurface (surface);
      return;
    }

  const int padding = 4;

  SDL_Rect bg = {
    x - padding,
    y,
    surface->w + padding * 2,
    surface->h
  };

  SDL_Rect dst = {
    x,
    y,
    surface->w,
    surface->h
  };

  SDL_SetRenderDrawColor (s.renderer, 0x00, 0x00, 0x00, 0xFF);
  SDL_RenderFillRect (s.renderer, &bg);

  SDL_RenderCopy (s.renderer, texture, NULL, &dst);

  SDL_FreeSurface (surface);
  SDL_DestroyTexture (texture);
}


/// Main


void
help (FILE *f, const char *self)
{
  fprintf (f, "USAGE: %s [OPTION]... <SCENE>\n", self);
  fprintf (f, "\n");
  fprintf (f, "OPTION:\n");
  fprintf (f, "  -w                <int>        Set the window's initial width in pixels\n");
  fprintf (f, "  -h                <int>        Set the window's initial height in pixels\n");
  fprintf (f, "  --re              <float>      Set the real part of C\n");
  fprintf (f, "  --im              <float>      Set the imaginary part of C\n");
  fprintf (f, "  --width           <float>      Set the viewport width\n");
  fprintf (f, "  --bailout         <float>      Set the bailout radius\n");
  fprintf (f, "  --light-angle     <float>      Set the light's angle\n");
  fprintf (f, "  --iter-max        <int>        Set the maximum number of iterations\n");
  fprintf (f, "  --iter-per-color  <int>        Set the iterations per color\n");
  fprintf (f, "  --formula         <NAME>       Formula: mandelbrot or omnibrot\n");
  fprintf (f, "  --a               <float>      Set omnibrot's 'a' parameter (default 0.005)\n");
  fprintf (f, "  --palette         <PALETTE>    Set the palette's path\n");
  fprintf (f, "\n");
  fprintf (f, "SCENE:\n");
  fprintf (f, "  A '*.sce' file\n");
  fprintf (f, "\n");
  fprintf (f, "PALETTE:\n");
  fprintf (f, "  A '*.pal' file\n");

}


struct opt_u64
{
  u64 value;

  bool has_value;
};


struct opt_f64
{
  f64 value;

  bool has_value;
};


int
main (i32 argc, char **argv)
{
  i32 initial_w = 800;
  i32 initial_h = 600;

  const char *scene_path = NULL;

  const char *c_re = NULL;
  const char *c_im = NULL;

  struct opt_f64 width = { 0 };

  struct opt_f64 bailout = { 0 };

  struct opt_f64 light_angle = { 0 };

  struct opt_u64 iter_max = { 0 };

  struct opt_u64 iter_per_color = { 0 };

  enum formula formula = FORMULA_MANDELBROT;

  struct opt_f64 omni_a = { 0 };

  const char *palette_path = NULL;

  i32 i = 1;

  while (i < argc)
    {
      const char *arg = argv[i];

      if (strcmp (arg, "--help") == 0)
        {
          help (stdout, argv[0]);
          return 0;
        }

      // INITIAL_W

      if (strcmp (arg, "-w") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '-w' requires an integer\n");
              return 1;
            }

          initial_w = strtol (argv[++i], NULL, 10);

          ++i;

          continue;
        }

      // INITIAL_H

      if (strcmp (arg, "-h") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '-h' requires an integer\n");
              fprintf (stderr, "Note: maybe you meant '--help'?\n");
              return 1;
            }

          initial_h = strtol (argv[++i], NULL, 10);

          ++i;

          continue;
        }

      // C_RE

      if (strcmp (arg, "--re") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--re' requires a float\n");
              return 1;
            }

          c_re = argv[++i];

          ++i;

          continue;
        }

      // C_IM

      if (strcmp (arg, "--im") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--im' requires a float\n");
              return 1;
            }

          c_im = argv[++i];

          ++i;

          continue;
        }

      // WIDTH

      if (strcmp (arg, "--width") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--width' requires a float\n");
              return 1;
            }

          width.value = strtod (argv[++i], NULL);
          width.has_value = true;

          ++i;

          continue;
        }

      // BAILOUT

      if (strcmp (arg, "--bailout") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--bailout' requires a float\n");
              return 1;
            }

          bailout.value = strtod (argv[++i], NULL);
          bailout.has_value = true;

          ++i;

          continue;
        }

      // LIGHT_ANGLE

      if (strcmp (arg, "--light-angle") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--light-angle' requires a float\n");
              return 1;
            }

          light_angle.value = strtod (argv[++i], NULL);
          light_angle.has_value = true;

          ++i;

          continue;
        }

      // ITER_MAX

      if (strcmp (arg, "--iter-max") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--iter-max' requires an integer\n");
              return 1;
            }

          iter_max.value = strtoull (argv[++i], NULL, 10);
          iter_max.has_value = true;

          ++i;

          continue;
        }

      // ITER_PER_COLOR

      if (strcmp (arg, "--iter-per-color") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--iter-per-color' requires an integer\n");
              return 1;
            }

          iter_per_color.value = strtoull (argv[++i], NULL, 10);
          iter_per_color.has_value = true;

          ++i;

          continue;
        }

      // FORMULA

      if (strcmp (arg, "--formula") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--formula' requires a name\n");
              return 1;
            }

          const char *name = argv[++i];

          if (strcmp (name, "mandelbrot") == 0 || strcmp (name, "m") == 0)
            formula = FORMULA_MANDELBROT;
          else if (strcmp (name, "omnibrot") == 0 || strcmp (name, "o") == 0)
            formula = FORMULA_OMNIBROT;
          else
            {
              fprintf (stderr, "Error: Unknown formula '%s' (use 'mandelbrot' or 'omnibrot')\n",
                       name);
              return 1;
            }

          ++i;

          continue;
        }

      // OMNIBROT A

      if (strcmp (arg, "--a") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--a' requires a number\n");
              return 1;
            }

          omni_a.value = strtod (argv[++i], NULL);
          omni_a.has_value = true;

          ++i;

          continue;
        }

      // PALETTE_PATH

      if (strcmp (arg, "--palette") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--palette' requires a path\n");
              return 1;
            }

          palette_path = argv[++i];

          ++i;

          continue;
        }

      if (scene_path)
        {
          fprintf (stderr, "Error: Scene path already specified (%s)\n", scene_path);
          return 1;
        }

      scene_path = argv[i++];
    }

  if (scene_path == NULL)
    {
      fprintf (stderr, "Error: Scene path not specified\n");
      help (stderr, argv[0]);
      return 1;
    }

  // Load scene

  s.scene = scene_load (scene_path);

  if (s.scene == NULL)
    {
      fprintf (stderr, "Error: Failed to load scene (%s)\n", scene_path);
      return 1;
    }

  printf ("Loaded '%s'\n", scene_path);

  s.formula = formula;

  s.omni_a = omni_a.has_value ? omni_a.value : OMNIBROT_A_DEFAULT;

  // Override

  if (c_re != NULL)
    mpfr_set_str (s.scene->c_re, c_re, 10, MPFR_RNDN);

  if (c_im != NULL)
    mpfr_set_str (s.scene->c_im, c_im, 10, MPFR_RNDN);

  if (width.has_value)
    s.scene->width = width.value;

  if (bailout.has_value)
    s.scene->bailout = bailout.value;

  if (light_angle.has_value)
    {
      const f64 as = sinl (light_angle.value * F64_DEG2RAD);
      const f64 ac = cosl (light_angle.value * F64_DEG2RAD);

      const f64 lx = as;
      const f64 ly = ac;
      const f64 lz = 1.0;

      const f64 ll = sqrtl (lx * lx + ly * ly + lz * lz);

      s.scene->lx = lx / ll;
      s.scene->ly = ly / ll;
      s.scene->lz = lz / ll;
    }

  if (iter_max.has_value)
    s.scene->iter_max = iter_max.value;

  if (iter_per_color.has_value)
    s.scene->iter_per_color = iter_per_color.value;

  // Load palette

  if (palette_path == NULL)
    palette_path = s.scene->palette_path;

  s.palette = palette_load (palette_path);

  if (s.palette == NULL)
    {
      fprintf (stderr, "Error: Failed to load palette (%s)\n", palette_path);
      return 1;
    }

  if (palette_path != s.scene->palette_path)
    snprintf (s.scene->palette_path, sizeof s.scene->palette_path, "%s", palette_path);

  printf ("Loaded '%s'\n", palette_path);

  SDL_Init (SDL_INIT_VIDEO);
  TTF_Init ();

  s.w = initial_w;
  s.h = initial_h;

  SDL_SetHint (SDL_HINT_RENDER_SCALE_QUALITY, "linear");

  s.window = SDL_CreateWindow (
    "Mandexplorer",
    SDL_WINDOWPOS_CENTERED,
    SDL_WINDOWPOS_CENTERED,
    s.w,
    s.h,
    SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
  );

  s.renderer = SDL_CreateRenderer (
    s.window,
    -1,
    SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
  );

  s.texture = SDL_CreateTexture (
    s.renderer,
    SDL_PIXELFORMAT_ARGB8888,
    SDL_TEXTUREACCESS_STREAMING,
    s.w,
    s.h
  );

  SDL_SetTextureBlendMode (s.texture, SDL_BLENDMODE_BLEND);

  // 2. Persistent Canvas Texture
  s.texture_canvas = SDL_CreateTexture (
    s.renderer,
    SDL_PIXELFORMAT_ARGB8888,
    SDL_TEXTUREACCESS_TARGET,
    s.w,
    s.h
  );

  SDL_SetTextureBlendMode (s.texture_canvas, SDL_BLENDMODE_BLEND);

  s.texture_helper = SDL_CreateTexture (
    s.renderer,
    SDL_PIXELFORMAT_ARGB8888,
    SDL_TEXTUREACCESS_TARGET,
    s.w,
    s.h
  );

  s.font = TTF_OpenFont (FONT_PATH, 18);

  s.running = true;

  s.phase = PHASE_IDLE;

  s.pool = thread_pool_create (SDL_GetCPUCount (), 1920 * 1080);

  s.image = image_create (s.w, s.h);

  s.orbit_re = calloc (s.scene->iter_max + 1, sizeof (f64));
  s.orbit_im = calloc (s.scene->iter_max + 1, sizeof (f64));

  s.ss = 1;

  s.pan_x = 0.0L;
  s.pan_y = 0.0L;
  s.scale = 1.0L;

  s.redraw = true;

  atomic_init (&s.generation, 0);

  s.work_generation = 0;

  atomic_init (&s.orbit_ready, false);

  atomic_init (&probe_done, false);

  auto_state = AUTO_HOLD;
  auto_render_ms = 0;
  auto_sensitivity = 0.0;

  struct work *works = calloc (n_tiles (), sizeof (struct work));

  // Measuring COMPUTE_IMAGE phase's time
  u32 compute_image_start = SDL_GetTicks ();
  u32 compute_image_total = 0;

  bool ui_info = false;
  bool ui_help = false;
  bool vibe = false;

  while (s.running)
    {
      SDL_Event event;

      while (SDL_PollEvent (&event))
        switch (event.type)
          {
          case SDL_QUIT:
            s.running = false;
            break;

          case SDL_KEYDOWN:
            switch (event.key.keysym.sym)
              {
              case SDLK_ESCAPE:
                terminate ();
                break;

              case SDLK_LALT:
              case SDLK_RALT:
                ui_info = !ui_info;
                break;

              case SDLK_F1:
              case SDLK_h:
                ui_help = !ui_help;
                break;

              case SDLK_v:
                vibe = !vibe;
                printf ("Vibe zoom: %s\n", vibe ? "on" : "off");
                s.redraw = true;
                break;

              // ADAPTIVE ITERATIONS
              case SDLK_i:
                auto_iter = !auto_iter;
                auto_state = auto_iter ? AUTO_HOLD : AUTO_OFF;
                printf ("Adaptive iterations: %s\n", auto_iter ? "on" : "off");
                s.redraw = true;
                break;

              // INTERIOR COLOR
              case SDLK_b:
                color_inside = !color_inside;
                printf ("Interior color: %s\n", color_inside ? "on" : "off");
                s.redraw = true;
                break;

              // FORMULA
              case SDLK_m:
                terminate ();

                s.formula = FORMULA_MANDELBROT;

                s.redraw = true;
                break;

              case SDLK_o:
                terminate ();

                s.formula = FORMULA_OMNIBROT;

                s.redraw = true;
                break;

              // OMNIBROT "a" PARAMETER
              case SDLK_LEFTBRACKET:
                terminate ();

                s.omni_a = fmax (s.omni_a / 2.0, 1e-6);

                s.redraw = true;
                break;

              case SDLK_RIGHTBRACKET:
                terminate ();

                s.omni_a = fmin (s.omni_a * 2.0, 1.0);

                s.redraw = true;
                break;

              // RESET
              case SDLK_x:
                terminate ();

                s.scene->width = 8.0;

                mpfr_set_d (s.scene->c_re, -0.75, MPFR_RNDN);
                mpfr_set_d (s.scene->c_im, +0.00, MPFR_RNDN);

                s.pan_x = 0.0;
                s.pan_y = 0.0;
                s.scale = 1.0;

                s.redraw = true;
                break;

              // CENTER
              case SDLK_c:
                terminate ();

                {
                  i32 mx, my;
                  SDL_GetMouseState (&mx, &my);

                  mx -= s.w / 2.0;
                  my -= s.h / 2.0;

                  s.pan_x += mx / s.scale;
                  s.pan_y += my / s.scale;
                }

                s.redraw = true;
                break;

              // BAILOUT+
              case SDLK_INSERT:
                terminate ();

                s.scene->bailout *= 2.0;

                s.redraw = true;
                break;

              // BAILOUT-
              case SDLK_DELETE:
                terminate ();

                s.scene->bailout = fmax (s.scene->bailout / 2.0, 2.0);

                s.redraw = true;
                break;

              // ITER_MAX+
              case SDLK_HOME:
                auto_iter = false;
                auto_state = AUTO_OFF;

                if (s.scene->iter_max < MAX_ITER_MAX)
                  apply_iter_max (s.scene->iter_max * 2);

                s.redraw = true;
                break;

              // ITER_MAX-
              case SDLK_END:
                auto_iter = false;
                auto_state = AUTO_OFF;

                if (s.scene->iter_max > 1)
                  apply_iter_max (s.scene->iter_max / 2);

                s.redraw = true;
                break;

              // ITER_PER_COLOR+
              case SDLK_PAGEUP:
                terminate ();

                s.scene->iter_per_color *= 2;

                s.redraw = true;
                break;

              // ITER_PER_COLOR-
              case SDLK_PAGEDOWN:
                terminate ();

                s.scene->iter_per_color = imax (s.scene->iter_per_color / 2, 1);

                s.redraw = true;
                break;

              // PERIOD TOL +
              case SDLK_PERIOD:
                terminate ();

                period_tol *= 10.0;

                printf ("Period tol: %g\n", period_tol);
                s.redraw = true;
                break;

              // PERIOD TOL -
              case SDLK_COMMA:
                terminate ();

                period_tol = fmax (period_tol / 10.0, 1e-61);

                printf ("Period tol: %g\n", period_tol);
                s.redraw = true;
                break;

              case SDLK_s:
                {
                  time_t raw_time;
                  struct tm *time_info;
                  char save_path[128];

                  time (&raw_time);
                  time_info = localtime (&raw_time);

                  strftime (save_path, sizeof save_path, "%Y%m%d_%H%M%S.sce", time_info);

                  scene_save (save_path, s.scene);

                  printf ("Saved '%s'\n", save_path);
                }
                break;

              case SDLK_1:
                terminate ();

                s.ss = 1;

                s.redraw = true;
                break;

              case SDLK_2:
                terminate ();

                s.ss = 2;

                s.redraw = true;
                break;

              case SDLK_4:
                terminate ();

                s.ss = 4;

                s.redraw = true;
                break;

              case SDLK_8:
                terminate ();

                s.ss = 8;

                s.redraw = true;
                break;

              default:
                break;
              }
            break;

          case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_RIGHT)
              {
                terminate ();

                s.redraw = 1;
              }
            break;

          case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT)
              {
                terminate ();

                s.redraw = 1;
              }
            break;

          case SDL_MOUSEMOTION:
            if (event.motion.state & SDL_BUTTON_LMASK)
              {
                s.pan_x -= event.motion.xrel / s.scale;
                s.pan_y -= event.motion.yrel / s.scale;
              }
            break;

          case SDL_WINDOWEVENT:
            if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
              {
                terminate ();

                s.w = event.window.data1;
                s.h = event.window.data2;

                SDL_DestroyTexture (s.texture);
                SDL_DestroyTexture (s.texture_canvas);
                SDL_DestroyTexture (s.texture_helper);

                s.texture = SDL_CreateTexture (
                  s.renderer,
                  SDL_PIXELFORMAT_ARGB8888,
                  SDL_TEXTUREACCESS_STREAMING,
                  s.w,
                  s.h
                );

                SDL_SetTextureBlendMode (s.texture, SDL_BLENDMODE_BLEND);

                s.texture_canvas = SDL_CreateTexture (
                  s.renderer,
                  SDL_PIXELFORMAT_ARGB8888,
                  SDL_TEXTUREACCESS_TARGET,
                  s.w,
                  s.h
                );

                SDL_SetTextureBlendMode (s.texture_canvas, SDL_BLENDMODE_BLEND);

                s.texture_helper = SDL_CreateTexture (
                  s.renderer,
                  SDL_PIXELFORMAT_ARGB8888,
                  SDL_TEXTUREACCESS_TARGET,
                  s.w,
                  s.h
                );

                SDL_SetRenderDrawColor (s.renderer, 0x00, 0x00, 0x00, 0x00);

                SDL_SetRenderTarget (s.renderer, s.texture_canvas);
                SDL_RenderClear (s.renderer);

                SDL_SetRenderTarget (s.renderer, s.texture_helper);
                SDL_RenderClear (s.renderer);

                SDL_SetRenderTarget (s.renderer, NULL);

                image_destroy (s.image);

                s.image = image_create (s.w, s.h);

                free (works);

                works = calloc (n_tiles (), sizeof (struct work));

                s.redraw = true;
              }
            break;

          case SDL_MOUSEWHEEL:
            terminate ();

            {
              i32 mx, my;
              SDL_GetMouseState (&mx, &my);

              const f64 sx = mx - s.w / 2.0;
              const f64 sy = my - s.h / 2.0;

              bool ctrl = SDL_GetModState () & KMOD_CTRL;

              const f64 anchor_x = ctrl ? 0.0 : sx;
              const f64 anchor_y = ctrl ? 0.0 : sy;

              bool shift = SDL_GetModState () & KMOD_SHIFT;

              const f64 zoom_factor = shift ? 1.1 : 1.4;

              const f64 zoom = (event.wheel.y > 0) ? zoom_factor : (1.0 / zoom_factor);

              const f64 world_x = (anchor_x / s.scale) + s.pan_x;
              const f64 world_y = (anchor_y / s.scale) + s.pan_y;

              s.scale *= zoom;

              s.pan_x = world_x - (anchor_x / s.scale);
              s.pan_y = world_y - (anchor_y / s.scale);
            }

            s.redraw = true;
            break;

          default:
            break;
          }

      if (vibe)
        {
          static u64 vibe_counter = 0;
          vibe_counter++;

          i32 mx, my;
          SDL_GetMouseState (&mx, &my);
          const f64 sx = mx - s.w / 2.0;
          const f64 sy = my - s.h / 2.0;
          const f64 zoom = 1.02;
          const f64 world_x = (sx / s.scale) + s.pan_x;
          const f64 world_y = (sy / s.scale) + s.pan_y;
          s.scale *= zoom;
          s.pan_x = world_x - (sx / s.scale);
          s.pan_y = world_y - (sy / s.scale);

          // The texture is scaled every frame; every 15th step starts a fresh
          // render at the current zoom.
          if (vibe_counter % 15 == 0)
            {
              terminate ();
              s.redraw = true;
            }
        }

      switch (s.phase)
        {
        case PHASE_IDLE:
          if (s.redraw)
            {
              s.redraw = false;

              SDL_SetRenderTarget (s.renderer, s.texture_helper);
              SDL_RenderClear (s.renderer);
              SDL_RenderCopy (s.renderer, s.texture_canvas, NULL, NULL);

              SDL_SetRenderTarget (s.renderer, s.texture_canvas);
              SDL_RenderClear (s.renderer);

              SDL_FRect vr;

              vr.x = (s.w * (1.0 - s.scale)) * 0.5 - s.pan_x * s.scale;
              vr.y = (s.h * (1.0 - s.scale)) * 0.5 - s.pan_y * s.scale;
              vr.w = s.w * s.scale;
              vr.h = s.h * s.scale;

              SDL_RenderCopyF (s.renderer, s.texture_helper, NULL, &vr);

              mpfr_add_d (s.scene->c_re, s.scene->c_re, s.pan_x * s.scene->width / s.w, MPFR_RNDN);
              mpfr_sub_d (s.scene->c_im, s.scene->c_im, s.pan_y * s.scene->width / s.w, MPFR_RNDN);

              s.scene->width /= s.scale;

              s.pan_x = 0.0L;
              s.pan_y = 0.0L;
              s.scale = 1.0L;

              image_fill (s.image, color_argb (0, 0, 0, 0));

              s.work_generation = atomic_load (&s.generation);

              s.center_re = mpfr_get_d (s.scene->c_re, MPFR_RNDN);
              s.center_im = mpfr_get_d (s.scene->c_im, MPFR_RNDN);

              // Snapshot the exact view that is about to be rendered.  The
              // probe must not observe a later pan/zoom or a different N.
              probe_center_re = s.center_re;
              probe_center_im = s.center_im;
              probe_width = s.scene->width;
              probe_bailout2 = s.scene->bailout * s.scene->bailout;
              probe_omni_a = s.omni_a;
              probe_iter = s.scene->iter_max;
              probe_iter_per_color = s.scene->iter_per_color;
              probe_screen_w = s.w;
              probe_screen_h = s.h;
              probe_formula = s.formula;

              s.phase = PHASE_COMPUTE_ORBIT;

              thread_pool_enqueue (s.pool, compute_orbit, NULL);
            }
          break;

        case PHASE_COMPUTE_ORBIT:
          if (atomic_load (&s.orbit_ready))
            {
              s.work_generation = atomic_load (&s.generation);

              compute_image_start = SDL_GetTicks ();

              s.phase = PHASE_COMPUTE_IMAGE;

              const i32 cols = (s.w + TILE_SIZE - 1) / TILE_SIZE;
              const i32 rows = (s.h + TILE_SIZE - 1) / TILE_SIZE;
              const i32 tiles_total = cols * rows;

              i32 mx, my;
              SDL_GetMouseState (&mx, &my);

              i32 tile_x = mx / TILE_SIZE;
              i32 tile_y = my / TILE_SIZE;

              if (tile_x < 0) tile_x = 0;
              if (tile_x >= cols) tile_x = cols - 1;
              if (tile_y < 0) tile_y = 0;
              if (tile_y >= rows) tile_y = rows - 1;

              i32 step_size = 1;
              i32 direction = 0;
              i32 tiles_enqueued = 0;

              usize n = 0;

              const i32 dx[] = { 1,  0, -1,  0 };
              const i32 dy[] = { 0,  1,  0, -1 };

              while (tiles_enqueued < tiles_total)
                {
                  for (i32 i = 0; i < 2; i++)
                    {
                      for (i32 j = 0; j < step_size; j++)
                        {
                          if (tile_x >= 0 && tile_x < cols && tile_y >= 0 && tile_y < rows)
                            {
                              struct work *work = &works[n];

                              work->x = tile_x * TILE_SIZE;
                              work->y = tile_y * TILE_SIZE;

                              thread_pool_enqueue(s.pool, worker, work);

                              n++;
                              tiles_enqueued++;
                            }

                          tile_x += dx[direction];
                          tile_y += dy[direction];
                        }

                      direction = (direction + 1) % 4;
                    }

                  step_size++;
                }

            }
          break;

        case PHASE_COMPUTE_IMAGE:
          if (s.tiles_done == n_tiles ())
            {
              compute_image_total = SDL_GetTicks () - compute_image_start;

              const bool can_probe = auto_iter
                                     && !vibe
                                     && probe_width > AUTO_DEEP_WIDTH
                                     && isfinite (probe_width);

              if (!can_probe)
                {
                  auto_iter_update (compute_image_total, vibe);
                  s.phase = PHASE_IDLE;
                  break;
                }

              probe_changed = 0;
              probe_total = 0;

              atomic_store (&probe_done, false);

              if (thread_pool_enqueue (s.pool, probe_job, NULL) == 0)
                s.phase = PHASE_PROBE;
              else
                {
                  auto_iter_update (compute_image_total, vibe);
                  s.phase = PHASE_IDLE;
                }
            }
          break;

        case PHASE_PROBE:
          if (atomic_load (&probe_done))
            {
              auto_iter_update (compute_image_total, vibe);

              s.phase = PHASE_IDLE;
            }
          break;

        default:
          break;
        }

      // Upload texture
      {
        i32 pitch;
        void *pixels;

        SDL_LockTexture (s.texture, NULL, &pixels, &pitch);

        for (i32 y = 0; y < s.image->h; ++y)
          {
            u32 *row = (u32 *)((u8 *)pixels + (usize)y * (usize)pitch);

            for (i32 x = 0; x < s.image->w; ++x)
              row[x] = image_get2 (s.image, x, y);
          }

        SDL_UnlockTexture (s.texture);
      }

      SDL_SetRenderTarget (s.renderer, s.texture_canvas);
      SDL_RenderCopy (s.renderer, s.texture, NULL, NULL);

      SDL_SetRenderTarget (s.renderer, NULL);
      SDL_RenderClear (s.renderer);

      SDL_FRect vr;

      vr.x = (s.w * (1.0 - s.scale)) * 0.5 - s.pan_x * s.scale;
      vr.y = (s.h * (1.0 - s.scale)) * 0.5 - s.pan_y * s.scale;
      vr.w = s.w * s.scale;
      vr.h = s.h * s.scale;

      SDL_RenderCopyF (s.renderer, s.texture_canvas, NULL, &vr);

      if (ui_info)
        {
          i32 y;

#define write_tl(y, ...) write_text (10, 10 + 32 * (y), 196, 196, 196, __VA_ARGS__)

          y = 0;

          write_tl (y++, "C_re: %+.1e", mpfr_get_d (s.scene->c_re, MPFR_RNDN));
          write_tl (y++, "C_im: %+.1e", mpfr_get_d (s.scene->c_im, MPFR_RNDN));
          write_tl (y++, "Width: %.1e", s.scene->width);

          y++;

          if (s.formula == FORMULA_OMNIBROT)
            {
              write_tl (y++, "Formula: Omnibrot");
              write_tl (y++, "Omnibrot a: %g", s.omni_a);
            }
          else
            {
              write_tl (y++, "Formula: Mandelbrot");
            }

          y++;

          write_tl (y++, "Bailout: %.1f", s.scene->bailout);
          write_tl (y++, "Period: %s (%.0e, ./,)", period_tol > 1e-61 ? "on" : "off", period_tol);

          y++;

          write_tl (y++, "Iterations: %ld", s.scene->iter_max);
          write_tl (y++, "Iterations per Color: %ld", s.scene->iter_per_color);

          write_tl (y++, "Adaptive iterations: %s", auto_state_name (auto_state));
          write_tl (y++, "  sensitivity: %.2f%%, frame: %lums",
                    auto_sensitivity * 100.0, (unsigned long)auto_render_ms);

          if (auto_state == AUTO_RESCUE)
            write_tl (y++, "  quality rescue: over 6s, up to 60s");

          write_tl (y++, "Interior color: %s (b)", color_inside ? "on" : "off");

          y++;

          write_tl (y++, "SSAA: %d", s.ss);

#undef write_tl

#define write_bl(y, ...) write_text (10, (s.h - 32) - 32 * (y), 196, 196, 196, __VA_ARGS__)

          y = 0;

          switch (s.phase)
            {
            case PHASE_IDLE:
              write_bl (y++, "IDLE (last image: %dms) [%s]",
                        compute_image_total, auto_state_name (auto_state));
              break;
            case PHASE_COMPUTE_ORBIT:
              write_bl (y++, "COMPUTING ORBIT");
              break;
            case PHASE_PROBE:
              write_bl (y++, "ANALYZING ITERATIONS (N: %lu)",
                        (unsigned long)s.scene->iter_max);
              break;
            case PHASE_COMPUTE_IMAGE:
              {
                const i64 total_time = SDL_GetTicks () - compute_image_start;

                const i64 tiles_done = atomic_load (&s.tiles_done);

                const f64 avarage_time = (f64)total_time / imax (tiles_done, 1);

                const f64 total_time_estimate = avarage_time * n_tiles ();

                const f64 remaining_time = fmax (total_time_estimate - total_time, 0.0);

                const f64 remaining_seconds_all = remaining_time / 1000.0;

                const f64 remaining_minutes = floor (remaining_seconds_all / 60.0);

                const f64 remaining_seconds = fmod (remaining_seconds_all, 60.0);

                write_bl (y++, "COMPUTING IMAGE %d/%d %.0fm %.0fs", atomic_load (&s.tiles_done),
                          n_tiles (), remaining_minutes, remaining_seconds);
              }
              break;
            }

#undef write_bl
        }

      if (ui_help)
        {
          i32 y;

#define write_ct(y, ...) write_text (350, 10 + 32 * (y), 196, 196, 196, __VA_ARGS__)

          y = 0;

          write_ct (y++, "Show/hide help UI ; F1 or h");
          write_ct (y++, "Show/hide info UI ; Alt");
          write_ct (y++, "Bailout +/- ; Insert/Delete");
          write_ct (y++, "Iterations +/- ; Home/End (manual)");
          write_ct (y++, "Adaptive iterations ; i");
          write_ct (y++, "Interior color ; b");
          write_ct (y++, "Vibe zoom ; v");
          write_ct (y++, "Iterations per Color +/- ; PageUp/PageDown");
          write_ct (y++, "SSAA ; 1/2/4/8");
          write_ct (y++, "Reset viewport ; x");
          write_ct (y++, "Center viewport on mouse ; c");
          write_ct (y++, "Mandelbrot formula ; m");
          write_ct (y++, "Omnibrot formula ; o");
          write_ct (y++, "Omnibrot a -/+ ; [ / ]");
          write_ct (y++, "Save scene ; s");
          write_ct (y++, "Terminate ; Escape");
          write_ct (y++, "Pan ; Left Click + Drag");
          write_ct (y++, "Zoom toward mouse ; Scroll Wheel");
          write_ct (y++, "Zoom toward center ; Control + Scroll Wheel");
          write_ct (y++, "Zoom slower ; Shift + Scroll Wheel");

#undef write_ct
        }

      SDL_RenderPresent (s.renderer);
    }

  terminate ();

  image_destroy (s.image);

  SDL_DestroyTexture (s.texture);
  SDL_DestroyRenderer (s.renderer);
  SDL_DestroyWindow (s.window);

  palette_destroy (s.palette);
  scene_destroy (s.scene);

  TTF_Quit ();
  SDL_Quit ();

  printf ("Goodbye.\n");

  return 0;
}

