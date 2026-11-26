#define _POSIX_C_SOURCE 200809L

#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <mpfr.h>
#include <stdatomic.h>
#include <time.h>

#include "bla.h"
#include "color.h"
#include "common.h"
#include "config.h"
#include "image.h"
#include "palette.h"
#include "scene.h"
#include "thread_pool.h"
#include "gpu.h"


/// General configuration


#define FONT_PATH "res/DMSans-Regular.ttf"


// #define FONT_PATH "res/Iosevka-Regular.ttf"

#define MAX_ITER_MAX 16777216

// Omnibrot: z <- z^2 * exp (a / z) + c
#define OMNIBROT_A_DEFAULT 0.005

#define ORBIT_REUSE_EPS       1e-3
#define ORBIT_REUSE_MIN_WIDTH 1e-10

#define VIDEO_FIT_MAX_ATTEMPTS    3
#define VIDEO_FIT_SHRINK_FACTOR   4
#define VIDEO_FIT_MIN_SHRINK      64
#define VIDEO_FIT_PROBE_DEADLINE  500
#define VIDEO_FIT_PROBE_MIN_SAMP  128
#define VIDEO_FIT_SENS_LO         0.005

#define KEYFRAME_MAX_MAG       1.5

// Number of video frames served by one full keyframe render.  Overridable
// for A/B measurement through MANDEX_DIFF_REUSE.
static i32 keyframe_max_reuse = 5;

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


static struct
{
  struct scene *scene;

  struct palette *palette;

  i32 w;
  i32 h;

  i32 sectors;

  i32 mouse_cx;
  i32 mouse_cy;

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
  i32 ss_prev;

  // Snapshot of the sampling/mode used by the render that is currently in
  // flight.  The display may toggle SSAA or vibe while workers are running;
  // those workers must keep the values they started with.
  i32 render_ss;
  bool render_vibe;

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


static struct point *render_pixels = NULL;
static usize render_pixel_count = 0;
static usize render_pixels_cap = 0;

// True while a GPU frame is being produced by a worker.  The display must not
// upload s.image while the GPU job is still writing it, and in vibe mode it
// must not wait for that job either: it keeps showing the last complete frame.
static _Atomic bool gpu_job_active;
static _Atomic bool gpu_vibe_done;
static _Atomic bool gpu_vibe_inflight;

// During vibe the canvas keeps the frame baked at the last scene rebase until
// the next full render is ready.  Uploading the stale image over the baked
// pixels is what makes the view snap back on every rebase.
static bool canvas_frozen = false;

// File scope so the preview path knows whether this is a vibe stream.
static bool vibe = false;

static usize *render_counts = NULL;
static usize render_counts_cap = 0;


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

static u64 probe_deadline_ms = PROBE_DEADLINE_MS;
static u64 probe_min_samples = PROBE_MIN_SAMPLES;

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

struct orbit_cache
{
  bool valid;
  u64 iter_max;
  mpfr_t c_re;
  mpfr_t c_im;
  f64 c_re_d;
  f64 c_im_d;
};

static struct orbit_cache orbit_cache;
static struct bla_table bla_table;

// Neural preview: render at half resolution, upscale with FSRCNN for display.
// s.image stays full-size (UI, video, texture); preview_render_image is the
// half-size render target.  Preview is never used for final output.
static bool preview_mode = false;
// Divisor for the preview render resolution: 1 = full (crisp), 2 = half
// (FSRCNN x2 upscale), 3-4 = bilinear fallback only.
static i32 preview_div = 2;
static struct image *preview_render_image = NULL;
static i32 render_w = 0;
static i32 render_h = 0;
static char preview_root[2048] = { 0 };

// The upscaler runs on its own thread so the display loop never blocks.
static pthread_t preview_thread;
static bool preview_thread_started = false;
static pthread_mutex_t preview_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t preview_cond = PTHREAD_COND_INITIALIZER;
static bool preview_stop = false;
static bool preview_job_pending = false;
static bool preview_result_ready = false;
static struct image *preview_job_src = NULL;  // half-res snapshot for worker
static struct image *preview_job_dst = NULL;  // full-res result from worker
static struct image *preview_fast_image = NULL;  // live bilinear fallback
static u32 preview_fast_stamp = 0;
// Once a neural frame exists we keep presenting it; flipping back to the
// partial bilinear image on every new render is what makes vibe jerky.
static bool preview_final_valid = false;

// log2 safety margin between |dz| and a BLA step's validity radius.
static f64 bla_margin = 16.0;
static bool bla_enabled = true;
static f64 delta_c_re = 0.0;
static f64 delta_c_im = 0.0;

static struct image *keyframe_image = NULL;
static struct view_state keyframe_view;
static bool has_keyframe = false;
static bool keyframe_pending = false;
static usize keyframe_render_index = 0;


// Recording mode state.

struct view_state
{
  // Exact view centre.  A plain double cannot hold it once the view width is
  // below ~1 ulp of the centre (~2e-16 near -1.75): recorded frames then
  // quantise to the same/p neighbouring doubles and a video zooms into the
  // wrong point, with the keyframe warp oscillating between those ulps.
  mpfr_t c_re_hi;
  mpfr_t c_im_hi;

  // Double snapshots, used only by the cheap iteration probes.
  f64 c_re;
  f64 c_im;
  f64 width;

  u64 iter_max;
  u64 recorded_iter_max;
  u64 iter_per_color;

  i32 ss;

  enum formula formula;

  f64 omni_a;
  f64 bailout;

  f64 lx;
  f64 ly;
  f64 lz;

  bool color_inside;
};


static void
view_state_init (struct view_state *v)
{
  mpfr_inits2 (FLOAT_PREC, v->c_re_hi, v->c_im_hi, (mpfr_ptr) NULL);
}


static void
view_state_set_center (struct view_state *v, mpfr_srcptr re, mpfr_srcptr im)
{
  mpfr_set (v->c_re_hi, re, MPFR_RNDN);
  mpfr_set (v->c_im_hi, im, MPFR_RNDN);
  v->c_re = mpfr_get_d (re, MPFR_RNDN);
  v->c_im = mpfr_get_d (im, MPFR_RNDN);
}


static void
view_state_copy (struct view_state *dst, const struct view_state *src)
{
  dst->c_re = src->c_re;
  dst->c_im = src->c_im;
  dst->width = src->width;

  dst->iter_max = src->iter_max;
  dst->recorded_iter_max = src->recorded_iter_max;
  dst->iter_per_color = src->iter_per_color;

  dst->ss = src->ss;
  dst->formula = src->formula;
  dst->omni_a = src->omni_a;
  dst->bailout = src->bailout;
  dst->lx = src->lx;
  dst->ly = src->ly;
  dst->lz = src->lz;
  dst->color_inside = src->color_inside;

  mpfr_set (dst->c_re_hi, src->c_re_hi, MPFR_RNDN);
  mpfr_set (dst->c_im_hi, src->c_im_hi, MPFR_RNDN);
}


static void
view_state_clear (struct view_state *v)
{
  mpfr_clears (v->c_re_hi, v->c_im_hi, (mpfr_ptr) NULL);
}


// Pixel-space offset between two recorded views, computed from the exact
// centres.  With double centres this difference is pure ulp noise once the
// zoom is deep, which makes keyframe warps jump to the wrong place.
static void
view_offset_px (const struct view_state *v, const struct view_state *kv,
                f64 *dx, f64 *dy)
{
  const f64 scale = (f64) s.w / kv->width;

  mpfr_t t;
  mpfr_init2 (t, FLOAT_PREC);

  mpfr_sub (t, v->c_re_hi, kv->c_re_hi, MPFR_RNDN);
  mpfr_mul_d (t, t, scale, MPFR_RNDN);
  *dx = mpfr_get_d (t, MPFR_RNDN);

  mpfr_sub (t, v->c_im_hi, kv->c_im_hi, MPFR_RNDN);
  mpfr_mul_d (t, t, scale, MPFR_RNDN);
  *dy = mpfr_get_d (t, MPFR_RNDN);

  mpfr_clear (t);
}


static struct view_state *recorded_views = NULL;
static usize recorded_count = 0;
static usize recorded_cap = 0;

static bool recording = false;
static bool rendering_video = false;
static bool video_frame_pending = false;
static usize render_index = 0;

// Preview is for navigation only.  While recording or rendering the video the
// full-resolution fractal must be written straight into s.image: no half-res
// render, no neural upscale, no FSRCNN output in the final MP4.
static bool
preview_active (void)
{
  return preview_mode && !recording && !rendering_video;
}

static void
recorded_views_free (void)
{
  for (usize i = 0; i < recorded_count; ++i)
    view_state_clear (&recorded_views[i]);

  free (recorded_views);
  recorded_views = NULL;
  recorded_count = 0;
  recorded_cap = 0;
}



static FILE *ffmpeg_pipe = NULL;
static char video_path[256] = { 0 };

// Optional frame dumps for A/B comparison of keyframe reuse settings.
static const char *dump_prefix = NULL;
static usize dump_limit = 0;
static usize dump_count = 0;
static usize auto_frames = 0;


void
terminate (void)
{
  atomic_fetch_add (&s.generation, 1);

  thread_pool_clear (s.pool);

  thread_pool_wait (s.pool);

  atomic_store (&gpu_job_active, false);
  atomic_store (&gpu_vibe_done, false);
  atomic_store (&gpu_vibe_inflight, false);

  s.phase = PHASE_IDLE;

  atomic_store (&s.orbit_ready, false);

  atomic_store (&s.tiles_done, 0);

  canvas_frozen = false;
}


static void
probe_job (void *argument);

static bool
can_reuse_keyframe (const struct view_state *kv, const struct view_state *v);

static void
warp_keyframe (const struct view_state *kv, const struct view_state *v);

static u32
bilinear_sample (const struct image *img, f64 x, f64 y);

static void
preview_update_dims (void);

static void
preview_restart (void);

static void
start_render_phase (void)
{
  rendering_video = true;

  // Video frames must be rendered at full resolution into s.image; switch the
  // preview pipeline back to full size before any frame is scheduled.
  if (preview_mode)
    {
      terminate ();
      preview_update_dims ();
      preview_restart ();
    }

  render_index = 0;
  video_frame_pending = false;
  has_keyframe = false;
  keyframe_pending = false;
  keyframe_render_index = 0;

  time_t raw_time;
  struct tm *time_info;

  time (&raw_time);
  time_info = localtime (&raw_time);

  strftime (video_path, sizeof video_path, "%Y%m%d_%H%M%S.mp4", time_info);

  char cmd[2048];

  snprintf (cmd, sizeof cmd,
            "ffmpeg -y -hide_banner -loglevel error "
            "-f rawvideo -pix_fmt rgb24 -s %dx%d -r 30 -i - "
            "-c:v libx264 -pix_fmt yuv444p -preset fast -g 60 "
            "-movflags +frag_keyframe+empty_moov "
            "\"%s\"",
            s.w, s.h, video_path);

  ffmpeg_pipe = popen (cmd, "w");

  if (!ffmpeg_pipe)
    {
      fprintf (stderr, "Failed to start ffmpeg\n");
      rendering_video = false;
    }
  else
    {
      printf ("Rendering %zu recorded frames to '%s'\n",
              recorded_count, video_path);
    }

  if (getenv ("MANDEX_DUMP_VIEWS"))
    {
      FILE *f = fopen ("/tmp/mandexplorer_views.txt", "w");

      if (f)
        {
          for (usize i = 0; i < recorded_count; ++i)
            {
              const struct view_state *v = &recorded_views[i];

              fprintf (f, "%zu %.17e ", i, v->width);
              mpfr_out_str (f, 10, 0, v->c_re_hi, MPFR_RNDN);
              fputc (' ', f);
              mpfr_out_str (f, 10, 0, v->c_im_hi, MPFR_RNDN);
              fputc ('\n', f);
            }

          fclose (f);
        }
    }
}


static u64
fit_video_n (struct view_state *v)
{
  // Do not shrink iter_max for video frames: the adaptive probe can
  // underestimate the required depth and produce half-black deep zooms.
  // Keep the (possibly adapted) iter_max captured for this frame.
  return v->iter_max;

  const u64 recorded = v->iter_max;

  u64 min_n = recorded / VIDEO_FIT_MIN_SHRINK;
  const u64 color_floor = v->iter_per_color * 4;
  if (min_n < color_floor)
    min_n = color_floor;
  if (min_n < 1)
    min_n = 1;

  u64 n = recorded;

  s.work_generation = atomic_load (&s.generation);

  const u64 saved_deadline = probe_deadline_ms;
  const u64 saved_min_samples = probe_min_samples;
  probe_deadline_ms = VIDEO_FIT_PROBE_DEADLINE;
  probe_min_samples = VIDEO_FIT_PROBE_MIN_SAMP;

  for (int attempt = 0; attempt < VIDEO_FIT_MAX_ATTEMPTS; ++attempt)
    {
      probe_iter = n;
      probe_iter_per_color = v->iter_per_color;
      probe_center_re = v->c_re;
      probe_center_im = v->c_im;
      probe_width = v->width;
      probe_bailout2 = v->bailout * v->bailout;
      probe_omni_a = v->omni_a;
      probe_screen_w = s.w;
      probe_screen_h = s.h;
      probe_formula = v->formula;

      probe_changed = 0;
      probe_total = 0;
      atomic_store (&probe_done, false);

      probe_job (NULL);

      if (probe_total < probe_min_samples)
        break;

      const f64 sens = (f64)probe_changed / (f64)probe_total;

      if (sens >= VIDEO_FIT_SENS_LO)
        break;

      u64 next = n / VIDEO_FIT_SHRINK_FACTOR;
      if (next < min_n)
        next = min_n;
      if (next >= n)
        break;

      n = next;
    }

  probe_deadline_ms = saved_deadline;
  probe_min_samples = saved_min_samples;

  return n;
}


static bool
can_reuse_keyframe (const struct view_state *kv, const struct view_state *v)
{
  if (kv->formula != v->formula) return false;
  if (kv->recorded_iter_max != v->recorded_iter_max) return false;
  if (kv->iter_per_color != v->iter_per_color) return false;
  if (kv->ss != v->ss) return false;
  if (kv->color_inside != v->color_inside) return false;
  if (kv->omni_a != v->omni_a) return false;
  if (kv->bailout != v->bailout) return false;
  if (kv->lx != v->lx) return false;
  if (kv->ly != v->ly) return false;
  if (kv->lz != v->lz) return false;

  const f64 S = v->width / kv->width;

  f64 Dx, Dy;
  view_offset_px (v, kv, &Dx, &Dy);

  const f64 cx = (s.w / 2.0) * (1.0 - S) + Dx;
  const f64 cy = (s.h / 2.0) * (1.0 - S) - Dy;

  if (cx < 0 || (f64)s.w * S + cx > (f64)s.w - 1) return false;
  if (cy < 0 || (f64)s.h * S + cy > (f64)s.h - 1) return false;

  if (S < 1.0 / KEYFRAME_MAX_MAG) return false;

  if (render_index - keyframe_render_index >= (usize) keyframe_max_reuse) return false;

  return true;
}


static u32
bilinear_sample (const struct image *img, f64 x, f64 y)
{
  const i32 W = img->w;
  const i32 H = img->h;

  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x > W - 1) x = W - 1;
  if (y > H - 1) y = H - 1;

  const i32 x0 = (i32)x;
  const i32 y0 = (i32)y;
  const i32 x1 = imin (x0 + 1, W - 1);
  const i32 y1 = imin (y0 + 1, H - 1);

  const f64 fx = x - x0;
  const f64 fy = y - y0;

  const u32 c00 = image_get2 (img, x0, y0);
  const u32 c10 = image_get2 (img, x1, y0);
  const u32 c01 = image_get2 (img, x0, y1);
  const u32 c11 = image_get2 (img, x1, y1);

  const f64 r00 = color_r (c00), g00 = color_g (c00), b00 = color_b (c00);
  const f64 r10 = color_r (c10), g10 = color_g (c10), b10 = color_b (c10);
  const f64 r01 = color_r (c01), g01 = color_g (c01), b01 = color_b (c01);
  const f64 r11 = color_r (c11), g11 = color_g (c11), b11 = color_b (c11);

  const f64 w00 = (1 - fx) * (1 - fy);
  const f64 w10 = fx * (1 - fy);
  const f64 w01 = (1 - fx) * fy;
  const f64 w11 = fx * fy;

  return color_rgb (
    (u8)(r00*w00 + r10*w10 + r01*w01 + r11*w11 + 0.5),
    (u8)(g00*w00 + g10*w10 + g01*w01 + g11*w11 + 0.5),
    (u8)(b00*w00 + b10*w10 + b01*w01 + b11*w11 + 0.5)
  );
}


static void
warp_keyframe (const struct view_state *kv, const struct view_state *v)
{
  const f64 S = v->width / kv->width;

  f64 Dx, Dy;
  view_offset_px (v, kv, &Dx, &Dy);

  const f64 cx = (s.w / 2.0) * (1.0 - S) + Dx;
  const f64 cy = (s.h / 2.0) * (1.0 - S) - Dy;

  for (i32 y = 0; y < s.h; ++y)
    {
      const f64 ky = y * S + cy;

      for (i32 x = 0; x < s.w; ++x)
        {
          const f64 kx = x * S + cx;

          const u32 c = bilinear_sample (keyframe_image, kx, ky);

          image_set2 (s.image, x, y, c);
        }
    }
}


static void
dump_ppm (void)
{
  if (!dump_prefix || dump_count >= dump_limit)
    return;

  char path[512];

  snprintf (path, sizeof path, "%s%04zu.ppm", dump_prefix, dump_count++);

  FILE *f = fopen (path, "wb");

  if (!f)
    return;

  fprintf (f, "P6\n%d %d\n255\n", s.w, s.h);

  for (i32 y = 0; y < s.h; ++y)
    for (i32 x = 0; x < s.w; ++x)
      {
        const u32 c = image_get2 (s.image, x, y);
        const unsigned char rgb[3] = { color_r (c), color_g (c), color_b (c) };
        fwrite (rgb, 1, 3, f);
      }

  fclose (f);
}


// Resolve the directory containing the binary so the preview helper works
// regardless of the current working directory.
static void
preview_init_root (void)
{
  ssize_t n = readlink ("/proc/self/exe", preview_root, sizeof preview_root - 1);

  if (n > 0)
    {
      preview_root[n] = '\0';
      char *slash = strrchr (preview_root, '/');
      if (slash) slash[1] = '\0';
    }
  else
    strcpy (preview_root, "./");
}


static void
preview_update_dims (void)
{
  if (preview_active ())
    {
      render_w = s.w / preview_div;
      render_h = s.h / preview_div;

      if (render_w < 1) render_w = 1;
      if (render_h < 1) render_h = 1;

      if (preview_render_image)
        image_destroy (preview_render_image);

      preview_render_image = image_create (render_w, render_h);
    }
  else
    {
      render_w = s.w;
      render_h = s.h;
    }
}


static bool
preview_write_ppm (const char *path, const struct image *img)
{
  FILE *f = fopen (path, "wb");
  if (!f)
    return false;

  fprintf (f, "P6\n%d %d\n255\n", img->w, img->h);

  for (i32 y = 0; y < img->h; ++y)
    for (i32 x = 0; x < img->w; ++x)
      {
        const u32 c = image_get2 (img, x, y);
        const unsigned char rgb[3] = { color_r (c), color_g (c), color_b (c) };
        fwrite (rgb, 1, 3, f);
      }

  fclose (f);
  return true;
}


static bool
preview_read_ppm (const char *path, struct image *img)
{
  FILE *f = fopen (path, "rb");
  if (!f)
    return false;

  char magic[3] = { 0 };
  i32 w = 0;
  i32 h = 0;
  i32 maxv = 0;
  bool ok = fscanf (f, "%2s %d %d %d", magic, &w, &h, &maxv) == 4
            && strcmp (magic, "P6") == 0
            && w == img->w && h == img->h && maxv == 255;

  if (ok)
    fgetc (f); // single whitespace follows maxval

  for (i32 y = 0; ok && y < h; ++y)
    for (i32 x = 0; ok && x < w; ++x)
      {
        unsigned char rgb[3];
        if (fread (rgb, 1, 3, f) != 3)
          {
            ok = false;
            break;
          }
        image_set2 (img, x, y, color_rgb (rgb[0], rgb[1], rgb[2]));
      }

  fclose (f);
  return ok;
}


// The slow FSRCNN pass.  Runs on the preview thread, never on the display
// thread, so navigation stays responsive while the upscale is in flight.
static void
preview_upscale_sync (const struct image *src, struct image *dst)
{
  const char *in = "/tmp/mandexplorer_preview_in.ppm";
  const char *out = "/tmp/mandexplorer_preview_out.ppm";

  if (!preview_write_ppm (in, src))
    return;

  char cmd[8192];

  snprintf (cmd, sizeof cmd,
            "\"%s.venv-preview/bin/python\" \"%stools/nn_preview.py\" "
            "%s %s > /tmp/mandexplorer_preview.log 2>&1",
            preview_root, preview_root, in, out);

  if (system (cmd) != 0)
    return;

  preview_read_ppm (out, dst);
}


static void *
preview_thread_main (void *arg)
{
  (void) arg;

  for (;;)
    {
      pthread_mutex_lock (&preview_mutex);

      while (!preview_job_pending && !preview_stop)
        pthread_cond_wait (&preview_cond, &preview_mutex);

      if (preview_stop)
        {
          pthread_mutex_unlock (&preview_mutex);
          break;
        }

      preview_job_pending = false;
      pthread_mutex_unlock (&preview_mutex);

      preview_upscale_sync (preview_job_src, preview_job_dst);

      pthread_mutex_lock (&preview_mutex);
      preview_result_ready = true;
      pthread_mutex_unlock (&preview_mutex);
    }

  return NULL;
}


static void preview_stop_thread (void);

// (Re)allocate buffers and (re)start the worker.  Called at startup and after
// a window resize; stopping the thread first avoids racing on the buffers.
static void
preview_restart (void)
{
  if (!preview_mode)
    return;

  preview_stop_thread ();

  if (preview_job_src) image_destroy (preview_job_src);
  if (preview_job_dst) image_destroy (preview_job_dst);
  if (preview_fast_image) image_destroy (preview_fast_image);

  preview_job_src = image_create (render_w, render_h);
  preview_job_dst = image_create (s.w, s.h);
  preview_fast_image = image_create (s.w, s.h);

  preview_stop = false;
  preview_job_pending = false;
  preview_result_ready = false;
  preview_final_valid = false;

  if (pthread_create (&preview_thread, NULL, preview_thread_main, NULL) == 0)
    preview_thread_started = true;
}


static void
preview_stop_thread (void)
{
  if (!preview_thread_started)
    return;

  pthread_mutex_lock (&preview_mutex);
  preview_stop = true;
  pthread_cond_signal (&preview_cond);
  pthread_mutex_unlock (&preview_mutex);

  pthread_join (preview_thread, NULL);
  preview_thread_started = false;
}


static void
preview_submit_job (void)
{
  // FSRCNN x2 only matches a half-resolution render.  At full resolution the
  // image is already crisp; at other divisors only the live fallback is used.
  if (!preview_active () || preview_div != 2
      || !preview_thread_started || !preview_render_image)
    return;

  pthread_mutex_lock (&preview_mutex);

  // At most one job in flight; the newest completed frame wins.
  if (!preview_job_pending)
    {
      image_copy (preview_render_image, preview_job_src);
      preview_job_pending = true;
      pthread_cond_signal (&preview_cond);
    }

  pthread_mutex_unlock (&preview_mutex);
}


// Returns the image the display should present this tick: the finished
// neural upscale when available, otherwise a cheap live bilinear preview of
// the half-res buffer so the zoom never freezes.
static struct image *
preview_display_image (void)
{
  if (!preview_active () || preview_div == 1)
    return s.image;

  // The neural result belongs to an older scene and arrives too late.  In
  // vibe we show the live half-res stream so the frame swapped in when the
  // canvas unfreezes always matches the current scene.
  if (vibe)
    {
      if (!preview_fast_image || !preview_render_image)
        return s.image;

      pthread_mutex_lock (&preview_mutex);
      if (preview_result_ready)
        preview_result_ready = false;
      pthread_mutex_unlock (&preview_mutex);

      for (i32 y = 0; y < s.h; ++y)
        for (i32 x = 0; x < s.w; ++x)
          {
            const f64 sx = ((f64) x + 0.5) * (f64) render_w / (f64) s.w - 0.5;
            const f64 sy = ((f64) y + 0.5) * (f64) render_h / (f64) s.h - 0.5;
            image_set2 (preview_fast_image, x, y,
                        bilinear_sample (preview_render_image, sx, sy));
          }

      return preview_fast_image;
    }

  bool have_result = false;

  pthread_mutex_lock (&preview_mutex);
  if (preview_result_ready)
    {
      image_copy (preview_job_dst, s.image);
      preview_result_ready = false;
      have_result = true;
      preview_final_valid = true;
    }
  pthread_mutex_unlock (&preview_mutex);

  // Keep the last complete neural frame on screen until the next one arrives.
  // The bilinear image is only a stopgap before the very first result.
  if (have_result || preview_final_valid
      || !preview_fast_image || !preview_render_image)
    return s.image;

  // Throttle: doing a full-screen bilinear pass on every display tick costs
  // more CPU than the render itself at deep zoom.  4 Hz is enough for a live
  // progress indicator.
  const u32 now = SDL_GetTicks ();
  if (now - preview_fast_stamp >= 250)
    {
      for (i32 y = 0; y < s.h; ++y)
        for (i32 x = 0; x < s.w; ++x)
          {
            const f64 sx = ((f64) x + 0.5) * (f64) render_w / (f64) s.w - 0.5;
            const f64 sy = ((f64) y + 0.5) * (f64) render_h / (f64) s.h - 0.5;
            image_set2 (preview_fast_image, x, y,
                        bilinear_sample (preview_render_image, sx, sy));
          }
      preview_fast_stamp = now;
    }

  return preview_fast_image;
}


i32
n_tiles (void)
{
  if (s.sectors < 1)
    return 4;

  return s.sectors;
}


// GPU rendering is a long, uninterrupted OpenCL call.  Running it on the
// display thread freezes the zoom for the whole frame time (clFinish plus a
// blocking read), so it runs in the pool and only publishes the result when
// the frame is complete.  There is at most one GPU job in flight; terminate()
// waits for it before the scene or the orbit buffers are mutated.
struct gpu_job_args
{
  struct gpu_view view;
  i32 generation;
  struct image *out;
  bool vibe;
};

static struct gpu_job_args gpu_job_args;
static struct gpu_job_args gpu_vibe_job_args;

static struct image *gpu_vibe_image = NULL;

static void
gpu_job (void *argument)
{
  struct gpu_job_args *args = argument;

  const bool current = args->generation == atomic_load (&s.generation);

  if (current)
    gpu_render_mandelbrot (&args->view, args->out ? args->out : s.image);

  if (args->vibe)
    atomic_store (&gpu_vibe_inflight, false);
  else
    atomic_store (&gpu_job_active, false);

  if (current && args->generation == atomic_load (&s.generation))
    {
      if (args->vibe)
        {
          atomic_store (&gpu_vibe_done, true);
        }
      else
        atomic_store (&s.tiles_done, n_tiles ());
    }
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

  orbit_cache.valid = false;
  bla_table.valid = false;

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

  orbit_cache.valid = false;

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
        {
          mpfr_clears (two, z_re, z_im, z_re_new, t, (mpfr_ptr)NULL);
          return;
        }

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

  mpfr_set (orbit_cache.c_re, s.scene->c_re, MPFR_RNDN);
  mpfr_set (orbit_cache.c_im, s.scene->c_im, MPFR_RNDN);
  orbit_cache.c_re_d = mpfr_get_d (s.scene->c_re, MPFR_RNDN);
  orbit_cache.c_im_d = mpfr_get_d (s.scene->c_im, MPFR_RNDN);
  orbit_cache.iter_max = s.scene->iter_max;
  orbit_cache.valid = true;

  // At shallow depths the skipped fraction is too small to pay for the
  // extended-exponent polynomial, so only engage BLA for real deep zooms.
  if (bla_enabled && s.scene->width <= 1e-12)
    {
      const f64 shift = hypot (delta_c_re, delta_c_im);
      bla_build (&bla_table, s.orbit_re, s.orbit_im, s.scene->iter_max,
                 s.scene->width * 0.75 + shift, bla_margin);
    }
  else
    bla_clear (&bla_table);

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
      if (s.work_generation != atomic_load (&s.generation))
        return 0.0;

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

          if (SDL_GetTicks () - probe_start >= probe_deadline_ms)
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

  if (total < probe_min_samples)
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


// When rendering a recorded video, carry any adaptive iter_max adjustment
// back into the frame's view state.  Otherwise the next-frame setup would
// immediately restore the captured N and the controller could never help.
static void
persist_video_iter (void)
{
  if (!rendering_video || render_index >= recorded_count) return;
  recorded_views[render_index].iter_max = s.scene->iter_max;
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

  // Adjust perturbation delta when reusing a cached reference orbit.
  delta_re += delta_c_re;
  delta_im += delta_c_im;

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
      if (s.work_generation != atomic_load (&s.generation))
        return 0.0;

      u64 span = 0;

      if (bla_table.valid
          && iter_orbit % BLA_BASE_STEP == 0
          && iter_orbit + BLA_BASE_STEP <= s.scene->iter_max
          && iter + BLA_BASE_STEP <= s.scene->iter_max
          && bla_try_skip (&bla_table, iter_orbit,
                           &epsilon_re, &epsilon_im, &deriv_re, &deriv_im,
                           delta_re, delta_im, &span))
        {
          iter += span;
          iter_orbit += span;

          const f64 zr = s.orbit_re[iter_orbit] + epsilon_re;
          const f64 zi = s.orbit_im[iter_orbit] + epsilon_im;
          const f64 z2 = zr * zr + zi * zi;

          if (z2 > bailout2)
            {
              const f64 a = zr * deriv_re + zi * deriv_im;
              const f64 bb = zi * deriv_re - zr * deriv_im;
              const f64 len = hypot (a, bb);

              if (nx) *nx = len > 0.0 ? a / len : 0.0;
              if (ny) *ny = len > 0.0 ? bb / len : 0.0;

              return fmax (0.0, iter + 1 - log (log (z2) * 0.5) / F64_LN2);
            }

          if (epsilon_re * epsilon_re + epsilon_im * epsilon_im > z2 / 2.0)
            {
              epsilon_re = zr;
              epsilon_im = zi;
              iter_orbit = 0;
            }

          continue;
        }

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


struct point
{
  i32 x;
  i32 y;
  f64 r2;
};


struct work
{
  usize offset;
  usize stride;
};


void
worker (void *argument)
{
  struct work *work = argument;

  const f64 sample = 1.0 / s.render_ss;
  const i32 samples = s.render_ss * s.render_ss;

  const f64 scale = s.scene->width / (f64) render_w;

  struct image *target =
    (preview_active () && preview_div > 1) ? preview_render_image : s.image;

  const usize offset = work->offset;
  const usize stride = work->stride;
  const usize total = render_pixel_count;

  for (usize i = offset; i < total; i += stride)
    {
      if (s.work_generation != atomic_load (&s.generation))
        return;

      if (s.render_vibe && render_pixels[i].r2 > 250.0 * 250.0)
        continue;

      const i32 x = render_pixels[i].x;
      const i32 y = render_pixels[i].y;

      u64 r = 0, g = 0, b = 0;

      for (i32 sy = 0; sy < s.render_ss; ++sy)
        for (i32 sx = 0; sx < s.render_ss; ++sx)
          {
            if (s.work_generation != atomic_load (&s.generation))
              return;

            const f64 dx = (x - render_w / 2.0 + (sx + 0.5) * sample) * scale;
            const f64 dy = (render_h / 2.0 - y - (sy + 0.5) * sample) * scale;

            f64 nx, ny;
            f64 nu = mandelbrot (dx, dy, &nx, &ny);

            if (nu >= s.scene->iter_max)
              {
                if (color_inside && s.scene->width <= 1e-3)
                  {
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

      image_set2 (target, x, y, color_rgb (r / samples, g / samples, b / samples));
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
  fprintf (f, "  --no-gpu                       Disable GPU acceleration\n");
  fprintf (f, "  --preview                      Preview mode: reduced-res render, async upscale\n");
  fprintf (f, "  --fullscreen                   Use the desktop resolution as the window size\n");
  fprintf (f, "  --preview-div  <int>          Preview render divisor: 1 full, 2 + FSRCNN (default 2)\n");
  fprintf (f, "  --gpu-info                     Print GPU info and exit\n");
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

  bool no_gpu = false;
  bool fullscreen = false;
  preview_mode = false;
  bool gpu_info = false;

  const char *scene_path = NULL;

  const char *c_re = NULL;
  const char *c_im = NULL;

  const char *reuse_env = getenv ("MANDEX_DIFF_REUSE");

  if (reuse_env)
    {
      const long v = strtol (reuse_env, NULL, 10);
      if (v > 0)
        keyframe_max_reuse = (i32) v;
    }

  const char *bla_env = getenv ("MANDEX_BLA");

  if (bla_env && strcmp (bla_env, "0") == 0)
    bla_enabled = false;

  bla_env = getenv ("MANDEX_BLA_MARGIN");

  if (bla_env)
    {
      const f64 v = strtod (bla_env, NULL);
      if (isfinite (v) && v >= 0.0 && v <= 64.0)
        bla_margin = v;
    }

  const char *frames_env = getenv ("MANDEX_AUTO_FRAMES");

  if (frames_env)
    {
      const long v = strtol (frames_env, NULL, 10);
      if (v > 0)
        auto_frames = (usize) v;
    }

  dump_prefix = getenv ("MANDEX_DUMP");

  if (dump_prefix)
    {
      const char *limit_env = getenv ("MANDEX_DUMP_LIMIT");
      const long v = limit_env ? strtol (limit_env, NULL, 10) : 8;
      dump_limit = v > 0 ? (usize) v : 8;
    }

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

      if (strcmp (arg, "--no-gpu") == 0)
        {
          no_gpu = true;
          ++i;
          continue;
        }

      if (strcmp (arg, "--preview") == 0)
        {
          preview_mode = true;
          no_gpu = true;
          ++i;
          continue;
        }

      if (strcmp (arg, "--fullscreen") == 0)
        {
          fullscreen = true;
          ++i;
          continue;
        }

      if (strcmp (arg, "--preview-div") == 0)
        {
          if (i + 1 >= argc)
            {
              fprintf (stderr, "Error: '--preview-div' requires an integer\n");
              return 1;
            }
          const long v = strtol (argv[++i], NULL, 10);
          if (v < 1 || v > 4)
            {
              fprintf (stderr, "Error: '--preview-div' must be 1..4\n");
              return 1;
            }
          preview_div = (i32) v;
          ++i;
          continue;
        }

      if (strcmp (arg, "--gpu-info") == 0)
        {
          gpu_info = true;
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

  if (gpu_info)
    {
      if (!no_gpu)
        gpu_init ();

      printf ("GPU available: %s\n", gpu_is_available () ? "yes" : "no");

      if (gpu_is_available ())
        printf ("Device: %s\n", gpu_device_name ());

      gpu_shutdown ();

      return 0;
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

  if (fullscreen)
    {
      SDL_DisplayMode mode;
      if (SDL_GetDesktopDisplayMode (0, &mode) == 0)
        {
          initial_w = mode.w;
          initial_h = mode.h;
        }
    }

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
    | (fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0)
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

  signal (SIGPIPE, SIG_IGN);

  if (!no_gpu && !preview_mode)
    gpu_init ();

  s.phase = PHASE_IDLE;

  s.pool = thread_pool_create (SDL_GetCPUCount (), 1920 * 1080);

  orbit_cache.valid = false;
  mpfr_inits2 (FLOAT_PREC, orbit_cache.c_re, orbit_cache.c_im, (mpfr_ptr)NULL);
  view_state_init (&keyframe_view);

  s.sectors = SDL_GetCPUCount ();

  if (s.sectors < 1)
    s.sectors = 4;

  s.image = image_create (s.w, s.h);

  preview_init_root ();
  preview_update_dims ();
  preview_restart ();

  gpu_vibe_image = image_create (s.w, s.h);

  keyframe_image = image_create (s.w, s.h);

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

  if (getenv ("MANDEX_NO_AUTO"))
    auto_iter = false;

  if (auto_frames > 0)
    {
      recorded_cap = auto_frames;
      recorded_views = calloc (recorded_cap, sizeof (struct view_state));

      if (recorded_views)
        {
          f64 width = s.scene->width;

          for (usize i = 0; i < auto_frames; ++i)
            {
              struct view_state *v = &recorded_views[recorded_count++];

              view_state_init (v);
              view_state_set_center (v, s.scene->c_re, s.scene->c_im);

              v->width = width;
              v->iter_max = s.scene->iter_max;
              v->recorded_iter_max = s.scene->iter_max;
              v->iter_per_color = s.scene->iter_per_color;
              v->ss = s.ss;
              v->formula = s.formula;
              v->omni_a = s.omni_a;
              v->bailout = s.scene->bailout;
              v->lx = s.scene->lx;
              v->ly = s.scene->ly;
              v->lz = s.scene->lz;
              v->color_inside = color_inside;

              width /= 1.02;
            }

          start_render_phase ();
        }
    }

  struct work *works = calloc (n_tiles (), sizeof (struct work));

  // Measuring COMPUTE_IMAGE phase's time
  u32 compute_image_start = SDL_GetTicks ();
  u32 compute_image_total = 0;

  bool ui_info = false;
  bool ui_help = false;

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
                if (!vibe)
                  canvas_frozen = false;
                printf ("Vibe zoom: %s\n", vibe ? "on" : "off");

                if (vibe)
                  {
                    // Disable SSAA in vibe mode for speed.
                    s.ss_prev = s.ss;
                    s.ss = 1;
                  }
                else if (s.ss_prev > 0)
                  {
                    s.ss = s.ss_prev;
                  }

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

                orbit_cache.valid = false;

                s.formula = FORMULA_MANDELBROT;

                s.redraw = true;
                break;

              case SDLK_o:
                terminate ();

                orbit_cache.valid = false;

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

              case SDLK_r:
                if (rendering_video)
                  break;

                if (!recording)
                  {
                    recorded_views_free ();

                    recording = true;

                    // Recording captures full-resolution frames: suspend the
                    // half-res/FSRCNN preview for the whole recording session.
                    if (preview_mode)
                      {
                        terminate ();
                        preview_update_dims ();
                        preview_restart ();
                      }

                    // Vibe zoom gives a smooth trajectory while recording.
                    vibe = true;

                    printf ("Recording started\n");
                  }
                else
                  {
                    recording = false;

                    vibe = false;

                    if (recorded_count == 0)
                      {
                        printf ("No frames recorded\n");

                        if (preview_mode)
                          {
                            terminate ();
                            preview_update_dims ();
                            preview_restart ();
                          }
                      }
                    else
                      {
                        start_render_phase ();
                      }
                  }

                s.redraw = true;
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
            if (rendering_video)
              break;
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

                preview_update_dims ();
                preview_restart ();

                if (gpu_vibe_image)
                  image_destroy (gpu_vibe_image);
                gpu_vibe_image = image_create (s.w, s.h);

                if (keyframe_image)
                  {
                    image_destroy (keyframe_image);
                    keyframe_image = image_create (s.w, s.h);
                  }

                has_keyframe = false;

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

      if (rendering_video)
        vibe = false;

      if (vibe)
        {
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

          // Adopt a finished background GPU frame at an idle boundary; never
          // while CPU workers are filling s.image.  Renders are no longer
          // killed by a frame counter: each one completes and the canvas is
          // swapped to it, so rebasing never happens on a partial frame.
          if (s.phase != PHASE_COMPUTE_IMAGE
              && atomic_load (&gpu_vibe_done))
            {
              atomic_store (&gpu_vibe_done, false);
              for (i32 y = 0; y < s.image->h; ++y)
                for (i32 x = 0; x < s.image->w; ++x)
                  image_set2 (s.image, x, y,
                              image_get2 (gpu_vibe_image, x, y));
            }
        }

      // Capture effective view while recording.
      if (recording && !rendering_video)
        {
          mpfr_t eff_re, eff_im;
          mpfr_inits2 (FLOAT_PREC, eff_re, eff_im, (mpfr_ptr) NULL);

          // Capture the exact centre the renderer uses (the MPFR scene centre
          // plus the uncommitted pan).  s.center_re is only a double snapshot
          // and quantises the zoom target once the view is deeper than an ulp.
          mpfr_add_d (eff_re, s.scene->c_re,
                      s.pan_x * s.scene->width / (f64) s.w, MPFR_RNDN);
          mpfr_sub_d (eff_im, s.scene->c_im,
                      s.pan_y * s.scene->width / (f64) s.w, MPFR_RNDN);

          const f64 width = s.scene->width / (s.scale == 0.0 ? 1.0 : s.scale);

          // A view is a duplicate only when it moved by less than a
          // thousandth of a pixel.  The old fixed 1e-12 tolerance was
          // absolute: once the zoom was deeper than ~5e-11 it exceeded a
          // whole 2% zoom step, so every frame was silently dropped and
          // recording froze at ~1300 frames.
          const f64 view_tol = width / (f64) s.w * 1e-3;

          bool dup = false;

          if (recorded_count > 0)
            {
              struct view_state *last = &recorded_views[recorded_count - 1];

              mpfr_t d, tol;
              mpfr_inits2 (FLOAT_PREC, d, tol, (mpfr_ptr) NULL);
              mpfr_set_d (tol, view_tol, MPFR_RNDN);

              mpfr_sub (d, last->c_re_hi, eff_re, MPFR_RNDN);
              const bool same_re = mpfr_cmpabs (d, tol) <= 0;

              mpfr_sub (d, last->c_im_hi, eff_im, MPFR_RNDN);
              const bool same_im = mpfr_cmpabs (d, tol) <= 0;

              mpfr_clears (d, tol, (mpfr_ptr) NULL);

              if (last->iter_max == s.scene->iter_max
                  && last->ss == s.ss
                  && last->formula == s.formula
                  && same_re && same_im
                  && fabs (last->width - width) <= view_tol)
                dup = true;
            }

          if (!dup)
            {
              if (recorded_count == recorded_cap)
                {
                  const usize old_cap = recorded_cap;
                  usize new_cap = recorded_cap ? recorded_cap * 2 : 256;

                  if (new_cap > 65536)
                    new_cap = 65536;

                  if (recorded_count < new_cap)
                    {
                      struct view_state *nv =
                        realloc (recorded_views, new_cap * sizeof (struct view_state));

                      if (nv)
                        {
                          recorded_views = nv;
                          recorded_cap = new_cap;

                          for (usize i = old_cap; i < new_cap; ++i)
                            view_state_init (&recorded_views[i]);
                        }
                    }
                }

              if (recorded_count < recorded_cap)
                {
                  struct view_state *v = &recorded_views[recorded_count++];

                  mpfr_set (v->c_re_hi, eff_re, MPFR_RNDN);
                  mpfr_set (v->c_im_hi, eff_im, MPFR_RNDN);

                  v->c_re = mpfr_get_d (eff_re, MPFR_RNDN);
                  v->c_im = mpfr_get_d (eff_im, MPFR_RNDN);
                  v->width = width;

                  v->iter_max = s.scene->iter_max;
                  v->iter_per_color = s.scene->iter_per_color;
                  v->ss = s.ss;
                  v->formula = s.formula;
                  v->omni_a = s.omni_a;
                  v->bailout = s.scene->bailout;
                  v->lx = s.scene->lx;
                  v->ly = s.scene->ly;
                  v->lz = s.scene->lz;
                  v->color_inside = color_inside;
                }
              else
                {
                  printf ("Recording buffer full at %zu frames; rendering now\n",
                          recorded_count);
                  recording = false;
                  vibe = false;

                  start_render_phase ();
                }
            }

          mpfr_clears (eff_re, eff_im, (mpfr_ptr) NULL);
        }

      // Schedule the next frame of the video render.
      if (rendering_video && !video_frame_pending
          && render_index < recorded_count
          && s.phase == PHASE_IDLE)
        {
          struct view_state *v = &recorded_views[render_index];

          if (has_keyframe && can_reuse_keyframe (&keyframe_view, v))
            {
              warp_keyframe (&keyframe_view, v);
              video_frame_pending = true;
              keyframe_pending = false;
            }
          else
            {
              terminate ();

              s.formula = v->formula;
              s.omni_a = v->omni_a;

              // Adaptive fit of iter_max for this frame.
              {
                const u64 fitted_n = fit_video_n (v);
                if (fitted_n != v->iter_max)
                  {
                    printf ("frame %zu/%zu: iter_max %lu -> %lu\n",
                            render_index, recorded_count,
                            (unsigned long)v->iter_max, (unsigned long)fitted_n);
                    v->iter_max = fitted_n;
                  }
              }

              if (v->iter_max != s.scene->iter_max)
                apply_iter_max (v->iter_max);

              s.scene->iter_per_color = v->iter_per_color;
              s.scene->bailout = v->bailout;
              s.scene->width = v->width;

              mpfr_set (s.scene->c_re, v->c_re_hi, MPFR_RNDN);
              mpfr_set (s.scene->c_im, v->c_im_hi, MPFR_RNDN);

              s.scene->lx = v->lx;
              s.scene->ly = v->ly;
              s.scene->lz = v->lz;

              s.ss = 2;

              color_inside = v->color_inside;

              s.pan_x = 0.0;
              s.pan_y = 0.0;
              s.scale = 1.0;

              s.center_re = mpfr_get_d (s.scene->c_re, MPFR_RNDN);
              s.center_im = mpfr_get_d (s.scene->c_im, MPFR_RNDN);

              // Keep adaptive iterations active while rendering video: they
              // are exactly what keeps deep frames from blowing up the
              // per-frame budget.  The fitted N below is only a starting
              // point; auto_iter_update may refine it after each frame.
              s.redraw = true;
              video_frame_pending = true;
              keyframe_pending = true;
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

              // Keep these baked pixels on screen while the next render is
              // produced; the stale image must not overwrite them.
              if (vibe)
                canvas_frozen = true;

              mpfr_add_d (s.scene->c_re, s.scene->c_re, s.pan_x * s.scene->width / s.w, MPFR_RNDN);
              mpfr_sub_d (s.scene->c_im, s.scene->c_im, s.pan_y * s.scene->width / s.w, MPFR_RNDN);

              s.scene->width /= s.scale;

              s.pan_x = 0.0L;
              s.pan_y = 0.0L;
              s.scale = 1.0L;

              {
                i32 mx, my;
                SDL_GetMouseState (&mx, &my);

                if (mx < 0) mx = 0;
                if (mx >= s.w) mx = s.w - 1;
                if (my < 0) my = 0;
                if (my >= s.h) my = s.h - 1;

                s.mouse_cx = mx;
                s.mouse_cy = my;
              }

              // Clear the buffer the render workers actually write to.  In
              // preview mode that is the half-res buffer; leaving it uncleared
              // kept stale pixels from previous vibe frames (ghosting).
              // s.image holds the last neural frame and must stay intact.
              if (preview_active () && preview_div > 1)
                image_fill (preview_render_image, color_argb (0, 0, 0, 0));
              else if (!preview_final_valid)
                image_fill (s.image, color_argb (0, 0, 0, 0));

              atomic_store (&s.tiles_done, 0);

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

                  bool cache_hit = false;

              if (s.formula == FORMULA_MANDELBROT
                  && orbit_cache.valid
                  && orbit_cache.iter_max == s.scene->iter_max)
                {
                  const f64 cur_re = mpfr_get_d (s.scene->c_re, MPFR_RNDN);
                  const f64 cur_im = mpfr_get_d (s.scene->c_im, MPFR_RNDN);

                  const f64 dc_re = cur_re - orbit_cache.c_re_d;
                  const f64 dc_im = cur_im - orbit_cache.c_im_d;
                  const f64 width = s.scene->width;

                  if (dc_re == 0.0 && dc_im == 0.0
                      && mpfr_equal_p (orbit_cache.c_re, s.scene->c_re)
                      && mpfr_equal_p (orbit_cache.c_im, s.scene->c_im))
                    {
                      delta_c_re = 0.0;
                      delta_c_im = 0.0;
                      cache_hit = true;
                    }
                  else if (width > ORBIT_REUSE_MIN_WIDTH
                           && fabs (dc_re) / width < ORBIT_REUSE_EPS
                           && fabs (dc_im) / width < ORBIT_REUSE_EPS)
                    {
                      delta_c_re = dc_re;
                      delta_c_im = dc_im;
                      cache_hit = true;
                    }
                }

              if (cache_hit)
                {
                  atomic_store (&s.orbit_ready, true);
                }
              else
                {
                  delta_c_re = 0.0;
                  delta_c_im = 0.0;

                  // orbit_ready is still true from the previous frame.  Clear
                  // it before queueing compute_orbit, otherwise the next tick
                  // starts render workers while compute_orbit is rebuilding
                  // bla_table -> use-after-free in bla_try_skip().
                  atomic_store (&s.orbit_ready, false);

                  thread_pool_enqueue (s.pool, compute_orbit, NULL);
                }
            }
          break;

        case PHASE_COMPUTE_ORBIT:
          if (atomic_load (&s.orbit_ready))
            {
              s.work_generation = atomic_load (&s.generation);

              compute_image_start = SDL_GetTicks ();

              s.phase = PHASE_COMPUTE_IMAGE;

              s.render_ss = s.ss;
              s.render_vibe = vibe;

              // In vibe mode render on the CPU so the expanding circle of
              // progress stays visible (GPU walks a 2D grid and commits the
              // whole frame at once).  Non-vibe renders may still use the GPU.
              // BLA owns the deep-zoom path (the GPU kernel cannot compile
              // its extended-exponent coefficients here); everywhere else the
              // GPU is free to accelerate the render.
              const bool use_gpu =
                !vibe
                && gpu_is_available ()
                && !preview_mode
                && !(bla_enabled && s.scene->width <= 1e-12)
                && gpu_should_use (s.w, s.h, s.render_ss, s.scene->iter_max,
                                   s.scene->width, s.formula, s.render_vibe);

              // Build a list of every pixel, sorted by distance from the mouse,
              // so CPU rendering fills the image in expanding rings (circles)
              // around the cursor with no square-tile artefacts.  The GPU
              // kernel walks its own 2D grid, so this work is skipped there.
              if (!use_gpu)
                {
                  render_pixel_count = (usize) render_w * (usize) render_h;

                  if (render_pixel_count > render_pixels_cap)
                    {
                      free (render_pixels);
                      render_pixels = malloc (render_pixel_count * sizeof (struct point));
                      render_pixels_cap = render_pixels ? render_pixel_count : 0;
                    }

                  if (render_pixels)
                    {
                      i32 maxr2 = 0;
                      const i32 mcx =
                        preview_active () ? s.mouse_cx / preview_div : s.mouse_cx;
                      const i32 mcy =
                        preview_active () ? s.mouse_cy / preview_div : s.mouse_cy;
                      const i32 dxs[2] = { mcx, render_w - 1 - mcx };
                      const i32 dys[2] = { mcy, render_h - 1 - mcy };

                      for (i32 i = 0; i < 2; ++i)
                        for (i32 j = 0; j < 2; ++j)
                          {
                            const i32 r2 = dxs[i] * dxs[i] + dys[j] * dys[j];
                            if (r2 > maxr2) maxr2 = r2;
                          }

                      if ((usize)maxr2 + 1 > render_counts_cap)
                        {
                          free (render_counts);
                          render_counts = calloc ((usize)maxr2 + 1, sizeof (usize));
                          render_counts_cap = render_counts ? (usize)maxr2 + 1 : 0;
                        }

                      if (render_counts)
                        {
                          memset (render_counts, 0, ((usize)maxr2 + 1) * sizeof (usize));

                          for (i32 y = 0; y < render_h; ++y)
                            for (i32 x = 0; x < render_w; ++x)
                              {
                                const i32 dx = x - mcx;
                                const i32 dy = y - mcy;
                                const i32 r2 = dx * dx + dy * dy;
                                render_counts[r2]++;
                              }

                          usize sum = 0;
                          for (i32 r = 0; r <= maxr2; ++r)
                            {
                              const usize c = render_counts[r];
                              render_counts[r] = sum;
                              sum += c;
                            }

                          for (i32 y = 0; y < render_h; ++y)
                            for (i32 x = 0; x < render_w; ++x)
                              {
                                const i32 dx = x - mcx;
                                const i32 dy = y - mcy;
                                const i32 r2 = dx * dx + dy * dy;

                                render_pixels[render_counts[r2]].x = x;
                                render_pixels[render_counts[r2]].y = y;
                                render_pixels[render_counts[r2]].r2 = (f64)r2;
                                render_counts[r2]++;
                              }
                        }
                    }
                }

              if (use_gpu)
                {
                  struct gpu_view gv;
                  gv.c_re = s.center_re;
                  gv.c_im = s.center_im;
                  gv.width = s.scene->width;
                  gv.iter_max = s.scene->iter_max;
                  gv.iter_per_color = s.scene->iter_per_color;
                  gv.ss = s.render_ss;
                  gv.formula = s.formula;
                  gv.omni_a = s.omni_a;
                  gv.bailout = s.scene->bailout;
                  gv.lx = s.scene->lx;
                  gv.ly = s.scene->ly;
                  gv.lz = s.scene->lz;
                  gv.color_inside = color_inside;
                  gv.w = s.w;
                  gv.h = s.h;
                  gv.palette = s.palette;
                  gv.orbit_re = s.orbit_re;
                  gv.orbit_im = s.orbit_im;
                  gv.delta_c_re = delta_c_re;
                  gv.delta_c_im = delta_c_im;
                  gv.period_tol = period_tol;

                  gpu_job_args.view = gv;
                  gpu_job_args.generation = s.work_generation;

                  atomic_store (&gpu_job_active, true);

                  if (thread_pool_enqueue (s.pool, gpu_job, &gpu_job_args) != 0)
                    {
                      // The pool is full.  Fall back to the old synchronous
                      // path instead of silently leaving the image empty.
                      atomic_store (&gpu_job_active, false);

                      gpu_render_mandelbrot (&gv, s.image);

                      atomic_store (&s.tiles_done, n_tiles ());
                    }
                }
              else
                {
                  const i32 workers = n_tiles ();

                  for (i32 wkr = 0; wkr < workers; ++wkr)
                    {
                      works[wkr].offset = wkr;
                      works[wkr].stride = workers;

                      thread_pool_enqueue (s.pool, worker, &works[wkr]);
                    }
                }

            }
              // In vibe mode, also kick off a background GPU render into a
              // separate buffer.  When it completes, its sharp result replaces
              // the partial CPU circle.
              if (vibe && s.scene->iter_max <= 1000 && !gpu_vibe_inflight)
                {
                  struct gpu_view gv;
                  gv.c_re = s.center_re;
                  gv.c_im = s.center_im;
                  gv.width = s.scene->width;
                  gv.iter_max = s.scene->iter_max;
                  gv.iter_per_color = s.scene->iter_per_color;
                  gv.ss = s.render_ss;
                  gv.formula = s.formula;
                  gv.omni_a = s.omni_a;
                  gv.bailout = s.scene->bailout;
                  gv.lx = s.scene->lx;
                  gv.ly = s.scene->ly;
                  gv.lz = s.scene->lz;
                  gv.color_inside = color_inside;
                  gv.w = s.w;
                  gv.h = s.h;
                  gv.palette = s.palette;
                  gv.orbit_re = s.orbit_re;
                  gv.orbit_im = s.orbit_im;
                  gv.delta_c_re = delta_c_re;
                  gv.delta_c_im = delta_c_im;
                  gv.period_tol = period_tol;

                  gpu_vibe_job_args.view = gv;
                  gpu_vibe_job_args.generation = s.work_generation;
                  gpu_vibe_job_args.out = gpu_vibe_image;
                  gpu_vibe_job_args.vibe = true;

                  atomic_store (&gpu_vibe_inflight, true);
                  atomic_store (&gpu_vibe_done, false);

                  if (thread_pool_enqueue (s.pool, gpu_job, &gpu_vibe_job_args) != 0)
                    {
                      atomic_store (&gpu_vibe_inflight, false);
                      gpu_render_mandelbrot (&gv, gpu_vibe_image);
                      atomic_store (&gpu_vibe_done, true);
                    }
                }

          break;

        case PHASE_COMPUTE_IMAGE:
          if (s.tiles_done == n_tiles ())
            {
              compute_image_total = SDL_GetTicks () - compute_image_start;

              // The buffer now holds a full frame for the current scene.
              canvas_frozen = false;

              preview_submit_job ();

              const bool can_probe = auto_iter
                                     && !vibe
                                     && !rendering_video
                                     && probe_width > AUTO_DEEP_WIDTH
                                     && isfinite (probe_width);

              if (!can_probe)
                {
                  if (!rendering_video)
                    {
                      auto_iter_update (compute_image_total, vibe);
                      persist_video_iter ();
                    }
                  s.phase = PHASE_IDLE;

                  if (vibe)
                    s.redraw = true;

                  break;
                }

              probe_changed = 0;
              probe_total = 0;

              atomic_store (&probe_done, false);

              if (thread_pool_enqueue (s.pool, probe_job, NULL) == 0)
                s.phase = PHASE_PROBE;
              else
                {
                  if (!rendering_video)
                    {
                      auto_iter_update (compute_image_total, vibe);
                      persist_video_iter ();
                    }
                  s.phase = PHASE_IDLE;
                }
            }
          break;

        case PHASE_PROBE:
          if (atomic_load (&probe_done))
            {
              if (!rendering_video)
                {
                  auto_iter_update (compute_image_total, vibe);
                  persist_video_iter ();
                }

              s.phase = PHASE_IDLE;
            }
          break;

        default:
          break;
        }

      // Write a completed video frame.
      if (rendering_video && video_frame_pending && s.phase == PHASE_IDLE)
        {
          if (ffmpeg_pipe)
            {
              for (i32 y = 0; y < s.image->h; ++y)
                for (i32 x = 0; x < s.image->w; ++x)
                  {
                    const u32 c = image_get2 (s.image, x, y);

                    const unsigned char rgb[3] = { color_r (c), color_g (c), color_b (c) };

                    fwrite (rgb, 1, 3, ffmpeg_pipe);
                  }

              fflush (ffmpeg_pipe);
            }

          if (keyframe_pending)
            {
              image_copy (s.image, keyframe_image);
              view_state_copy (&keyframe_view, &recorded_views[render_index]);
              has_keyframe = true;
              keyframe_pending = false;
              keyframe_render_index = render_index;
            }

          dump_ppm ();

          video_frame_pending = false;
          render_index++;

          if (render_index >= recorded_count)
            {
              if (ffmpeg_pipe)
                {
                  pclose (ffmpeg_pipe);
                  ffmpeg_pipe = NULL;
                }

              rendering_video = false;

              // Navigation preview resumes only after the last video frame;
              // restoring it earlier would let the neural upscale leak into
              // the remaining frames of the MP4.
              if (preview_mode)
                {
                  preview_update_dims ();
                  preview_restart ();
                }

              printf ("Recording render finished: %zu frames -> '%s'\n",
                      recorded_count, video_path);

              if (auto_frames > 0)
                s.running = false;
            }
        }

      // A GPU job writes s.image from a worker.  Never upload a frame while
      // that job is still running: the result is not a coherent image yet.
      // (CPU renders update the image atomically, so partial frames are safe
      // to upload and show the expanding circle of progress.)
      const bool gpu_busy = atomic_load (&gpu_job_active);

      const bool vibe_upload_ok = !gpu_busy && !canvas_frozen;

      if (vibe_upload_ok)
        {
          struct image *shown = preview_display_image ();

          // Upload texture (atomic per-pixel read, safe for partial frames).
          {
            i32 pitch;
            void *pixels;

            SDL_LockTexture (s.texture, NULL, &pixels, &pitch);

            for (i32 y = 0; y < shown->h; ++y)
              {
                u32 *row = (u32 *)((u8 *)pixels + (usize)y * (usize)pitch);

                for (i32 x = 0; x < shown->w; ++x)
                  row[x] = image_get2 (shown, x, y);
              }

            SDL_UnlockTexture (s.texture);
          }

          SDL_SetRenderTarget (s.renderer, s.texture_canvas);
          SDL_RenderCopy (s.renderer, s.texture, NULL, NULL);
        }

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
          write_ct (y++, "Record video (R toggle) ; r");
          write_ct (y++, "Terminate ; Escape");
          write_ct (y++, "Pan ; Left Click + Drag");
          write_ct (y++, "Zoom toward mouse ; Scroll Wheel");
          write_ct (y++, "Zoom toward center ; Control + Scroll Wheel");
          write_ct (y++, "Zoom slower ; Shift + Scroll Wheel");

#undef write_ct
        }

      if (recording)
        write_text (s.w - 80, 10, 255, 64, 64, "REC");

      if (rendering_video)
        write_text (10, 10, 196, 196, 196, "Rendering video: %zu/%zu",
                    render_index, recorded_count);

      SDL_RenderPresent (s.renderer);
    }

  if (ffmpeg_pipe)
    {
      pclose (ffmpeg_pipe);
      ffmpeg_pipe = NULL;
    }

  recorded_views_free ();
  view_state_clear (&keyframe_view);

  terminate ();

  mpfr_clears (orbit_cache.c_re, orbit_cache.c_im, (mpfr_ptr)NULL);

  if (keyframe_image)
    image_destroy (keyframe_image);

  if (gpu_vibe_image)
    image_destroy (gpu_vibe_image);

  image_destroy (s.image);

  preview_stop_thread ();

  if (preview_job_src) image_destroy (preview_job_src);
  if (preview_job_dst) image_destroy (preview_job_dst);
  if (preview_fast_image) image_destroy (preview_fast_image);
  if (preview_render_image) image_destroy (preview_render_image);

  free (render_pixels);
  free (render_counts);

  bla_clear (&bla_table);

  SDL_DestroyTexture (s.texture);
  SDL_DestroyRenderer (s.renderer);
  SDL_DestroyWindow (s.window);

  palette_destroy (s.palette);
  scene_destroy (s.scene);

  gpu_shutdown ();

  TTF_Quit ();
  SDL_Quit ();

  printf ("Goodbye.\n");

  return 0;
}

