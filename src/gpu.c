#include "gpu.h"

#include "color.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#if HAVE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif


#if HAVE_OPENCL

static bool g_available = false;
static bool g_fp64 = false;

static cl_platform_id g_platform = NULL;
static cl_device_id g_device = NULL;
static cl_context g_context = NULL;
static cl_command_queue g_queue = NULL;
static cl_program g_program = NULL;
static cl_kernel g_kernel = NULL;

static cl_mem g_output = NULL;
static i32 g_out_w = 0;
static i32 g_out_h = 0;

static cl_mem g_palette = NULL;
static const struct palette *g_palette_ptr = NULL;
static i32 g_palette_size = 0;

static cl_mem g_orbit_re = NULL;
static cl_mem g_orbit_im = NULL;
static usize g_orbit_n = 0;

static char g_device_name[256] = { 0 };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"

static const char *kernel_src =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void mandelbrot_kernel(\n"
"  const double c_re, const double c_im, const double width,\n"
"  const ulong iter_max, const ulong iter_per_color,\n"
"  const int ss, const double bailout,\n"
"  const double lx, const double ly, const double lz,\n"
"  const int color_inside,\n"
"  __global const double *orbit_re, __global const double *orbit_im,\n"
"  const double delta_c_re, const double delta_c_im, const double period_tol,\n"
"  __global const uint *palette, const int palette_size,\n"
"  __global uchar4 *output,\n"
"  const int W, const int H)\n"
"{\n"
"  const int x = get_global_id(0);\n"
"  const int y = get_global_id(1);\n"
"  if (x >= W || y >= H) return;\n"
"  const double scale = width / (double)W;\n"
"  const double sample = 1.0 / (double)ss;\n"
"  double r = 0.0, g = 0.0, b = 0.0;\n"
"  for (int sy = 0; sy < ss; ++sy) {\n"
"    for (int sx = 0; sx < ss; ++sx) {\n"
"      const double dx = ((double)x - (double)W / 2.0 + (sx + 0.5) * sample) * scale;\n"
"      const double dy = ((double)H / 2.0 - (double)y - (sy + 0.5) * sample) * scale;\n"
"      const double px = c_re + dx;\n"
"      const double py = c_im + dy;\n"
"      const double dx2 = px + 1.0;\n"
"      if (dx2 * dx2 + py * py <= 1.0 / 16.0) {\n"
"        if (color_inside && width <= 1e-3) { r += 24; g += 20; b += 48; }\n"
"        continue;\n"
"      }\n"
"      const double qx = px - 0.25;\n"
"      const double q = qx * qx + py * py;\n"
"      if (q * (q + qx) <= py * py / 4.0) {\n"
"        if (color_inside && width <= 1e-3) { r += 24; g += 20; b += 48; }\n"
"        continue;\n"
"      }\n"
"      double delta_re = dx + delta_c_re;\n"
"      double delta_im = dy + delta_c_im;\n"
"      double epsilon_re = 0.0, epsilon_im = 0.0;\n"
"      double deriv_re = 0.0, deriv_im = 0.0;\n"
"      ulong iter = 0;\n"
"      ulong iter_orbit = 0;\n"
"      ulong period = 1;\n"
"      double saved_re = 0.0, saved_im = 0.0;\n"
"      int have_saved = 0;\n"
"      const double bailout2 = bailout * bailout;\n"
"      int escaped = 0;\n"
"      double nu = (double)iter_max;\n"
"      double nx = 0.0, ny = 0.0;\n"
"      while (iter < iter_max && !escaped) {\n"
"        double r_re = orbit_re[iter_orbit];\n"
"        double r_im = orbit_im[iter_orbit];\n"
"        double z_re = r_re + epsilon_re;\n"
"        double z_im = r_im + epsilon_im;\n"
"        double new_dr = 2.0 * (z_re * deriv_re - z_im * deriv_im) + 1.0;\n"
"        double new_di = 2.0 * (z_re * deriv_im + z_im * deriv_re);\n"
"        deriv_re = new_dr;\n"
"        deriv_im = new_di;\n"
"        double lin_re = 2.0 * (r_re * epsilon_re - r_im * epsilon_im);\n"
"        double lin_im = 2.0 * (r_re * epsilon_im + r_im * epsilon_re);\n"
"        double quad_re = epsilon_re * epsilon_re - epsilon_im * epsilon_im;\n"
"        double quad_im = 2.0 * epsilon_re * epsilon_im;\n"
"        epsilon_re = lin_re + quad_re + delta_re;\n"
"        epsilon_im = lin_im + quad_im + delta_im;\n"
"        r_re = orbit_re[iter_orbit + 1];\n"
"        r_im = orbit_im[iter_orbit + 1];\n"
"        z_re = r_re + epsilon_re;\n"
"        z_im = r_im + epsilon_im;\n"
"        double z_mag2 = z_re * z_re + z_im * z_im;\n"
"        if (have_saved) {\n"
"          double drp = z_re - saved_re;\n"
"          double dip = z_im - saved_im;\n"
"          if (period_tol > 1e-61 && (drp*drp + dip*dip) < period_tol * (1.0 + z_mag2)) {\n"
"            iter = iter_max;\n"
"            break;\n"
"          }\n"
"        }\n"
"        if (iter + 1 == period) {\n"
"          saved_re = z_re;\n"
"          saved_im = z_im;\n"
"          have_saved = 1;\n"
"          period *= 2;\n"
"          if (period > iter_max) period = iter_max;\n"
"        }\n"
"        if (z_mag2 > bailout2) {\n"
"          double a = z_re * deriv_re + z_im * deriv_im;\n"
"          double bb = z_im * deriv_re - z_re * deriv_im;\n"
"          double len = hypot(a, bb);\n"
"          nx = (len > 0.0) ? a / len : 0.0;\n"
"          ny = (len > 0.0) ? bb / len : 0.0;\n"
"          nu = (double)(iter + 1) - log(log(z_mag2) * 0.5) / 0.6931471805599453;\n"
"          if (nu < 0.0) nu = 0.0;\n"
"          escaped = 1;\n"
"          break;\n"
"        }\n"
"        iter++;\n"
"        iter_orbit++;\n"
"        if (epsilon_re * epsilon_re + epsilon_im * epsilon_im > z_mag2 / 2.0) {\n"
"          epsilon_re = z_re;\n"
"          epsilon_im = z_im;\n"
"          iter_orbit = 0;\n"
"        }\n"
"      }\n"
"      if (!escaped) {\n"
"        if (color_inside && width <= 1e-3) { r += 24; g += 20; b += 48; }\n"
"        continue;\n"
"      }\n"
"      const double ipc = (iter_per_color > 0) ? (double)iter_per_color : 1.0;\n"
"      const double xcol = nu / ipc;\n"
"      long ii = (long)xcol;\n"
"      double ff = xcol - (double)ii;\n"
"      if (ff < 0.0) { ff += 1.0; ii -= 1; }\n"
"      int i1 = (int)(ii % (long)palette_size);\n"
"      if (i1 < 0) i1 += palette_size;\n"
"      int i2 = (i1 + 1) % palette_size;\n"
"      uint c1 = palette[i1];\n"
"      uint c2 = palette[i2];\n"
"      int r1 = (c1 >> 16) & 0xFF;\n"
"      int g1 = (c1 >> 8) & 0xFF;\n"
"      int b1 = c1 & 0xFF;\n"
"      int r2 = (c2 >> 16) & 0xFF;\n"
"      int g2 = (c2 >> 8) & 0xFF;\n"
"      int b2 = c2 & 0xFF;\n"
"      int rr = r1 + (int)((r2 - r1) * ff + 0.5);\n"
"      int gg = g1 + (int)((g2 - g1) * ff + 0.5);\n"
"      int bb = b1 + (int)((b2 - b1) * ff + 0.5);\n"
"      double shade = (nx * lx + ny * ly + lz) / sqrt(nx * nx + ny * ny + 1.0);\n"
"      shade = fmax(0.0, shade * 0.7 + 0.3);\n"
"      r += rr * shade;\n"
"      g += gg * shade;\n"
"      b += bb * shade;\n"
"    }\n"
"  }\n"
"  const int samples = ss * ss;\n"
"  uchar rrr = (uchar)(r / samples);\n"
"  uchar ggg = (uchar)(g / samples);\n"
"  uchar bbb = (uchar)(b / samples);\n"
"  output[y * W + x] = (uchar4)(rrr, ggg, bbb, 255);\n"
"}\n";

#pragma GCC diagnostic pop

static bool
has_extension (cl_device_id dev, const char *ext)
{
  size_t size = 0;

  if (clGetDeviceInfo (dev, CL_DEVICE_EXTENSIONS, 0, NULL, &size) != CL_SUCCESS)
    return false;

  char *str = malloc (size + 1);

  if (!str)
    return false;

  if (clGetDeviceInfo (dev, CL_DEVICE_EXTENSIONS, size, str, NULL) != CL_SUCCESS)
    {
      free (str);
      return false;
    }

  str[size] = '\0';

  bool found = strstr (str, ext) != NULL;

  free (str);

  return found;
}


static void
get_info_str (cl_device_info param, char *buf, size_t size)
{
  size_t sz = 0;

  clGetDeviceInfo (g_device, param, 0, NULL, &sz);

  if (sz > size)
    sz = size - 1;

  clGetDeviceInfo (g_device, param, sz, buf, NULL);

  buf[sz] = '\0';
}


static void
print_build_log (void)
{
  size_t log_size = 0;

  clGetProgramBuildInfo (g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                         &log_size);

  char *log = malloc (log_size + 1);

  if (!log)
    return;

  clGetProgramBuildInfo (g_program, g_device, CL_PROGRAM_BUILD_LOG, log_size,
                         log, NULL);

  log[log_size] = '\0';

  fprintf (stderr, "OpenCL build log:\n%s\n", log);

  free (log);
}


static void
compute_cache_path (char *buf, size_t size)
{
  buf[0] = '\0';

  const char *home = getenv ("HOME");

  if (!home)
    return;

  char name[256] = { 0 };
  char ver[256] = { 0 };
  char drv[256] = { 0 };

  get_info_str (CL_DEVICE_NAME, name, sizeof name);
  get_info_str (CL_DEVICE_VERSION, ver, sizeof ver);
  get_info_str (CL_DRIVER_VERSION, drv, sizeof drv);

  unsigned long h = 5381;

  char tmp[1024];

  snprintf (tmp, sizeof tmp, "%s|%s|%s", name, ver, drv);

  for (char *p = tmp; *p; ++p)
    h = h * 33 + (unsigned char) *p;

  snprintf (buf, size, "%s/.cache/mandexplorer/kernel_%lx.bin", home, h);

  char dir[4096];

  snprintf (dir, sizeof dir, "%s/.cache/mandexplorer", home);

  mkdir (dir, 0755);
}


static void
save_cache (const char *path)
{
  size_t bin_size = 0;

  clGetProgramInfo (g_program, CL_PROGRAM_BINARY_SIZES, sizeof (size_t),
                    &bin_size, NULL);

  if (bin_size == 0)
    return;

  unsigned char *bin = malloc (bin_size);

  if (!bin)
    return;

  unsigned char **bins = &bin;

  clGetProgramInfo (g_program, CL_PROGRAM_BINARIES, sizeof (unsigned char *),
                    bins, NULL);

  FILE *f = fopen (path, "wb");

  if (f)
    {
      fwrite (bin, 1, bin_size, f);
      fclose (f);
    }

  free (bin);
}


static bool
build_program (void)
{
  char path[4096];

  compute_cache_path (path, sizeof path);

  bool loaded = false;

  if (path[0])
    {
      FILE *f = fopen (path, "rb");

      if (f)
        {
          fseek (f, 0, SEEK_END);

          long sz = ftell (f);

          fseek (f, 0, SEEK_SET);

          if (sz > 0)
            {
              unsigned char *buf = malloc (sz);

              if (buf && fread (buf, 1, sz, f) == (size_t) sz)
                {
                  size_t len = (size_t) sz;

                  cl_int err_bin = 0;
                  cl_int status = 0;

                  g_program = clCreateProgramWithBinary (g_context, 1, &g_device,
                                                        &len,
                                                        (const unsigned char **) &buf,
                                                        &status, &err_bin);

                  if (err_bin == CL_SUCCESS && status == CL_SUCCESS)
                    {
                      cl_int berr = clBuildProgram (g_program, 1, &g_device, "", NULL,
                                                   NULL);

                      if (berr == CL_SUCCESS)
                        loaded = true;
                      else
                        {
                          print_build_log ();

                          clReleaseProgram (g_program);
                          g_program = NULL;
                        }
                    }
                  else
                    {
                      if (g_program)
                        {
                          clReleaseProgram (g_program);
                          g_program = NULL;
                        }
                    }
                }

              free (buf);
            }

          fclose (f);
        }
    }

  if (!loaded)
    {
      cl_int err;

      size_t src_len = strlen (kernel_src);

      g_program = clCreateProgramWithSource (g_context, 1, &kernel_src, &src_len,
                                            &err);

      if (err != CL_SUCCESS || !g_program)
        return false;

      cl_int berr = clBuildProgram (g_program, 1, &g_device, "", NULL, NULL);

      if (berr != CL_SUCCESS)
        {
          print_build_log ();

          clReleaseProgram (g_program);
          g_program = NULL;

          return false;
        }

      if (path[0])
        save_cache (path);
    }

  return true;
}


static void
ensure_output (i32 w, i32 h)
{
  if (g_output && g_out_w == w && g_out_h == h)
    return;

  if (g_output)
    {
      clReleaseMemObject (g_output);
      g_output = NULL;
    }

  g_out_w = w;
  g_out_h = h;

  cl_int err;

  g_output = clCreateBuffer (g_context, CL_MEM_WRITE_ONLY,
                             (size_t) w * h * 4, NULL, &err);

  if (err != CL_SUCCESS)
    g_output = NULL;
}


static void
ensure_palette (const struct palette *pal)
{
  if (g_palette && g_palette_ptr == pal && g_palette_size == (i32) pal->size)
    return;

  if (g_palette)
    {
      clReleaseMemObject (g_palette);
      g_palette = NULL;
    }

  g_palette_ptr = pal;
  g_palette_size = (i32) pal->size;

  cl_int err;

  g_palette = clCreateBuffer (g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              pal->size * sizeof (cl_uint), (void *) pal->data,
                              &err);

  if (err != CL_SUCCESS)
    g_palette = NULL;
}


static void
ensure_orbit (const f64 *re, const f64 *im, u64 n)
{
  if (n == 0)
    return;

  if (!g_orbit_re || g_orbit_n != n)
    {
      if (g_orbit_re)
        clReleaseMemObject (g_orbit_re);

      if (g_orbit_im)
        clReleaseMemObject (g_orbit_im);

      g_orbit_re = NULL;
      g_orbit_im = NULL;

      cl_int err;

      g_orbit_re = clCreateBuffer (g_context, CL_MEM_READ_ONLY,
                                    (n + 1) * sizeof (f64), NULL, &err);

      g_orbit_im = clCreateBuffer (g_context, CL_MEM_READ_ONLY,
                                    (n + 1) * sizeof (f64), NULL, &err);

      g_orbit_n = n;
    }

  clEnqueueWriteBuffer (g_queue, g_orbit_re, CL_TRUE, 0,
                        (n + 1) * sizeof (f64), re, 0, NULL, NULL);

  clEnqueueWriteBuffer (g_queue, g_orbit_im, CL_TRUE, 0,
                        (n + 1) * sizeof (f64), im, 0, NULL, NULL);
}


bool
gpu_init (void)
{
  cl_int err;

  cl_uint num_platforms = 0;

  if (clGetPlatformIDs (0, NULL, &num_platforms) != CL_SUCCESS
      || num_platforms == 0)
    return false;

  cl_platform_id *platforms = calloc (num_platforms, sizeof (cl_platform_id));

  if (!platforms)
    return false;

  clGetPlatformIDs (num_platforms, platforms, NULL);

  cl_device_id best_dev = NULL;
  cl_platform_id best_plat = NULL;
  int best_score = 0;

  for (cl_uint p = 0; p < num_platforms; ++p)
    {
      cl_uint num_devs = 0;

      if (clGetDeviceIDs (platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devs)
          != CL_SUCCESS)
        continue;

      if (num_devs == 0)
        continue;

      cl_device_id *devs = calloc (num_devs, sizeof (cl_device_id));

      if (!devs)
        continue;

      clGetDeviceIDs (platforms[p], CL_DEVICE_TYPE_ALL, num_devs, devs, NULL);

      for (cl_uint d = 0; d < num_devs; ++d)
        {
          if (!has_extension (devs[d], "cl_khr_fp64"))
            continue;

          cl_device_type t = 0;

          clGetDeviceInfo (devs[d], CL_DEVICE_TYPE, sizeof (t), &t, NULL);

          int score = (t & CL_DEVICE_TYPE_GPU) ? 2 : 1;

          if (score > best_score)
            {
              best_score = score;
              best_dev = devs[d];
              best_plat = platforms[p];
            }
        }

      free (devs);
    }

  free (platforms);

  if (!best_dev)
    return false;

  g_platform = best_plat;
  g_device = best_dev;
  g_fp64 = true;

  size_t sz = 0;

  clGetDeviceInfo (g_device, CL_DEVICE_NAME, 0, NULL, &sz);

  if (sz > sizeof (g_device_name))
    sz = sizeof (g_device_name) - 1;

  clGetDeviceInfo (g_device, CL_DEVICE_NAME, sz, g_device_name, NULL);

  g_device_name[sz] = '\0';

  g_context = clCreateContext (NULL, 1, &g_device, NULL, NULL, &err);

  if (err != CL_SUCCESS || !g_context)
    return false;

  g_queue = clCreateCommandQueue (g_context, g_device, 0, &err);

  if (err != CL_SUCCESS || !g_queue)
    {
      clReleaseContext (g_context);
      g_context = NULL;
      return false;
    }

  if (!build_program ())
    {
      clReleaseCommandQueue (g_queue);
      g_queue = NULL;
      clReleaseContext (g_context);
      g_context = NULL;
      return false;
    }

  g_kernel = clCreateKernel (g_program, "mandelbrot_kernel", &err);

  if (err != CL_SUCCESS || !g_kernel)
    {
      clReleaseProgram (g_program);
      g_program = NULL;
      clReleaseCommandQueue (g_queue);
      g_queue = NULL;
      clReleaseContext (g_context);
      g_context = NULL;
      return false;
    }

  g_available = true;

  return true;
}


void
gpu_shutdown (void)
{
  if (g_kernel)
    {
      clReleaseKernel (g_kernel);
      g_kernel = NULL;
    }

  if (g_program)
    {
      clReleaseProgram (g_program);
      g_program = NULL;
    }

  if (g_output)
    {
      clReleaseMemObject (g_output);
      g_output = NULL;
    }

  if (g_palette)
    {
      clReleaseMemObject (g_palette);
      g_palette = NULL;
    }

  if (g_orbit_re)
    {
      clReleaseMemObject (g_orbit_re);
      g_orbit_re = NULL;
    }

  if (g_orbit_im)
    {
      clReleaseMemObject (g_orbit_im);
      g_orbit_im = NULL;
    }

  if (g_queue)
    {
      clReleaseCommandQueue (g_queue);
      g_queue = NULL;
    }

  if (g_context)
    {
      clReleaseContext (g_context);
      g_context = NULL;
    }

  g_available = false;
  g_fp64 = false;
}


bool
gpu_is_available (void)
{
  return g_available;
}


bool
gpu_supports_fp64 (void)
{
  return g_fp64;
}


bool
gpu_should_use (i32 w, i32 h, i32 ss, u64 iter_max, f64 width,
                enum formula formula, bool vibe)
{
  if (!g_available)
    return false;

  if (formula == FORMULA_OMNIBROT)
    return false;

  if (w * h < 128 * 128)
    return false;

  if (vibe)
    {
      if (iter_max > 500000 && width < 1e-13)
        return false;
    }
  else if (iter_max > 500000 && width < 1e-10)
    return false;

  (void) ss;

  return true;
}


void
gpu_render_mandelbrot (const struct gpu_view *v, struct image *out)
{
  if (!g_available || !v || !out)
    return;



  if (v->formula == FORMULA_OMNIBROT)
    return;

  ensure_output (v->w, v->h);
  ensure_palette (v->palette);

  if (v->orbit_re && v->orbit_im)
    ensure_orbit (v->orbit_re, v->orbit_im, v->iter_max);

  if (!g_output || !g_palette)
    return;

  cl_int err;

  int arg = 0;

  double c_re = v->c_re;
  double c_im = v->c_im;
  double width = v->width;
  cl_ulong iter_max = v->iter_max;
  cl_ulong iter_per_color = v->iter_per_color;
  cl_int ss = v->ss;
  double bailout = v->bailout;
  double lx = v->lx, ly = v->ly, lz = v->lz;
  cl_int color_inside = v->color_inside ? 1 : 0;
  cl_int psz = g_palette_size;
  cl_int W = v->w, H = v->h;

  clSetKernelArg (g_kernel, arg++, sizeof (double), &c_re);
  clSetKernelArg (g_kernel, arg++, sizeof (double), &c_im);
  clSetKernelArg (g_kernel, arg++, sizeof (double), &width);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_ulong), &iter_max);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_ulong), &iter_per_color);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_int), &ss);
  clSetKernelArg (g_kernel, arg++, sizeof (double), &bailout);
  clSetKernelArg (g_kernel, arg++, sizeof (double), &lx);
  clSetKernelArg (g_kernel, arg++, sizeof (double), &ly);
  clSetKernelArg (g_kernel, arg++, sizeof (double), &lz);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_int), &color_inside);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_mem), &g_orbit_re);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_mem), &g_orbit_im);
  double delta_c_re = v->delta_c_re;
  double delta_c_im = v->delta_c_im;
  double period_tol = v->period_tol;
  clSetKernelArg (g_kernel, arg++, sizeof (double), &delta_c_re);
  clSetKernelArg (g_kernel, arg++, sizeof (double), &delta_c_im);
  clSetKernelArg (g_kernel, arg++, sizeof (double), &period_tol);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_mem), &g_palette);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_int), &psz);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_mem), &g_output);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_int), &W);
  clSetKernelArg (g_kernel, arg++, sizeof (cl_int), &H);

  size_t global[2] = { (size_t) v->w, (size_t) v->h };

  err = clEnqueueNDRangeKernel (g_queue, g_kernel, 2, NULL, global, NULL, 0, NULL,
                                NULL);

  if (err != CL_SUCCESS)
    {
      fprintf (stderr, "clEnqueueNDRangeKernel: %d\n", err);
      return;
    }

  clFinish (g_queue);

  size_t bytes = (size_t) v->w * v->h * 4;

  unsigned char *buf = malloc (bytes);

  if (!buf)
    return;

  err = clEnqueueReadBuffer (g_queue, g_output, CL_TRUE, 0, bytes, buf, 0, NULL,
                             NULL);

  if (err == CL_SUCCESS)
    {
      for (i32 y = 0; y < v->h; ++y)
        for (i32 x = 0; x < v->w; ++x)
          {
            size_t idx = ((size_t) y * v->w + x) * 4;

            u32 color = color_argb (255, buf[idx], buf[idx + 1], buf[idx + 2]);

            image_set2 (out, x, y, color);
          }
    }
  else
    {
      fprintf (stderr, "clEnqueueReadBuffer: %d\n", err);
    }

  free (buf);
}


const char *
gpu_device_name (void)
{
  return g_device_name;
}

#else /* !HAVE_OPENCL */

bool gpu_init (void) { return false; }
void gpu_shutdown (void) { }
bool gpu_is_available (void) { return false; }
bool gpu_supports_fp64 (void) { return false; }
bool gpu_should_use (i32 w, i32 h, i32 ss, u64 iter_max, f64 width,
                     enum formula formula, bool vibe)
{
  (void) w;
  (void) h;
  (void) ss;
  (void) iter_max;
  (void) width;
  (void) formula;
  return false;
}
void gpu_render_mandelbrot (const struct gpu_view *v, struct image *out)
{
  (void) v;
  (void) out;
}
const char *gpu_device_name (void) { return NULL; }

#endif