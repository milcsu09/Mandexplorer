#include "pa.h"

static struct pa_z
pa_norm (f64 x, f64 y, i32 e)
{
  const f64 m = fmax (fabs (x), fabs (y));

  if (m == 0.0 || !isfinite (m))
    return (struct pa_z) { 0.0, 0.0, 0 };

  i32 shift;
  frexp (m, &shift);

  const f64 sc = ldexp (1.0, -shift);
  return (struct pa_z) { x * sc, y * sc, e + shift };
}


static struct pa_z
pa_mul_complex (struct pa_z a, f64 x, f64 y)
{
  return pa_norm (a.x * x - a.y * y, a.x * y + a.y * x, a.e);
}


static struct pa_z
pa_add (struct pa_z a, struct pa_z b)
{
  if (a.x == 0.0 && a.y == 0.0)
    return b;
  if (b.x == 0.0 && b.y == 0.0)
    return a;

  const i32 d = a.e - b.e;

  if (d > 80)
    return a;
  if (d < -80)
    return b;

  if (d >= 0)
    {
      const f64 sc = ldexp (1.0, -d);
      return pa_norm (a.x + b.x * sc, a.y + b.y * sc, a.e);
    }

  const f64 sc = ldexp (1.0, d);
  return pa_norm (a.x * sc + b.x, a.y * sc + b.y, b.e);
}


static struct pa_z
pa_add_one (struct pa_z a)
{
  const struct pa_z one = { 1.0, 0.0, 0 };
  return pa_add (a, one);
}


static f64
pa_log2 (struct pa_z v)
{
  const f64 m = v.x * v.x + v.y * v.y;
  return m == 0.0 ? -INFINITY : (f64) v.e + 0.5 * log2 (m);
}


static void
pa_apply (struct pa_z a, f64 x, f64 y, f64 *out_re, f64 *out_im)
{
  const struct pa_z r = pa_mul_complex (a, x, y);
  *out_re = ldexp (r.x, r.e);
  *out_im = ldexp (r.y, r.e);
}


void
pa_init (struct pa_table *t)
{
  memset (t, 0, sizeof *t);
}


void
pa_clear (struct pa_table *t)
{
  free (t->entries);
  memset (t, 0, sizeof *t);
}


bool
pa_build (struct pa_table *t, struct orbit *o, f64 dc_max, f64 epsilon)
{
  pa_clear (t);

  const u64 n = o->n;

  if (!(epsilon > 0.0) || n < (u64) PA_ALIGN * 2)
    return false;

  // RFF never looks up PA at refIteration 0.  We build a bounded lookahead PA
  // for EVERY BLA-aligned start, so a BLA skip always lands on a usable entry;
  // the finite horizon keeps build cost O(N) and the radius guard keeps it
  // safe.  Period candidates are still used elsewhere (stage 4 MPA).
  usize count = (usize) (n / PA_ALIGN);

  if (count > PA_MAX_ENTRIES)
    count = PA_MAX_ENTRIES;

  t->entries = calloc (count, sizeof *t->entries);

  if (!t->entries)
    return false;

  t->count = count;

  const f64 dc_log2 = dc_max > 0.0 ? log2 (dc_max) : -INFINITY;
  const f64 eps_log2 = log2 (epsilon);
  const u64 aligned_n = n - n % PA_ALIGN;

  // idx 0 would start at iteration 0, which RFF explicitly rejects.
  for (usize idx = 1; idx < count; ++idx)
    {
      const u64 start = (u64) idx * PA_ALIGN;
      u64 end = start + PA_SKIP_CAP;

      if (end > aligned_n)
        end = aligned_n;

      struct pa_entry *q = &t->entries[idx];

      struct pa_z an = { 1.0, 0.0, 0 };
      struct pa_z bn = { 0.0, 0.0, 0 };
      struct pa_z best_an = an;
      struct pa_z best_bn = bn;
      f64 radius = INFINITY;
      f64 best_radius = 0.0;
      u64 best_len = 0;

      for (u64 j = start; j < end; ++j)
        {
          f64 rr, ri;

          if (!orbit_at (o, j, &rr, &ri))
            {
              pa_clear (t);
              return false;
            }

          const f64 z2r = 2.0 * rr;
          const f64 z2i = 2.0 * ri;

          if (z2r != 0.0 || z2i != 0.0)
            {
              const f64 b = pa_log2 (bn) + dc_log2;
              const f64 a = eps_log2 + log2 (hypot (z2r, z2i));
              const f64 an_log2 = pa_log2 (an);

              if (b >= a || an_log2 == -INFINITY)
                radius = 0.0;
              else
                {
                  const f64 corr =
                    a - b > 80.0 ? 0.0 : log2 (1.0 - exp2 (b - a));
                  radius = fmin (radius, a + corr - an_log2);
                }
            }

          // The guard only tightens, so the first failure ends this entry.
          if (radius == 0.0)
            break;

          an = pa_mul_complex (an, z2r, z2i);
          bn = pa_add_one (pa_mul_complex (bn, z2r, z2i));

          // Only keep snapshots on the BLA grid, so a skip never breaks BLA
          // alignment and the stored coefficients match the stored length.
          if ((j - start + 1) % PA_ALIGN == 0)
            {
              best_len = j - start + 1;
              best_radius = radius;
              best_an = an;
              best_bn = bn;
            }
        }

      q->start = start;
      q->skip = best_len;
      q->an = best_an;
      q->bn = best_bn;
      q->radius = best_len ? best_radius : 0.0;
    }

  t->valid = true;
  return true;
}


bool
pa_try_skip (const struct pa_table *t, u64 iter_orbit,
             f64 eps_re, f64 eps_im, f64 dc_re, f64 dc_im,
             f64 deriv_re, f64 deriv_im,
             f64 *new_eps_re, f64 *new_eps_im,
             f64 *new_deriv_re, f64 *new_deriv_im,
             u64 *skip)
{
  if (!t || !t->valid || t->count == 0 || iter_orbit < PA_ALIGN
      || iter_orbit % PA_ALIGN != 0)
    return false;

  const usize idx = (usize) (iter_orbit / PA_ALIGN);

  if (idx >= t->count)
    return false;

  const struct pa_entry *q = &t->entries[idx];

  const f64 emag = hypot (eps_re, eps_im);
  const f64 elog = emag > 0.0 ? log2 (emag) : -INFINITY;

  if (q->start != iter_orbit || q->radius == 0.0 || !(elog < q->radius))
    return false;

  *skip = q->skip;

  pa_apply (q->an, eps_re, eps_im, new_eps_re, new_eps_im);
  pa_apply (q->an, deriv_re, deriv_im, new_deriv_re, new_deriv_im);

  // eps' = an*eps + bn*dc, D' = an*D + bn (bn is the derivative
  // contribution of an eps of 1).
  f64 er, ei, br, bi;
  pa_apply (q->bn, dc_re, dc_im, &er, &ei);
  pa_apply (q->bn, 1.0, 0.0, &br, &bi);
  *new_eps_re += er;
  *new_eps_im += ei;
  *new_deriv_re += br;
  *new_deriv_im += bi;

  return true;
}


// ---------------------------------------------------------------------------
// Stage 4: multi-periodic approximation (MPA).
//
// Port of RFF's MB2MPAGenerator / MPAInfo / MPAView (uncompressed path).  All
// state uses the same extended-exponent complex as the naive PA above, so the
// per-step arithmetic is identical and the renderer can mix both.

static struct pa_z
pa_mul (struct pa_z a, struct pa_z b)
{
  struct pa_z r = pa_mul_complex (a, b.x, b.y);
  r.e += b.e;
  return r;
}


// One PAGenerator: the first-order map eps_{start+skip} = an*eps_start + bn*dc
// accumulated over [start, start+skip), plus the log2 radius guard.
struct pa_gen
{
  u64 start;
  u64 skip;
  struct pa_z an;
  struct pa_z bn;
  f64 radius;
};


static void
pa_gen_reuse (struct pa_gen *g, u64 start)
{
  g->start = start;
  g->skip = 0;
  g->an = (struct pa_z) { 1.0, 0.0, 0 };
  g->bn = (struct pa_z) { 0.0, 0.0, 0 };
  g->radius = INFINITY;
}


static void
pa_gen_step (struct pa_gen *g, struct orbit *o, f64 dc_max, f64 epsilon)
{
  const u64 iter = g->start + g->skip;

  f64 rr, ri;

  if (!orbit_at (o, iter, &rr, &ri))
    {
      g->skip++;
      g->radius = 0.0;
      return;
    }

  g->skip++;

  if (g->radius == -INFINITY)
    return;

  const f64 z2r = 2.0 * rr;
  const f64 z2i = 2.0 * ri;

  if (z2r != 0.0 || z2i != 0.0)
    {
      const f64 dc_log2 = dc_max > 0.0 ? log2 (dc_max) : -INFINITY;
      const f64 b = pa_log2 (g->bn) + dc_log2;
      const f64 a = log2 (epsilon) + log2 (hypot (z2r, z2i));
      const f64 an_log2 = pa_log2 (g->an);
      f64 r;

      // Log2 of the linear guard radius.  Negative values are valid (a tiny
      // linear radius); only an empty/negative linear radius is invalid and
      // maps to -INFINITY.
      if (b >= a || an_log2 == -INFINITY)
        r = -INFINITY;
      else
        {
          const f64 corr = a - b > 80.0 ? 0.0 : log2 (1.0 - exp2 (b - a));
          r = a + corr - an_log2;
        }

      if (r < g->radius)
        g->radius = r;
    }

  g->an = pa_mul_complex (g->an, z2r, z2i);
  g->bn = pa_add_one (pa_mul_complex (g->bn, z2r, z2i));
}


// g <- g after target: RFF PAGenerator::merge.
static void
pa_gen_merge (struct pa_gen *g, const struct pa_gen *tgt, f64 dc_max)
{
  g->skip += tgt->skip;

  if (g->radius == -INFINITY || tgt->radius == -INFINITY)
    {
      g->radius = -INFINITY;
      return;
    }

  const f64 dc_log2 = dc_max > 0.0 ? log2 (dc_max) : -INFINITY;
  const f64 b = pa_log2 (g->bn) + dc_log2;
  const f64 an_log2 = pa_log2 (g->an);
  f64 rad;

  if (b >= tgt->radius)
    rad = -INFINITY;
  else if (an_log2 == -INFINITY)
    rad = g->radius;
  else
    {
      const f64 corr = tgt->radius - b > 80.0
                         ? 0.0
                         : log2 (1.0 - exp2 (b - tgt->radius));
      rad = tgt->radius + corr - an_log2;
    }

  if (rad > g->radius)
    rad = g->radius;

  g->radius = rad;

  g->an = pa_mul (tgt->an, g->an);
  g->bn = pa_add (pa_mul (tgt->an, g->bn), tgt->bn);
}


void
mpa_init (struct mpa_table *t)
{
  memset (t, 0, sizeof *t);
}


void
mpa_clear (struct mpa_table *t)
{
  free (t->periods);
  free (t->artificial);
  free (t->skippable);
  free (t->elements);
  free (t->table);
  memset (t, 0, sizeof *t);
}


usize
mpa_level_count (const struct mpa_table *t)
{
  return t ? t->levels : 0;
}


// RFF MPAIndexMapperUtils::iterationToFlattenTableIndexMapper: the inverse of
// the mixed-radix table layout.  Returns false when `iteration` is not on a
// boundary where a skip may start.
static bool
mpa_mapper (const struct mpa_table *t, u64 iteration, usize *mapped,
            usize *levels)
{
  if (iteration == 0)
    return false;

  if (iteration == 1)
    {
      *mapped = 0;
      *levels = t->levels;
      return true;
    }

  u64 index = 0;
  usize lev = 0;
  u64 remainder = iteration;
  u64 max_skip = UINT64_MAX;

  for (usize i = t->levels; i > 0; --i)
    {
      const u64 period = t->periods[i - 1];
      const u64 quotient = remainder / period;
      const u64 sub = quotient * period;

      remainder -= sub;
      max_skip = sub >= max_skip
                   ? 0
                   : (period < max_skip - sub ? period : max_skip - sub);

      if (remainder == 0 || max_skip < t->periods[0])
        return false;

      lev += (max_skip == period && remainder == 1);

      index += quotient * t->elements[i - 1] + (max_skip == period);
    }

  index -= lev;

  if (remainder != 1 || max_skip < t->periods[0])
    return false;

  *mapped = (usize) index;
  *levels = lev;
  return true;
}


bool
mpa_build (struct mpa_table *t, struct orbit *o, f64 dc_max, f64 epsilon)
{
  mpa_clear (t);

  if (!(epsilon > 0.0) || o == NULL || o->period_count < 2)
    return false;

  const usize np = o->period_count;
  const u64 longest = o->periods[np - 1].iter;

  if (longest <= MPA_MIN_SKIP)
    return false;

  const usize cap = np + MPA_MAX_LEVELS + 8;

  t->periods = malloc (cap * sizeof (u64));

  if (!t->periods)
    {
      mpa_clear (t);
      return false;
    }

  // RFF MPAInfo::generateTablePeriod: start from minSkip and walk the record
  // minima, inserting artificial periods so consecutive levels differ by at
  // most MPA_MAX_MULT^2.
  usize L = 0;
  u64 current = MPA_MIN_SKIP;

  t->periods[L++] = current;

  for (usize i = 0; i < np; ++i)
    {
      const u64 p = o->periods[i].iter;

      if (p >= MPA_MIN_SKIP
          && ((p == longest && current != longest)
              || current * MPA_MAX_MULT <= p))
        {
          while (current >= MPA_MIN_SKIP
                 && current * MPA_MAX_MULT * MPA_MAX_MULT < p)
            {
              if (L >= MPA_MAX_LEVELS)
                break;

              t->periods[L++] = current * MPA_MAX_MULT;
              current *= MPA_MAX_MULT;
            }

          if (L >= MPA_MAX_LEVELS)
            break;

          t->periods[L++] = p;
          current = p;
        }
    }

  t->levels = L;

  if (L < 2 || t->periods[L - 1] != longest)
    {
      mpa_clear (t);
      return false;
    }

  t->skippable = calloc (L, sizeof (u64));
  t->elements = calloc (L, sizeof (u64));

  if (!t->skippable || !t->elements)
    {
      mpa_clear (t);
      return false;
    }

  // RFF MPAInfo::generateCountInfos.
  t->skippable[0] = 1;
  t->elements[0] = 1;

  for (usize i = 1; i < L; ++i)
    {
      u64 elem = 0;
      u64 skip = 0;
      u64 rem = t->periods[i];

      for (usize j = i; j > 0; --j)
        {
          const u64 group = rem / t->periods[j - 1];

          rem %= t->periods[j - 1];
          elem += group * t->elements[j - 1];
          skip += group * t->skippable[j - 1];
        }

      t->elements[i] = elem + 1;
      t->skippable[i] = skip;
    }

  t->table_len = (usize) t->elements[L - 1];

  if (t->table_len == 0 || t->table_len > MPA_MAX_TABLE)
    {
      mpa_clear (t);
      return false;
    }

  t->table = calloc (t->table_len, sizeof *t->table);

  if (!t->table)
    {
      mpa_clear (t);
      return false;
    }

  for (usize i = 0; i < t->table_len; ++i)
    t->table[i].radius = -INFINITY;

  t->epsilon = epsilon;

  u64 *it_count = calloc (L, sizeof (u64));
  u64 *it_lim = calloc (L, sizeof (u64));
  bool *avail = calloc (L, sizeof (bool));
  struct pa_gen *cur = calloc (L, sizeof *cur);

  if (!it_count || !it_lim || !avail || !cur)
    {
      free (it_count);
      free (it_lim);
      free (avail);
      free (cur);
      mpa_clear (t);
      return false;
    }

  // RFF uInitProcessor(1, epsilon).
  for (usize i = L; i > 0; --i)
    {
      const usize lv = i - 1;

      it_lim[lv] = t->periods[lv];
      it_count[lv] = 0;
      avail[lv] = it_count[lv] < t->periods[lv] - MPA_PERTURB_REQ;
      pa_gen_reuse (&cur[lv], 1);
    }

  u64 iteration = 1;

  // Optional layout check: RFF records the mapper in a running array, while we
  // recompute it from the mixed-radix formula.  Both must agree.
  const bool mapcheck = getenv ("MANDEX_MPA_MAPCHECK") != NULL;
  struct run_map { usize mapped; usize gen; } *run = NULL;
  u64 flat_index = 0;
  usize map_bad_index = 0;
  usize map_bad_levels = 0;
  usize store_bad = 0;

  if (mapcheck)
    {
      run = calloc ((usize) longest + 1, sizeof *run);

      if (!run)
        {
          free (it_count);
          free (it_lim);
          free (avail);
          free (cur);
          mpa_clear (t);
          return false;
        }
    }

  // RFF uGenerateTable, single-threaded.
  while (iteration <= longest)
    {
      // uStepOnce: build the source of the level-0 PA for this iteration.
      usize levels_count = 0;

      while (levels_count < L
             && it_count[levels_count] == 0
             && it_lim[levels_count] == t->periods[levels_count])
        ++levels_count;

      if (mapcheck && (usize) iteration <= (usize) longest)
        {
          usize im, ig;

          run[iteration].mapped = flat_index;
          run[iteration].gen = levels_count;

          if (levels_count > 0)
            flat_index += levels_count;

          if (mpa_mapper (t, iteration, &im, &ig))
            {
              if (im != run[iteration].mapped)
                ++map_bad_index;
              if (ig != levels_count)
                ++map_bad_levels;
            }
          else if (levels_count > 0)
            ++map_bad_index;
        }

      pa_gen_step (&cur[0], o, dc_max, epsilon);
      it_count[0]++;
      iteration++;

      // uVerifyPA: flush every level that has completed its period.
      usize level = 0;

      while (level < L
             && (cur[level].skip == t->periods[level] - MPA_PERTURB_REQ
                 || it_lim[level] != t->periods[level]
                 || !avail[level]))
        {
          if (it_lim[level] == t->periods[level] && avail[level])
            {
              usize mapped, gen;

              if (mpa_mapper (t, cur[level].start, &mapped, &gen)
                  && (usize) mapped + level < t->table_len)
                {
                  if (mapcheck && cur[level].start <= longest
                      && (run[cur[level].start].mapped != mapped
                          || run[cur[level].start].gen <= level))
                    ++store_bad;

                  struct mpa_entry *e = &t->table[mapped + level];

                  e->skip = cur[level].skip;
                  e->an = cur[level].an;
                  e->bn = cur[level].bn;
                  e->radius = cur[level].radius;
                }

              avail[level] = false;
            }

          if (level < L - 1)
            pa_gen_merge (&cur[level + 1], &cur[level], dc_max);

          pa_gen_reuse (&cur[level], iteration);
          ++level;
        }

      // uRefreshCounter: propagate completed counters upwards.
      level = 0;

      while (level < L - 1 && it_count[level] == it_lim[level])
        {
          it_count[level + 1] += it_count[level];
          pa_gen_merge (&cur[level + 1], &cur[level], dc_max);
          pa_gen_reuse (&cur[level], iteration);
          ++level;
        }

      while (level > 0)
        {
          --level;

          const u64 nxt = it_lim[level + 1] - it_count[level + 1];

          it_lim[level] = t->periods[level] < nxt ? t->periods[level] : nxt;
          it_count[level] = 0;
          avail[level] = it_lim[level] == t->periods[level];
        }
    }

  if (mapcheck)
    {
      fprintf (stderr,
               "MPA mapcheck: bad index %zu, bad levels %zu, bad store %zu"
               " (final flat %llu)\n",
               map_bad_index, map_bad_levels, store_bad,
               (unsigned long long) flat_index);
      free (run);
    }

  free (it_count);
  free (it_lim);
  free (avail);
  free (cur);

  t->valid = true;
  return true;
}


bool
mpa_try_skip (const struct mpa_table *t, u64 ref_iter,
              f64 eps_re, f64 eps_im, f64 deriv_re, f64 deriv_im,
              f64 dc_re, f64 dc_im,
              f64 *new_eps_re, f64 *new_eps_im,
              f64 *new_deriv_re, f64 *new_deriv_im,
              u64 *skip)
{
  if (!t || !t->valid || t->levels == 0 || t->table_len == 0)
    return false;

  usize mapped, gen;

  if (!mpa_mapper (t, ref_iter, &mapped, &gen) || mapped >= t->table_len)
    return false;

  usize n = gen;

  if (mapped + n > t->table_len)
    n = t->table_len - mapped;

  if (n == 0 || t->table[mapped].skip == 0
      || t->table[mapped].radius == -INFINITY)
    return false;

  const f64 emag = hypot (eps_re, eps_im);
  const f64 elog = emag > 0.0 ? log2 (emag) : -INFINITY;

  if (!(elog < t->table[mapped].radius))
    return false;

  const struct mpa_entry *best = &t->table[mapped];

  for (usize j = n; j > 1; --j)
    if (t->table[mapped + j - 1].radius != -INFINITY
        && elog < t->table[mapped + j - 1].radius)
      {
        best = &t->table[mapped + j - 1];
        break;
      }

  *skip = best->skip;

  pa_apply (best->an, eps_re, eps_im, new_eps_re, new_eps_im);
  pa_apply (best->an, deriv_re, deriv_im, new_deriv_re, new_deriv_im);

  f64 er, ei, br, bi;

  pa_apply (best->bn, dc_re, dc_im, &er, &ei);
  pa_apply (best->bn, 1.0, 0.0, &br, &bi);

  *new_eps_re += er;
  *new_eps_im += ei;
  *new_deriv_re += br;
  *new_deriv_im += bi;

  return true;
}


static void
mpa_exact_step (struct orbit *o, u64 j, f64 *er, f64 *ei, f64 dr, f64 di)
{
  f64 rr, ri;

  if (!orbit_at (o, j, &rr, &ri))
    {
      *er = *ei = NAN;
      return;
    }

  // eps_{n+1} = 2*Z_n*eps_n + eps_n^2 + dc  (the linear term uses the
  // reference Z_n, not z_n = Z_n + eps_n).
  const f64 lin_r = 2.0 * (rr * *er - ri * *ei);
  const f64 lin_i = 2.0 * (rr * *ei + ri * *er);
  const f64 q_r = *er * *er - *ei * *ei;
  const f64 q_i = 2.0 * *er * *ei;

  *er = lin_r + q_r + dr;
  *ei = lin_i + q_i + di;
}


void
mpa_selftest (const struct mpa_table *t, struct orbit *o, f64 dc)
{
  if (!t || !t->valid)
    {
      fprintf (stderr, "MPA selftest: no table\n");
      return;
    }

  const f64 dr = dc * 0.1;
  const f64 di = dc * 0.05;

  usize tested = 0;
  usize bad = 0;
  f64 worst = -INFINITY;
  u64 worst_start = 0;
  u64 worst_skip = 0;

  const u64 scan = t->periods[t->levels - 1] < 200000
                     ? t->periods[t->levels - 1] : 200000;

  for (u64 start = 1; start <= scan; ++start)
    {
      usize mapped, gen;

      if (!mpa_mapper (t, start, &mapped, &gen))
        continue;

      if ((usize) mapped + gen > t->table_len)
        continue;

      for (usize lv = 0; lv < gen; ++lv)
        {
          const struct mpa_entry *e = &t->table[mapped + lv];

          if (e->skip == 0 || e->skip > 200000 || e->radius == -INFINITY)
            continue;

          f64 er = 0.0, ei = 0.0;

          for (u64 j = start; j < start + e->skip; ++j)
            mpa_exact_step (o, j, &er, &ei, dr, di);

          f64 pr, pi, qr, qi;
          u64 skip = 0;

          if (!mpa_try_skip (t, start, 0.0, 0.0, 0.0, 0.0,
                             dr, di, &pr, &pi, &qr, &qi, &skip)
              || skip != e->skip)
            continue;

          const f64 d = hypot (er - pr, ei - pi) / (hypot (er, ei) + 1e-300);

          ++tested;

          if (!(d < 1e-4))
            {
              ++bad;

              if (bad <= 5)
                fprintf (stderr,
                         "  bad start=%llu lv=%zu skip=%llu exact=(%.6e,%.6e)"
                         " pa=(%.6e,%.6e) an=(%.3e,%.3e)e%d bn=(%.3e,%.3e)e%d"
                         " rad=%.1f\n",
                         (unsigned long long) start, lv,
                         (unsigned long long) e->skip, er, ei, pr, pi,
                         e->an.x, e->an.y, e->an.e,
                         e->bn.x, e->bn.y, e->bn.e, e->radius);
            }

          if (d > worst)
            {
              worst = d;
              worst_start = start;
              worst_skip = e->skip;
            }
        }
    }

  fprintf (stderr,
           "MPA selftest: %zu entries, %zu bad, worst rel %.3e"
           " (start %llu skip %llu)\n",
           tested, bad, worst, (unsigned long long) worst_start,
           (unsigned long long) worst_skip);
}
