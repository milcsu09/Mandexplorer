#include "color.h"
#include "palette.h"


struct palette *
palette_create (usize size)
{
  struct palette *palette = malloc0 (sizeof (struct palette) + size * sizeof (u32));

  palette->size = size;

  return palette;
}


struct palette *
palette_create_from (usize size, u32 *data)
{
  struct palette *palette = malloc0 (sizeof (struct palette) + size * sizeof (u32));

  palette->size = size;

  memcpy (palette->data, data, size * sizeof (u32));

  return palette;
}


void
palette_destroy (struct palette *palette)
{
  free (palette);
}


u32
palette_get (struct palette *palette, usize i)
{
  return palette->data[i % palette->size];
}


i32
palette_save (const char *path, struct palette *palette)
{
  FILE *f = fopen (path, "w");

  if (!f)
    return 1;

  // VERSION
  fprintf (f, "%" PRIu32 "\n", PALETTE_FILE_VERSION);

  // SIZE
  fprintf (f, "%" PRIu64 "\n", palette->size);

  // COLOR...
  for (usize i = 0; i < palette->size; ++i)
    {
      const u32 color = palette->data[i];

      const u8 a = color_a (color);
      const u8 r = color_r (color);
      const u8 g = color_g (color);
      const u8 b = color_b (color);

      fprintf (f, "%" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32 "\n", a, r, g, b);
    }

  fclose (f);

  return 0;
}


struct palette *
palette_load (const char *path)
{
  FILE *f = fopen (path, "r");

  if (!f)
    return NULL;

  // VERSION

  u32 version;

  if (fscanf (f, "%" SCNu32 "\n", &version) != 1 || version != PALETTE_FILE_VERSION)
    {
      fclose (f);

      return NULL;
    }

  // SIZE

  usize size;

  if (fscanf (f, "%" SCNu64 "\n", &size) != 1)
    {
      fclose (f);

      return NULL;
    }

  struct palette *palette = palette_create (size);

  if (!palette)
    {
      fclose (f);

      return NULL;
    }

  // COLOR...

  for (usize i = 0; i < size; ++i)
    {
      u32 a, r, g, b;

      if (fscanf (f, "%" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32 "\n", &a, &r, &g, &b) != 4)
        {
          palette_destroy (palette);

          fclose (f);

          return NULL;
        }

      palette->data[i] = color_argb (a, r, g, b);
    }

  fclose (f);

  return palette;
}


