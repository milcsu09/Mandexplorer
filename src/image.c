#include "color.h"
#include "image.h"


struct image *
image_create (i32 w, i32 h)
{
  struct image *image = malloc0 (sizeof (struct image) + w * h * sizeof (_Atomic u32));

  image->w = w;
  image->h = h;

  return image;
}


void
image_destroy (struct image *image)
{
  free (image);
}


void
image_set (struct image *image, i32 i, u32 color)
{
  atomic_store_explicit (&image->pixels[i], color, memory_order_relaxed);
}


void
image_set2 (struct image *image, i32 x, i32 y, u32 color)
{
  image_set (image, y * image->w + x, color);
}


u32
image_get (const struct image *image, i32 i)
{
  return atomic_load_explicit (&image->pixels[i], memory_order_relaxed);
}


u32
image_get2 (const struct image *image, i32 x, i32 y)
{
  return image_get (image, y * image->w + x);
}


void
image_fill (struct image *image, u32 color)
{
  for (i32 i = 0; i < image->w * image->h; ++i)
    image_set (image, i, color);
}


void
image_copy (struct image *a, struct image *b)
{
  for (i32 i = 0; i < b->w * b->h; ++i)
    image_set (b, i, image_get (a, i));
}

