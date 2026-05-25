#ifndef IMAGE_H
#define IMAGE_H


#include "common.h"


struct image
{
  i32 w;
  i32 h;

  _Atomic u32 pixels[];
};


struct image *
image_create (i32 w, i32 h);

void
image_destroy (struct image *image);

void
image_set (struct image *image, i32 i, u32 color);

void
image_set2 (struct image *image, i32 x, i32 y, u32 color);

u32
image_get (const struct image *image, i32 i);

u32
image_get2 (const struct image *image, i32 x, i32 y);

void
image_fill (struct image *image, u32 color);

// NOTE: A and B must have same W and H.
void
image_copy (struct image *a, struct image *b);


#endif // IMAGE_H

