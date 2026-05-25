#ifndef PALETTE_H
#define PALETTE_H


#include "common.h"


#define PALETTE_FILE_VERSION 1


struct palette
{
  usize size;

  u32 data[];
};


struct palette *
palette_create (usize size);

struct palette *
palette_create_from (usize size, u32 *data);

void
palette_destroy (struct palette *palette);

u32
palette_get (struct palette *palette, usize i);

i32
palette_save (const char *path, struct palette *palette);

struct palette *
palette_load (const char *path);


#endif // PALETTE_H

