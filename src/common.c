#include "common.h"


void *
malloc0 (usize size)
{
  return calloc (1, size);
}


i64
imin (i64 a, i64 b)
{
  if (a < b)
    return a;
  else
    return b;
}


i64
imax (i64 a, i64 b)
{
  if (a > b)
    return a;
  else
    return b;
}

