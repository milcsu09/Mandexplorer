#include "color.h"


u32
color_rgb (u8 r, u8 g, u8 b)
{
  return (0xFF << 24) | (r << 16) | (g << 8) | b;
}


u32
color_argb (u8 a, u8 r, u8 g, u8 b)
{
  return (a << 24) | (r << 16) | (g << 8) | b;
}


u8
color_a (u32 color)
{
  return (color >> 24) & 0xFF;
}


u8
color_r (u32 color)
{
  return (color >> 16) & 0xFF;
}


u8
color_g (u32 color)
{
  return (color >> 8) & 0xFF;
}


u8
color_b (u32 color)
{
  return color & 0xFF;
}


u32
color_lerp (u32 c1, u32 c2, f64 f)
{
  const u8 a1 = color_a (c1);
  const u8 r1 = color_r (c1);
  const u8 g1 = color_g (c1);
  const u8 b1 = color_b (c1);

  const u8 a2 = color_a (c2);
  const u8 r2 = color_r (c2);
  const u8 g2 = color_g (c2);
  const u8 b2 = color_b (c2);

  const u8 a = a1 + (a2 - a1) * f;
  const u8 r = r1 + (r2 - r1) * f;
  const u8 g = g1 + (g2 - g1) * f;
  const u8 b = b1 + (b2 - b1) * f;

  return color_argb (a, r, g, b);
}


u32
color_lerp_bilinear (u32 c00, u32 c10, u32 c01, u32 c11, f64 dx, f64 dy)
{
  u32 a = color_lerp (c00, c10, dx);
  u32 b = color_lerp (c01, c11, dx);

  return color_lerp (a, b, dy);
  /*
  i32 tx = dx * 256.0L;
  i32 ty = dy * 256.0L;

  u32 rb00 = (c00     ) & 0x00FF00FF;
  u32 rb10 = (c10     ) & 0x00FF00FF;
  u32 rb01 = (c01     ) & 0x00FF00FF;
  u32 rb11 = (c11     ) & 0x00FF00FF;

  u32 ag00 = (c00 >> 8) & 0x00FF00FF;
  u32 ag10 = (c10 >> 8) & 0x00FF00FF;
  u32 ag01 = (c01 >> 8) & 0x00FF00FF;
  u32 ag11 = (c11 >> 8) & 0x00FF00FF;

  u32 w00 = ((256 - tx) * (256 - ty)) >> 8;
  u32 w10 = ((      tx) * (256 - ty)) >> 8;
  u32 w01 = ((256 - tx) * (      ty)) >> 8;
  u32 w11 = ((      tx) * (      ty)) >> 8;

  u64 rb = (u64)rb00 * w00
         + (u64)rb10 * w10
         + (u64)rb01 * w01
         + (u64)rb11 * w11;

  u64 ag = (u64)ag00 * w00
         + (u64)ag10 * w10
         + (u64)ag01 * w01
         + (u64)ag11 * w11;

  rb = (rb >> 8) & 0x00FF00FF;
  ag = (ag >> 8) & 0x00FF00FF;

  return (ag << 8) | rb;
  */
}

