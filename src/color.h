#ifndef COLOR_H
#define COLOR_H


#include "common.h"


u32
color_rgb (u8 r, u8 g, u8 b);

u32
color_argb (u8 a, u8 r, u8 g, u8 b);

u8
color_a (u32 color);

u8
color_r (u32 color);

u8
color_g (u32 color);

u8
color_b (u32 color);

u32
color_lerp (u32 a, u32 b, f64 f);

u32
color_lerp_bilinear (u32 c00, u32 c10, u32 c01, u32 c11, f64 dx, f64 dy);


#endif // COLOR_H

