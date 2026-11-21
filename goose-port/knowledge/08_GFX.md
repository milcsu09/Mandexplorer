# gfx module (SDL3 GPU)

## Overview
- Graphics on SDL3 GPU API (D3D12/Vulkan/Metal).
- Optional: needs compiler built with SDL submodule.
- Program links what `goose --gfx-link msvc|cc` prints.
- Namespace `gfx`.
- Window/input, buffers, textures/samplers, pipelines, render/compute passes, readback.
- Text rendering provided by `ui` (label/draw_text/render).

## Shaders
- `embed_shader(stage, ...parts)` builtin; compiles GLSL 450 to SPIR-V/MSL/HLSL at compile time; result `const u8[:]`.
- Parts can be strings/lets/consts; joined as lines.
- `#include` relative.
- No doubles, no geometry/tessellation; uniform blocks no instance names; storage buffer blocks one runtime array.

## Resource binding sets
Vertex: set0 samplers/storage textures/storage buffers (ro); set1 uniform blocks.
Fragment: set2 samplers/storage textures/storage buffers (ro); set3 uniform blocks.
Compute: set0 samplers/ro storage textures/ro storage buffers; set1 rw storage textures/rw storage buffers; set2 uniform blocks.

## Device/window/frames
- `open(title,width,height)` / `open(...,flags)` / `open_headless`.
- `close`, `frame()` (ends frame; false if close requested), `quit`, `flush`, `error`, `check`.
- `screen()` texture (RGBA8), `screen_depth()`, `screen_size()`.
- `begin_screen(clear)`, `begin_pass`, `end_pass`.
- Flags: WINDOW_RESIZABLE, WINDOW_HIDDEN, WINDOW_FULLSCREEN, NO_VSYNC, DEBUG.
- Headless via `GOOSE_GFX_HEADLESS=1`.

## Input
- `key_down/pressed/released`, `mouse_down/pressed/released`, `mouse_pos/delta`, `mouse_wheel`, `mouse_in_window`, `focused`.
- `events()` last frame input.
- `set_mouse_relative`, `set_cursor`, `set_cursor_visible`, text_input.
- `clipboard`, `scancode`, `key_name`.

## Buffers
- `buffer(usage,data)`, `vertex_buffer`, `index_buffer`, `update_buffer`, `read_buffer`, `buffer_size`, `release`.
- Usage: BUFFER_VERTEX, INDEX, INDIRECT, STORAGE, COMPUTE_READ, COMPUTE_WRITE.

## Textures/samplers
- `texture(desc)`, `texture2d`, `render_target`, `depth_target`, `texture3d`, `texture_cube`, `texture_array`, `load_texture`, `update_texture`, `read_texture`, `read_pixels`, `save_png`, `screenshot`, `generate_mips`, `release`.
- Formats: RGBA8, BGRA8, etc; RGBA32F etc.
- `sampler(desc)` / `sampler(filter,wrap)`, `shadow_sampler`, `release`.

## Pipelines/drawing
- `pipeline(vs,fs)` / with PipelineDesc.
- `begin_pass(desc)` / `begin_pass(target,depth,clear)` / `begin_screen`.
- `bind(p)`, `bind_vertex_buffer`, `bind_index_buffer`, `bind_texture`, `bind_storage_texture`, `bind_storage_buffer`, `uniforms<T>`, `push_uniforms`, `viewport`, `scissor`.
- `draw`, `draw_instanced`, `draw_indexed`, `draw_indexed_instanced`, variants with offsets.
- Pipeline not tied to formats; made for pass targets on first bind.
- Vertex inputs packed as Goose struct fields in location order.
- Uniform blocks std140; Goose structs packed -> need padding (pad fields).

## Compute
- `compute_pipeline(cs)`, `begin_compute(rw_buffers[,rw_textures])`, `bind`, `dispatch`, `end_compute`, `release`.

## Matrices
- `mat4` column-major; identity, mul, translate, scale, rotate, perspective, ortho, look_at.
EOF