# audio, ui, physics (brief)

## audio
- Procedural mono/stereo PCM through SDL3 without graphics window.
- `import audio;` reference in stdlib.

## ui
- Fonts, text rendering, game HUDs.
- Windows/widgets on Nuklear drawn through gfx.
- Immediate-mode widgets; each frame lists windows/widgets and reads user actions.
- Text in Roboto from Nuklear submodule; scaling support.
- `ui::label`, `ui::draw_text`, `ui::render`.

## physics
- Rigid body physics on Box3D.
- Simulates up to 8 worker threads.
- Each frame retrieves all transforms in one call into instance buffer.
- Explosion, mouse view, rate adjustments.
EOF