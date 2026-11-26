# Build and run Goose

## Build compiler
```bash
git clone https://github.com/aardappel/goose
cd goose
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGOOSE_BUILTIN_LIBS=all
cmake --build build --config Release
```
- Needs CMake 3.27+, C++20 compiler, Python 3.
- `-DGOOSE_BUILTIN_LIBS=all` fetches gfx/audio/ui/physics.
- SDL dev packages on Linux.

## Run/build
```bash
build/goose samples/01_tour.goose          # JIT in-process
build/goose run file.goose
build/goose build file.goose && ./file     # AOT executable
build/goose check file.goose               # typecheck only
build/goose -o out.c file.goose           # emit C
```
- Threads require AOT (`goose build`), not JIT.
- `--debug`, `--bce-lines`, `--stack-reserve`.
- `goose toolchain` shows C compiler.

## Packages
- `import name from "source";` source dir or git repo@tag.
- Sidecar `foo.build` beside `foo.goose`: `c_sources = ["impl.c"];`.
- Program's own C can be added `--include header.h`.

## gfx linking
- `goose --gfx-link msvc|cc` prints response file for linking gfx.

## Tests
```bash
python test/run_tests.py
python samples/run_samples.py
```
EOF