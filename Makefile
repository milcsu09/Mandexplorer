OPENCL_CFLAGS := $(shell PKG_CONFIG_PATH=/usr/share/pkgconfig pkg-config --cflags OpenCL 2>/dev/null)
OPENCL_LIBS := $(shell PKG_CONFIG_PATH=/usr/share/pkgconfig pkg-config --libs OpenCL 2>/dev/null)

ifneq ($(OPENCL_LIBS),)
  EXTRA_CFLAGS := -DHAVE_OPENCL=1 $(OPENCL_CFLAGS)
  EXTRA_LIBS := $(OPENCL_LIBS)
else
  EXTRA_CFLAGS := -DHAVE_OPENCL=0
  EXTRA_LIBS :=
endif

all:
	gcc -std=c11 -Wall -Wextra -Wpedantic -O3 -pthread $(EXTRA_CFLAGS) -oa $(wildcard src/*.c) -lm -lSDL2 -lSDL2_ttf -lmpfr $(EXTRA_LIBS)