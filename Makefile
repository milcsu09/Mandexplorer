
all:
	gcc -std=c11 -Wall -Wextra -Wpedantic -O3 -oa $(wildcard src/*.c) -lm -lSDL2 -lSDL2_ttf -lmpfr

