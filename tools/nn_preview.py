#!/usr/bin/env python3
"""Fast neural preview upscaler (FSRCNN x2) for Mandexplorer frames.

Reads a binary P6 PPM, upscales it with OpenCV's dnn_superres and writes
another P6 PPM.  This is a *preview* tool: it reconstructs detail that was
never rendered, so it must never be used for final output.

Usage:
    python tools/nn_preview.py in.ppm out.ppm [--model models/FSRCNN_x2.pb]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
from cv2 import dnn_superres


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model", type=Path,
                   default=Path(__file__).resolve().parent.parent / "models" / "FSRCNN_x2.onnx")
    return p.parse_args()


def main():
    args = parse_args()

    img = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"cannot read {args.input}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(args.model))
    sr.setModel("fsrcnn", 2)
    out = sr.upsample(img)
    dt = (time.perf_counter() - t0) * 1000.0

    if not cv2.imwrite(str(args.output), out):
        print(f"cannot write {args.output}", file=sys.stderr)
        return 1

    print(f"{args.input}: {img.shape[1]}x{img.shape[0]} -> "
          f"{out.shape[1]}x{out.shape[0]}, {dt:.1f} ms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
