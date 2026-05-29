#!/usr/bin/env python3
"""
External reference for the 1-D convolution simulation.

Reads `out/convolution-framework.json` (produced by `dist/des/main-convolution.js`)
and writes `out/external/convolution/numpy.json` containing the numpy
ground-truth output for the same signal + kernel. The TypeScript validation
driver (`dist/des/runners/validate-convolution.js`) loads both and computes
max-abs-error and RMSE.

Usage (called by external-references/run-all.sh):
    python3 conv.py
"""
import json
import os
import sys
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IN_PATH = ROOT / "out" / "convolution-framework.json"
OUT_DIR = ROOT / "out" / "external" / "convolution"
OUT_PATH = OUT_DIR / "numpy.json"


def main() -> int:
    if not IN_PATH.exists():
        print(f"[conv] input {IN_PATH} not found; "
              f"run `node dist/des/main-convolution.js` first.", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open() as f:
        payload = json.load(f)
    x = np.asarray(payload["signal"], dtype=np.float64)
    h = np.asarray(payload["kernel"], dtype=np.float64)

    y_full = np.convolve(x, h, mode="full")

    out = {
        "tool": "numpy.convolve",
        "mode": "full",
        "y": y_full.tolist(),
        "expected_length": int(len(x) + len(h) - 1),
    }
    with OUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[conv] wrote {OUT_PATH}  (len={len(y_full)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
