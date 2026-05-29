#!/usr/bin/env python3
"""
External reference for the backpropagation simulation.

Reads the SAME initial weights produced by the TypeScript framework
(out/backprop-framework.json), retrains a 2-3-1 sigmoid network on XOR with
identical sample order, learning rate, and naive nested-loop matrix
multiplications, then dumps final weights, loss history, and predictions to
out/external/backpropagation/numpy.json.

The TypeScript framework and this script share `init`, `lr`, sample order, and
operation order down to the loop nesting -- so trained weights should agree
within a few ULPs even though the framework runs the math through a network
of stations and this script runs it as straight nested loops.

Usage (called by external-references/run-all.sh):
    python3 bp.py
"""
import json
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IN_PATH = ROOT / "out" / "backprop-framework.json"
OUT_DIR = ROOT / "out" / "external" / "backpropagation"
OUT_PATH = OUT_DIR / "numpy.json"

XOR = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def matvec_naive(W, x, b):
    """W·x + b with naive nested loops (same float-summation order as the TS sim)."""
    out = []
    for i in range(len(W)):
        zi = b[i]
        for j in range(len(W[i])):
            zi += W[i][j] * x[j]
        out.append(zi)
    return out


def main() -> int:
    if not IN_PATH.exists():
        print(f"[bp] input {IN_PATH} not found; "
              f"run `node dist/des/main-backpropagation.js` first.", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open() as f:
        payload = json.load(f)

    cfg = payload["config"]
    init = payload["init"]
    lr = cfg["lr"]
    N = cfg["N"]

    W1 = [list(row) for row in init["W1"]]
    b1 = list(init["b1"])
    W2 = [list(row) for row in init["W2"]]
    b2 = list(init["b2"])

    losses = []
    for k in range(N):
        x, y = XOR[k % 4]

        # forward
        z1 = matvec_naive(W1, x, b1)
        a1 = [sigmoid(z) for z in z1]
        z2 = matvec_naive(W2, a1, b2)
        a2 = [sigmoid(z) for z in z2]

        # MSE loss
        loss = 0.0
        grad_a2 = [0.0] * len(a2)
        for i in range(len(a2)):
            e = a2[i] - y[i]
            loss += 0.5 * e * e
            grad_a2[i] = e
        losses.append(loss)

        # backward L2
        dz2 = [grad_a2[i] * a2[i] * (1.0 - a2[i]) for i in range(len(a2))]
        # grad_a1 = W2^T · dz2  (compute BEFORE we mutate W2).
        grad_a1 = [0.0] * len(a1)
        for i in range(len(W2)):
            for j in range(len(W2[i])):
                grad_a1[j] += W2[i][j] * dz2[i]
        # update W2, b2
        for i in range(len(W2)):
            for j in range(len(W2[i])):
                W2[i][j] -= lr * dz2[i] * a1[j]
            b2[i] -= lr * dz2[i]

        # backward L1
        dz1 = [grad_a1[i] * a1[i] * (1.0 - a1[i]) for i in range(len(a1))]
        for i in range(len(W1)):
            for j in range(len(W1[i])):
                W1[i][j] -= lr * dz1[i] * x[j]
            b1[i] -= lr * dz1[i]

    # Final predictions on the 4 XOR cases.
    preds = []
    for x, _ in XOR:
        a1 = [sigmoid(z) for z in matvec_naive(W1, x, b1)]
        a2 = [sigmoid(z) for z in matvec_naive(W2, a1, b2)]
        preds.append(a2[0])

    out = {
        "tool": "naive-nested-loop python (float64)",
        "config": cfg,
        "final": {"W1": W1, "b1": b1, "W2": W2, "b2": b2},
        "predictions": preds,
        "lossHistory": losses,
    }
    with OUT_PATH.open("w") as f:
        json.dump(out, f)
    last100 = losses[-100:]
    print(f"[bp] wrote {OUT_PATH}  (avg loss over last 100 = {sum(last100)/len(last100):.3e})")
    print(f"[bp] predictions: {[round(p, 4) for p in preds]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
