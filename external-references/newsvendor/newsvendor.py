#!/usr/bin/env python3
"""
External reference for the newsvendor and multi-period inventory MDP.

Two solution methods, neither dependent on the TypeScript code:

  (1) Newsvendor critical-fractile q*  via scipy.stats.poisson.ppf
  (2) Multi-period inventory MDP        via numpy value iteration

Run:
    python3 newsvendor.py   --lambda 50  --c 0.5  --p 1.0  --s 0.1
    python3 newsvendor.py   --multi  --lambda 20  --c 1.0  --K 10

Output (JSON on stdout) is consumed by validate-newsvendor.ts as a
cross-engine sanity check. Set NEWSVENDOR_PY=/path/to/python to override
the binary used by the TS validator; default is `python3` on $PATH.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def poisson_pmf(lam: float, d_max: int) -> np.ndarray:
    pmf = np.zeros(d_max + 1)
    p = np.exp(-lam)
    pmf[0] = p
    for k in range(1, d_max + 1):
        p = p * lam / k
        pmf[k] = p
    pmf[d_max] += 1.0 - pmf.sum()  # absorb tail at d_max
    return pmf


def newsvendor_optimal(c: float, p: float, s: float, pmf: np.ndarray) -> dict:
    cu, co = p - c, c - s
    cr = cu / (cu + co)
    cdf = np.cumsum(pmf)
    q_star = int(np.searchsorted(cdf, cr, side="left"))
    if q_star >= len(pmf):
        q_star = len(pmf) - 1
    qs = np.arange(len(pmf))
    expected_profit = []
    for q in qs:
        sold = np.minimum(q, qs)
        leftover = np.maximum(0, q - qs)
        expected_profit.append(np.sum(pmf * (p * sold + s * leftover - c * q)))
    return {
        "q_star": q_star,
        "critical_ratio": float(cr),
        "expected_profit_at_qstar": float(expected_profit[q_star]),
    }


def inventory_mdp(
    pmf: np.ndarray,
    x_max: int,
    a_max: int,
    c: float,
    K: float,
    p: float,
    h: float,
    L: float,
    gamma: float,
    tol: float = 1e-9,
    max_iter: int = 5000,
) -> dict:
    n_states = x_max + 1
    # Pre-build expected one-step rewards and transition kernel.
    # Lost-sales: x' = max(0, x + a - D).
    R = np.full((n_states, a_max + 1), -np.inf)  # -inf for illegal actions
    T = [[None] * (a_max + 1) for _ in range(n_states)]
    for x in range(n_states):
        for a in range(a_max + 1):
            if x + a > x_max:
                continue
            after = x + a
            d = np.arange(len(pmf))
            sold = np.minimum(after, d)
            leftover = np.maximum(0, after - d)
            lost = np.maximum(0, d - after)
            r = (p * sold - c * a - (K if a > 0 else 0.0)
                 - h * leftover - L * lost)
            R[x, a] = float(np.sum(pmf * r))
            # Successor distribution: nextX = leftover (clipped at x_max).
            nx = np.clip(leftover, 0, x_max)
            tk = np.zeros(n_states)
            np.add.at(tk, nx, pmf)
            T[x][a] = tk

    V = np.zeros(n_states)
    iters = 0
    for it in range(max_iter):
        Vn = np.full(n_states, -np.inf)
        for x in range(n_states):
            best = -np.inf
            for a in range(a_max + 1):
                if not np.isfinite(R[x, a]):
                    continue
                q = R[x, a] + gamma * float(np.dot(T[x][a], V))
                if q > best:
                    best = q
            Vn[x] = best
        delta = float(np.max(np.abs(Vn - V)))
        V = Vn
        iters = it + 1
        if delta < tol:
            break

    policy = np.full(n_states, -1, dtype=int)
    for x in range(n_states):
        best = -np.inf
        bestA = -1
        for a in range(a_max + 1):
            if not np.isfinite(R[x, a]):
                continue
            q = R[x, a] + gamma * float(np.dot(T[x][a], V))
            if q > best:
                best = q
                bestA = a
        policy[x] = bestA

    return {
        "iterations": iters,
        "final_delta": float(delta),
        "V": V.tolist(),
        "policy": policy.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda", dest="lam", type=float, default=50.0)
    ap.add_argument("--d-max", type=int, default=None)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--s", type=float, default=0.1)
    ap.add_argument("--multi", action="store_true",
                    help="Multi-period inventory MDP instead of newsvendor.")
    ap.add_argument("--x-max", type=int, default=None)
    ap.add_argument("--a-max", type=int, default=None)
    ap.add_argument("--K", type=float, default=0.0)
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--L", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.95)
    args = ap.parse_args()

    d_max = args.d_max if args.d_max is not None else int(args.lam * 2.5 + 1)
    pmf = poisson_pmf(args.lam, d_max)
    payload: dict = {
        "lambda": args.lam, "d_max": d_max,
        "c": args.c, "p": args.p, "s": args.s,
    }
    if not args.multi:
        result = newsvendor_optimal(args.c, args.p, args.s, pmf)
        payload.update({"newsvendor": result})
    else:
        x_max = args.x_max if args.x_max is not None else int(args.lam * 2.5 + 1)
        a_max = args.a_max if args.a_max is not None else x_max
        result = inventory_mdp(pmf, x_max, a_max,
                               args.c, args.K, args.p, args.h, args.L, args.gamma)
        payload.update({
            "x_max": x_max, "a_max": a_max,
            "K": args.K, "h": args.h, "L": args.L, "gamma": args.gamma,
            "inventory_mdp": {
                "iterations": result["iterations"],
                "final_delta": result["final_delta"],
                "V_at_zero": result["V"][0],
                "policy_first_20": result["policy"][:20],
            },
        })
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
