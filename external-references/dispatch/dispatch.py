#!/usr/bin/env python3
"""
external-references/dispatch/dispatch.py — scipy reference for the
multi-class parallel-server fluid LP relaxation, plus a Hungarian
assignment for the deterministic-batch baseline.

Two sub-tools:

    fluid-lp     ← read JSON {M, K, lambda, p, mu} from stdin, write
                   {x, t_star, status} to stdout. x is the K×M assignment
                   fraction matrix. Solved with scipy.optimize.linprog
                   (HiGHS / interior-point / simplex selectable).

    assign       ← read JSON {cost: K×M cost matrix} from stdin and write
                   the optimal min-cost assignment (Hungarian algorithm
                   from scipy.optimize.linear_sum_assignment).

Both are just thin Python wrappers; the actual TS pipeline calls these
ONLY for cross-validation, not as a runtime dependency.

USAGE
-----
    echo '{"M":2,"K":2,"lambda":1.6,"p":[0.6,0.4],
           "mu":[[2.0,0.8],[0.8,2.0]]}' \\
        | python3 dispatch.py fluid-lp --method highs

    echo '{"cost":[[1,4],[3,2]]}' \\
        | python3 dispatch.py assign
"""

import argparse
import json
import sys
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment


def fluid_lp(payload, method='highs'):
    """Solve   min t   s.t.  Σ_m x_{c,m} = 1  ∀c
                              λ Σ_c p_c x_{c,m} / μ_{c,m} ≤ t  ∀m
                              x ≥ 0

    Returns a dict {x, t_star, status, message}.
    """
    M = payload['M']
    K = payload['K']
    lam = payload['lambda']
    p = payload['p']
    mu = payload['mu']
    n_vars = K * M + 1   # last var is t
    c = np.zeros(n_vars); c[-1] = 1.0
    A_eq = np.zeros((K, n_vars))
    b_eq = np.ones(K)
    for k in range(K):
        for m in range(M):
            A_eq[k, k * M + m] = 1.0
    A_ub = np.zeros((M, n_vars))
    b_ub = np.zeros(M)
    for m in range(M):
        for k in range(K):
            A_ub[m, k * M + m] = lam * p[k] / max(1e-12, mu[k][m])
        A_ub[m, -1] = -1.0
    bounds = [(0, None)] * n_vars
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=method)
    x_mat = []
    if res.x is not None:
        for k in range(K):
            x_mat.append([float(res.x[k * M + m]) for m in range(M)])
        t_star = float(res.x[-1])
    else:
        t_star = float('nan')
    return {
        'x': x_mat,
        't_star': t_star,
        'status': 'optimal' if res.status == 0 else f'scipy-status-{res.status}',
        'message': str(res.message),
        'method': method,
    }


def assign(payload):
    """Min-cost assignment via Hungarian algorithm."""
    cost = np.array(payload['cost'])
    rows, cols = linear_sum_assignment(cost)
    total = float(cost[rows, cols].sum())
    pairs = [[int(r), int(c)] for r, c in zip(rows, cols)]
    return {'pairs': pairs, 'total_cost': total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['fluid-lp', 'assign'])
    parser.add_argument('--method', default='highs',
                        choices=['highs', 'highs-ds', 'highs-ipm', 'simplex', 'interior-point'])
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    if args.cmd == 'fluid-lp':
        result = fluid_lp(payload, method=args.method)
    elif args.cmd == 'assign':
        result = assign(payload)
    else:
        raise ValueError(f'unknown command {args.cmd}')
    json.dump(result, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
