#!/usr/bin/env python3
"""
external-references/soccer/soccer.py — scipy reference for the
multi-period 7v7 player rotation problem.

Two sub-tools:

    rotation-lp   ← read JSON {affinity, numPeriods, benchSize} from stdin,
                    write {x, objective, status} to stdout. The LP solves
                    exactly the same multi-period assignment relaxation
                    that the TS pipeline solves; numerical objectives must
                    match to ~1e-6.

    period-assign ← single-period bipartite max-weight assignment via
                    scipy.optimize.linear_sum_assignment (Hungarian).
                    Reads JSON {weights: P×K} where P ≤ 12 is candidate
                    on-field players (one row per player) and K = 7
                    positions; writes {rows, total} to stdout.

USAGE
-----
    cat <<EOF | python3 soccer.py rotation-lp --method highs-ds
    {"affinity": [[[…12 players…], …], …], "numPeriods": 4, "benchSize": 5}
    EOF

    echo '{"weights": [[0.9, 0.6, 0.2, 0.1, 0.5, 0.4, 0.3], …]}' \\
        | python3 soccer.py period-assign
"""
import argparse
import json
import sys

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment


def rotation_lp(payload, method='highs'):
    aff = np.array(payload['affinity'])  # shape (P, K, T)
    P, K, T = aff.shape
    benchSize = payload['benchSize']
    n_vars = P * K * T

    def idx(p, pos, t):
        return (p * K + pos) * T + t

    c = np.zeros(n_vars)
    for p in range(P):
        for pos in range(K):
            for t in range(T):
                c[idx(p, pos, t)] = -aff[p, pos, t]   # linprog minimises

    A_eq = []
    b_eq = []
    # Σ_p x_{p,pos,t} = 1   for each (pos, t)
    for pos in range(K):
        for t in range(T):
            row = np.zeros(n_vars)
            for p in range(P):
                row[idx(p, pos, t)] = 1.0
            A_eq.append(row); b_eq.append(1.0)

    A_ub = []
    b_ub = []
    # Σ_pos x_{p,pos,t} ≤ 1   for each (p, t)
    for p in range(P):
        for t in range(T):
            row = np.zeros(n_vars)
            for pos in range(K):
                row[idx(p, pos, t)] = 1.0
            A_ub.append(row); b_ub.append(1.0)
    # Σ_pos x_{p,pos,t} + Σ_pos x_{p,pos,t+1} ≥ 1  ⇔  -Σ - Σ ≤ -1
    for p in range(P):
        for t in range(T - 1):
            row = np.zeros(n_vars)
            for pos in range(K):
                row[idx(p, pos, t)]     -= 1.0
                row[idx(p, pos, t + 1)] -= 1.0
            A_ub.append(row); b_ub.append(-1.0)

    bounds = [(0, 1)] * n_vars
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method=method)
    if res.x is None:
        return {'status': f'scipy-status-{res.status}', 'message': str(res.message)}
    objective = -float(res.fun)   # we minimised the negated objective
    return {
        'status': 'optimal' if res.status == 0 else f'scipy-status-{res.status}',
        'objective': objective,
        'method': method,
    }


def period_assign(payload):
    w = np.array(payload['weights'])
    # linear_sum_assignment minimises cost; convert to max-weight by negating.
    rows, cols = linear_sum_assignment(-w)
    total = float(w[rows, cols].sum())
    return {
        'rows': [int(c) for c in cols],
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['rotation-lp', 'period-assign'])
    parser.add_argument('--method', default='highs',
                        choices=['highs', 'highs-ds', 'highs-ipm', 'simplex', 'interior-point'])
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    if args.cmd == 'rotation-lp':
        result = rotation_lp(payload, method=args.method)
    elif args.cmd == 'period-assign':
        result = period_assign(payload)
    else:
        raise ValueError(f'unknown command {args.cmd}')
    json.dump(result, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
