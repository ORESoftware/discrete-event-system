#!/usr/bin/env python3
"""
External LP-solver bridge for the DES framework.

Reads a JSON LP description from stdin in the shape
{
  "lp": {
    "sense": "max" | "min",
    "c":    [n floats],
    "A_ub": [m lists of n floats],   // optional
    "b_ub": [m floats],
    "A_eq": [p lists of n floats],   // optional
    "b_eq": [p floats],
    "lb":   [n floats or null],      // null = −infinity
    "ub":   [n floats or null]       // null = +infinity
  },
  "method": "highs" | "highs-ds" | "highs-ipm" | "simplex" | "interior-point"
}

Calls scipy.optimize.linprog with the chosen method, returns the solution
as JSON on stdout in the shape

{
  "status": "optimal" | "infeasible" | "unbounded" | "iter-limit" | "numerical-error",
  "x":            [n floats],     // empty if not optimal
  "objective":    float | null,   // c^T x in the original sense
  "dualUB":       [m floats],
  "dualEQ":       [p floats],
  "reducedCosts": [n floats],
  "iters":        int,
  "message":      str
}

Usage
-----

By default invoked from TypeScript via `child_process.spawnSync`. The path
is resolved relative to the repository root by `lp.ts`. To run manually:

    echo '{"lp":{"sense":"max","c":[3,2],"A_ub":[[1,1]],"b_ub":[4]},"method":"highs"}' \
        | python3 external-references/lp/lp_solve.py --method highs

Required: scipy ≥ 1.6 (for HiGHS), numpy.

Solver methods (passed to scipy.optimize.linprog):
  - highs       — default; HiGHS will choose simplex or IPM internally
  - highs-ds    — HiGHS dual simplex
  - highs-ipm   — HiGHS interior-point
  - simplex     — legacy scipy simplex (deprecated upstream but kept for parity)
  - interior-point — legacy scipy interior-point

This script is a thin shim with no business logic; the LP is built and
interpreted entirely on the TypeScript side.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def _to_float(x):
    """Convert scalar to float, mapping None ↦ NaN for JSON."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _to_list(arr):
    """Convert a numpy array (or None) to a JSON-friendly list."""
    if arr is None:
        return []
    try:
        return [float(v) for v in arr.tolist()]
    except AttributeError:
        return [float(v) for v in arr]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Solve an LP via scipy.optimize.linprog.')
    parser.add_argument('--method', default='highs',
                        choices=['highs', 'highs-ds', 'highs-ipm', 'simplex', 'interior-point'],
                        help='scipy linprog method')
    args = parser.parse_args(argv)

    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError as exc:
        json.dump({
            'status': 'numerical-error',
            'x': [], 'objective': None,
            'dualUB': [], 'dualEQ': [], 'reducedCosts': [],
            'iters': 0,
            'message': f'scipy/numpy not available: {exc}',
        }, sys.stdout)
        return 0

    payload = json.loads(sys.stdin.read())
    lp = payload['lp']
    method = payload.get('method', args.method)

    sense = lp.get('sense', 'min')
    c = np.asarray(lp['c'], dtype=float)
    if sense == 'max':
        c_min = -c            # scipy minimises; flip for max
    else:
        c_min = c

    A_ub = np.asarray(lp['A_ub'], dtype=float) if lp.get('A_ub') else None
    b_ub = np.asarray(lp['b_ub'], dtype=float) if lp.get('b_ub') else None
    A_eq = np.asarray(lp['A_eq'], dtype=float) if lp.get('A_eq') else None
    b_eq = np.asarray(lp['b_eq'], dtype=float) if lp.get('b_eq') else None

    n = c.shape[0]
    lb_in = lp.get('lb')
    ub_in = lp.get('ub')
    if lb_in is None and ub_in is None:
        bounds = [(0, None)] * n          # scipy default
    else:
        lb_in = lb_in or [0] * n
        ub_in = ub_in or [None] * n
        bounds = [(lb_in[i], ub_in[i]) for i in range(n)]

    try:
        result = linprog(c_min, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method=method)
    except Exception as exc:
        json.dump({
            'status': 'numerical-error', 'x': [], 'objective': None,
            'dualUB': [], 'dualEQ': [], 'reducedCosts': [],
            'iters': 0,
            'message': f'linprog raised: {exc}',
        }, sys.stdout)
        return 0

    # Map scipy status codes to our enum.
    #   0 = optimal, 1 = iter limit, 2 = infeasible, 3 = unbounded, 4 = numerical
    status_map = {0: 'optimal', 1: 'iter-limit', 2: 'infeasible',
                  3: 'unbounded', 4: 'numerical-error'}
    status = status_map.get(int(result.status), 'numerical-error')

    x = _to_list(result.x) if status == 'optimal' else []
    if status == 'optimal':
        # Convert objective back to original sense.
        obj = float(c_min @ result.x)
        if sense == 'max':
            obj = -obj
    else:
        obj = None

    # Duals are exposed by HiGHS as result.ineqlin.marginals / eqlin.marginals.
    dual_ub, dual_eq, rc = [], [], []
    if hasattr(result, 'ineqlin') and result.ineqlin is not None:
        if getattr(result.ineqlin, 'marginals', None) is not None:
            dual_ub = _to_list(result.ineqlin.marginals)
            # scipy returns duals in min-form; flip sign if user asked for max.
            if sense == 'max':
                dual_ub = [-v for v in dual_ub]
    if hasattr(result, 'eqlin') and result.eqlin is not None:
        if getattr(result.eqlin, 'marginals', None) is not None:
            dual_eq = _to_list(result.eqlin.marginals)
            if sense == 'max':
                dual_eq = [-v for v in dual_eq]
    if hasattr(result, 'lower') and result.lower is not None:
        if getattr(result.lower, 'marginals', None) is not None:
            rc = _to_list(result.lower.marginals)
            if sense == 'max':
                rc = [-v for v in rc]

    json.dump({
        'status': status,
        'x': x,
        'objective': _to_float(obj),
        'dualUB': dual_ub,
        'dualEQ': dual_eq,
        'reducedCosts': rc,
        'iters': int(getattr(result, 'nit', 0) or 0),
        'message': str(getattr(result, 'message', '')),
    }, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
