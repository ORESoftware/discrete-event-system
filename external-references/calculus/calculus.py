#!/usr/bin/env python3
"""External reference solvers for ODE / PDE / quadrature, used to cross-validate
the TypeScript station-network solvers in main-calculus.ts.

Required: numpy, scipy, sympy. Install with  pip install numpy scipy sympy.

Usage:

  PROBLEM=ode python3 calculus.py
  PROBLEM=pde-heat python3 calculus.py
  PROBLEM=pde-wave python3 calculus.py
  PROBLEM=poisson  python3 calculus.py
  PROBLEM=quad     python3 calculus.py

All paths print bit-comparable JSON to stdout (last line) for the TS
validator runner to read.
"""

from __future__ import annotations
import json
import math
import os
import sys

try:
    import numpy as np
    from scipy.integrate import solve_ivp, quad
except Exception as exc:                                       # pragma: no cover
    sys.stderr.write(f"# numpy/scipy not available: {exc}\n")
    sys.exit(1)


def problem_ode() -> dict:
    """Reference ODE: y'' + ω² y = 0, state [y, v], y(0)=1, v(0)=0."""
    omega = float(os.environ.get('OMEGA', '1.0'))
    t1 = float(os.environ.get('T_END', str(2 * math.pi)))
    y0 = [1.0, 0.0]

    def rhs(t, y):
        return [y[1], -omega * omega * y[0]]

    sol = solve_ivp(rhs, (0.0, t1), y0, method='DOP853',
                    rtol=1e-13, atol=1e-15,
                    t_eval=[t1])
    return {
        'problem': 'ode',
        'omega': omega,
        't1': t1,
        'y_at_t1': float(sol.y[0, -1]),
        'v_at_t1': float(sol.y[1, -1]),
        'analytical_y': math.cos(omega * t1),
        'analytical_v': -omega * math.sin(omega * t1),
    }


def problem_pde_heat() -> dict:
    """1D heat equation u_t = α u_xx, [0, 1], u(0,t)=u(1,t)=0, u(x,0)=sin(πx)."""
    alpha = float(os.environ.get('ALPHA', '0.1'))
    T = float(os.environ.get('T_END', '0.5'))
    N = int(os.environ.get('N', '51'))
    dx = 1.0 / (N - 1)
    xs = np.linspace(0, 1, N)

    def rhs(t, u):
        du = np.zeros_like(u)
        du[1:-1] = alpha * (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx * dx)
        return du

    u0 = np.sin(np.pi * xs)
    sol = solve_ivp(rhs, (0.0, T), u0, method='LSODA',
                    rtol=1e-9, atol=1e-12, t_eval=[T])
    decay = math.exp(-alpha * math.pi * math.pi * T)
    analytical = decay * np.sin(np.pi * xs)
    err = float(np.max(np.abs(sol.y[:, -1] - analytical)))
    return {
        'problem': 'pde-heat',
        'alpha': alpha, 'T': T, 'N': N,
        'final_max_err_vs_analytical': err,
        'final_peak': float(sol.y[N // 2, -1]),
        'expected_peak': decay,
        'final_values': sol.y[:, -1].tolist(),
        'xs': xs.tolist(),
    }


def problem_pde_wave() -> dict:
    """1D wave u_tt = c² u_xx, [0,1], u(0)=u(1)=0, u(x,0)=sin(πx), v(x,0)=0."""
    c = float(os.environ.get('C', '1.0'))
    T = float(os.environ.get('T_END', '0.5'))
    N = int(os.environ.get('N', '51'))
    dx = 1.0 / (N - 1)
    xs = np.linspace(0, 1, N)

    def rhs(t, y):
        u = y[:N]; v = y[N:]
        du = v.copy()
        dv = np.zeros_like(v)
        dv[1:-1] = c * c * (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx * dx)
        return np.concatenate([du, dv])

    u0 = np.sin(np.pi * xs)
    v0 = np.zeros(N)
    y0 = np.concatenate([u0, v0])
    sol = solve_ivp(rhs, (0.0, T), y0, method='RK45',
                    rtol=1e-9, atol=1e-12, t_eval=[T])
    analytical = np.sin(np.pi * xs) * math.cos(math.pi * c * T)
    err = float(np.max(np.abs(sol.y[:N, -1] - analytical)))
    return {
        'problem': 'pde-wave',
        'c': c, 'T': T, 'N': N,
        'final_max_err_vs_analytical': err,
        'final_values': sol.y[:N, -1].tolist(),
        'xs': xs.tolist(),
    }


def problem_poisson() -> dict:
    """2D Poisson on Nx×Ny grid via Jacobi iteration to convergence."""
    N = int(os.environ.get('N', '41'))
    tol = float(os.environ.get('TOL', '1e-8'))
    xs = np.linspace(0, 1, N)
    ys = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    rho = 2 * math.pi**2 * np.sin(math.pi * X) * np.sin(math.pi * Y)
    dx = 1.0 / (N - 1)
    dy = dx
    u = np.zeros_like(rho)
    # Jacobi.
    iters = 0
    while True:
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (
            (dy * dy) * (u[1:-1, 2:] + u[1:-1, :-2])
            + (dx * dx) * (u[2:, 1:-1] + u[:-2, 1:-1])
            + (dx * dx * dy * dy) * rho[1:-1, 1:-1]
        ) / (2 * (dx * dx + dy * dy))
        delta = float(np.max(np.abs(u_new - u)))
        u = u_new
        iters += 1
        if delta < tol or iters > 100000:
            break
    analytical = np.sin(math.pi * X) * np.sin(math.pi * Y)
    err = float(np.max(np.abs(u - analytical)))
    return {
        'problem': 'poisson',
        'N': N,
        'tol': tol,
        'iterations': iters,
        'final_delta': delta,
        'max_err_vs_analytical': err,
        'centre_value': float(u[N // 2, N // 2]),
        'xs': xs.tolist(),
    }


def problem_quad() -> dict:
    """Quadrature reference: ∫_0^π (x² sin(x) + e^{-x}) dx via scipy.integrate.quad."""
    a = float(os.environ.get('A', '0'))
    b = float(os.environ.get('B', str(math.pi)))
    f = lambda x: x * x * math.sin(x) + math.exp(-x)
    val, est = quad(f, a, b, epsabs=1e-15, epsrel=1e-15)
    return {
        'problem': 'quad',
        'a': a, 'b': b,
        'value': val,
        'abs_err_estimate': est,
    }


def main() -> int:
    problem = os.environ.get('PROBLEM', 'ode')
    handlers = {
        'ode': problem_ode,
        'pde-heat': problem_pde_heat,
        'pde-wave': problem_pde_wave,
        'poisson': problem_poisson,
        'quad': problem_quad,
    }
    if problem not in handlers:
        sys.stderr.write(f"unknown PROBLEM='{problem}'.  options: {list(handlers)}\n")
        return 1
    result = handlers[problem]()
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
