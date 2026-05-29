#!/usr/bin/env python3
"""
External reference for the series RLC step-response simulation.

For the framework's input config (R, L, C, V_step), this script produces:
  * ANALYTICAL underdamped step response in closed form.
  * scipy.integrate.solve_ivp with LSODA (rtol=1e-10, atol=1e-12) — the
    "ground-truth" numerical reference.

State equations:
    dI/dt   = (V_in - I*R - V_C) / L
    dV_C/dt = I / C

Closed-form analytical (R, L, C such that 4L > R^2*C ⇒ underdamped):
    α  = R / (2L)
    ω0 = 1 / sqrt(L*C)
    ωd = sqrt(ω0^2 - α^2)
    V_C(t) = V_step * (1 - exp(-α*t) * (cos(ωd*t) + (α/ωd)*sin(ωd*t)))
    I(t)   = V_step / (L*ωd) * exp(-α*t) * sin(ωd*t)

Output: out/external/electric-circuit/reference.json with traces sampled at
the SAME t grid as the framework's smallest-dt run (so element-wise
comparison is straightforward).

Usage (called by external-references/run-all.sh):
    python3 circuit.py
"""
import json
import math
import os
import pathlib
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IN_PATH = ROOT / "out" / "electric-circuit-framework.json"
OUT_DIR = ROOT / "out" / "external" / "electric-circuit"
OUT_PATH = OUT_DIR / "reference.json"


def analytical_underdamped(R: float, L: float, C: float, Vstep: float, t):
    """Closed-form V_C(t), I(t) for a series RLC underdamped step response."""
    t = np.asarray(t, dtype=np.float64)
    alpha = R / (2 * L)
    omega0 = 1.0 / math.sqrt(L * C)
    if alpha >= omega0:
        raise ValueError(
            f"Not underdamped: alpha={alpha} >= omega0={omega0}")
    omegad = math.sqrt(omega0 ** 2 - alpha ** 2)
    e = np.exp(-alpha * t)
    V_C = Vstep * (1.0 - e * (np.cos(omegad * t) + (alpha / omegad) * np.sin(omegad * t)))
    I   = Vstep / (L * omegad) * e * np.sin(omegad * t)
    return I, V_C


def scipy_lsoda(R: float, L: float, C: float, Vstep: float, t_eval):
    """Reference numerical integration via scipy LSODA."""
    def rhs(t, y):
        I, V_C = y
        V_in = Vstep if t >= 0 else 0.0
        return [(V_in - I * R - V_C) / L, I / C]
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [0.0, 0.0],
                    t_eval=t_eval, method='LSODA',
                    rtol=1e-10, atol=1e-12, max_step=0.01)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.y[0].tolist(), sol.y[1].tolist()


def main() -> int:
    if not IN_PATH.exists():
        print(f"[circuit] input {IN_PATH} not found; "
              f"run `node dist/des/main-electric-circuit.js` first.", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open() as f:
        payload = json.load(f)

    cfg = payload["config"]
    R, L, C = cfg["R"], cfg["L"], cfg["C"]
    Vstep = cfg["Vstep"]
    T = cfg["T"]

    # Sample the analytical & scipy traces at the dt used by the SMALLEST-dt
    # framework run (so we have a high-resolution reference grid).
    smallest = min(payload["sweep"], key=lambda s: s["dt"])
    grid_dt = smallest["dt"]
    n_grid = int(round(T / grid_dt))
    t_grid = np.arange(n_grid + 1, dtype=np.float64) * grid_dt

    I_ana, V_ana = analytical_underdamped(R, L, C, Vstep, t_grid)
    I_sci, V_sci = scipy_lsoda(R, L, C, Vstep, t_grid)

    # Sanity: analytical and scipy must agree to ~1e-8 (limited by LSODA).
    err_V = float(np.max(np.abs(np.array(V_sci) - V_ana)))
    err_I = float(np.max(np.abs(np.array(I_sci) - I_ana)))
    print(f"[circuit] T={T}s  grid dt={grid_dt}  N={n_grid+1} points")
    print(f"[circuit] LSODA vs analytical: max|V_C err|={err_V:.3e}  max|I err|={err_I:.3e}")

    out = {
        "tool": "scipy.solve_ivp(LSODA) + analytical underdamped",
        "config": cfg,
        "grid_dt": grid_dt,
        "t":   t_grid.tolist(),
        "V_C_analytical": V_ana.tolist(),
        "I_analytical":   I_ana.tolist(),
        "V_C_scipy": V_sci,
        "I_scipy":   I_sci,
        "self_check": {"max_abs_V_C": err_V, "max_abs_I": err_I},
    }
    with OUT_PATH.open("w") as f:
        json.dump(out, f)
    print(f"[circuit] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
